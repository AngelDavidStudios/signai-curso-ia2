"""Generación automática de rondas para el ciclo automático (TT-14 · spec FR-019, research R-16; TT-22).

La primera ronda es la de research R-07 de TT-22: la línea base (la configuración del modelo de la
versión 1, con `EPOCAS_BASE` épocas) y cinco variantes de aumento de datos. Cada ronda
siguiente varía **una dimensión** de la mejor configuración (la elegible de mejor media de Top-1 o, si
aún no hay elegibles, la de mejor media), recorriendo un vecindario fijo de valores, sin repetir
configuraciones del barrido y sin pasar del límite de parámetros. Si alrededor de la mejor no queda
nada nuevo, sigue con la segunda, y así. La ronda se escribe como TOML de contracts/ronda.md, de modo
que queda declarada y versionada antes de lanzarse, igual que una escrita a mano.
"""

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from . import CLASES, LIMITE_PARAMETROS
from .configuracion import Configuracion, configuracion_desde_dict
from .errores import RondaNoValida
from .modelo import KPoseFormer, contar_parametros
from .ronda import Ronda
from .seleccion import Agregado, clave_configuracion

POR_RONDA = 6
EPOCAS_BASE = 60
FORMA_BASE = {"capas": 4, "dimension": 98, "cabezas": 7, "intermedia": 256}
ENTRENAMIENTO_BASE = {
    "optimizador": "adamw", "tasa": 1e-3, "programacion": "coseno", "regularizacion": 0.05, "dropout": 0.2,
    "recorte_gradiente": 1.0, "lote": 32, "suavizado_etiquetas": 0.0,
}
AUMENTO_POR_DEFECTO = {
    "desplazamiento": {"probabilidad": 0.5, "intensidad": 0.05},
    "escala": {"probabilidad": 0.5, "intensidad": 0.1},
    "rotacion": {"probabilidad": 0.5, "intensidad": 0.15},
    "ocultacion": {"probabilidad": 0.3, "intensidad": 0.1},
    "velocidad": {"probabilidad": 0.5, "intensidad": 0.2},
    "recorte_temporal": {"probabilidad": 0.5, "intensidad": 0.7},
}


@dataclass(frozen=True)
class Variacion:
    """Una dimensión del vecindario y sus valores ordenados; se prueban los dos vecinos del actual."""

    seccion: str
    clave: str
    valores: tuple


# Cada entrada es una `Variacion` o «aumento.<tipo>» (poner o quitar ese aumento con sus valores por defecto).
VECINDARIO: tuple = (
    Variacion("entrenamiento", "tasa", (3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3)),
    Variacion("entrenamiento", "dropout", (0.0, 0.05, 0.1, 0.2, 0.3)),
    Variacion("entrenamiento", "regularizacion", (0.0, 0.01, 0.05, 0.1, 0.2)),
    "aumento.desplazamiento",
    "aumento.rotacion",
    Variacion("forma", "capas", (2, 3, 4, 5, 6)),
    "aumento.escala",
    Variacion("entrenamiento", "epocas", (60, 100, 150)),
    "aumento.recorte_temporal",
    Variacion("forma", "intermedia", (128, 256, 384, 512)),
    "aumento.ocultacion",
    Variacion("forma", "dimension", ((64, 8), (98, 7), (128, 8), (192, 8))),
    "aumento.velocidad",
    Variacion("entrenamiento", "lote", (16, 32, 64)),
    Variacion("entrenamiento", "suavizado_etiquetas", (0.0, 0.05, 0.1)),
)


# --- TOML ----------------------------------------------------------------------------------------


def _valor(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    return json.dumps(v, ensure_ascii=False)


def _tabla(valores: dict) -> str:
    return "{ " + ", ".join(f"{k} = {_valor(v)}" for k, v in valores.items()) + " }"


def _a_dict(cfg: Configuracion) -> dict:
    return {
        "forma": asdict(cfg.forma),
        "entrenamiento": asdict(cfg.entrenamiento),
        "aumento": {tipo: asdict(a) for tipo, a in cfg.aumento},
    }


def _toml(barrido: str, numero: int, comentario: str, base: dict, configuraciones: list[tuple[str, dict]]) -> str:
    lineas = [f"# {comentario}", f'barrido = "{barrido}"', f"ronda = {numero}", "", "[base.forma]"]
    lineas += [f"{k} = {_valor(v)}" for k, v in base["forma"].items()]
    lineas += ["", "[base.entrenamiento]"] + [f"{k} = {_valor(v)}" for k, v in base["entrenamiento"].items()]
    for nombre, datos in configuraciones:
        lineas += ["", "[[configuracion]]", f'nombre = "{nombre}"']
        for seccion in ("forma", "entrenamiento"):
            cambios = {k: v for k, v in datos[seccion].items() if base[seccion].get(k) != v}
            if cambios:
                lineas.append(f"{seccion} = {_tabla(cambios)}")
        lineas += [f"aumento.{tipo} = {_tabla(valores)}" for tipo, valores in datos.get("aumento", {}).items()]
    return "\n".join(lineas) + "\n"


# --- Primera ronda -------------------------------------------------------------------------------


def generar_primera_ronda(barrido: str) -> str:
    base = {"forma": dict(FORMA_BASE), "entrenamiento": {**ENTRENAMIENTO_BASE, "epocas": EPOCAS_BASE}}

    def variante(entrenamiento: dict | None = None, aumento: dict | None = None) -> dict:
        return {"forma": base["forma"], "entrenamiento": {**base["entrenamiento"], **(entrenamiento or {})},
                "aumento": aumento or {}}

    def con(*tipos: str) -> dict:
        return variante(aumento={tipo: AUMENTO_POR_DEFECTO[tipo] for tipo in tipos})

    configuraciones = [
        ("r01-linea-base", variante()),
        ("r01-espacial", con("desplazamiento", "escala")),
        ("r01-rotacion", con("rotacion")),
        ("r01-temporal", con("velocidad", "recorte_temporal")),
        ("r01-ocultacion", con("ocultacion")),
        ("r01-todo", con(*AUMENTO_POR_DEFECTO)),
    ]
    return _toml(barrido, 1, f"Ronda 1 de {barrido}, generada por `signai-reconocedor automatico` "
                 f"(TT-22 research R-07: línea base de la versión 1 y aumentos, {EPOCAS_BASE} épocas).",
                 base, configuraciones)


# --- Rondas siguientes ---------------------------------------------------------------------------


def _slug(valor) -> str:
    numero = valor[0] if isinstance(valor, tuple) else valor
    return f"{numero:g}".replace(".", "p").replace("-", "m").replace("+", "")


def _vecinos(datos: dict, variacion) -> list[tuple[str, dict]]:
    if isinstance(variacion, str):
        tipo = variacion.removeprefix("aumento.")
        nuevo = {**datos, "aumento": dict(datos["aumento"])}
        if tipo in nuevo["aumento"]:
            del nuevo["aumento"][tipo]
            return [(f"sin-{tipo.replace('_', '-')}", nuevo)]
        nuevo["aumento"][tipo] = AUMENTO_POR_DEFECTO[tipo]
        return [(f"con-{tipo.replace('_', '-')}", nuevo)]
    seccion, clave, valores = variacion.seccion, variacion.clave, variacion.valores
    actual = datos[seccion][clave]
    escalares = [v[0] if isinstance(v, tuple) else v for v in valores]
    if actual in escalares:
        i = escalares.index(actual)
        elegidos = [valores[j] for j in (i - 1, i + 1) if 0 <= j < len(valores)]
    else:
        elegidos = sorted(valores, key=lambda v: abs((v[0] if isinstance(v, tuple) else v) - actual))[:2]
    resultado = []
    for valor in elegidos:
        nuevo = {**datos, seccion: dict(datos[seccion])}
        if clave == "dimension":
            nuevo[seccion]["dimension"], nuevo[seccion]["cabezas"] = valor
        else:
            nuevo[seccion][clave] = valor
        resultado.append((f"{clave.replace('_', '-')}-{_slug(valor)}", nuevo))
    return resultado


def _parametros(cfg: Configuracion) -> int:
    return contar_parametros(KPoseFormer(cfg.forma, clases=CLASES))


def generar_siguiente_ronda(
    barrido: str,
    numero: int,
    rondas: tuple[Ronda, ...],
    agregados: list[Agregado],
    vecindario: tuple | None = None,
    por_ronda: int = POR_RONDA,
    limite: int = LIMITE_PARAMETROS,
) -> str | None:
    """El TOML de la ronda `numero`, o `None` si alrededor de las configuraciones ya no queda nada nuevo."""
    vecindario = VECINDARIO if vecindario is None else vecindario
    declaradas = {cfg.nombre: cfg for r in rondas for cfg in r.configuraciones}
    vistas = {cfg.valores() for cfg in declaradas.values()}
    ranking = sorted(agregados, key=lambda a: (not a.elegible, clave_configuracion(a)))
    # Cada dimensión aporta hasta 2 vecinos: avanzar `por_ronda // 2` dimensiones por ronda hace que
    # rondas seguidas exploren dimensiones nuevas en lugar de repetir las primeras.
    giro = ((numero - 2) * max(1, por_ronda // 2)) % len(vecindario) if vecindario else 0
    ordenado = vecindario[giro:] + vecindario[:giro]
    nuevas: list[tuple[str, dict]] = []
    for agregado in ranking:
        datos = _a_dict(declaradas[agregado.configuracion])
        for variacion in ordenado:
            for etiqueta, candidato in _vecinos(datos, variacion):
                nombre = f"r{numero:02d}-{etiqueta}"
                while nombre in {n for n, _ in nuevas}:
                    nombre += "-b"
                try:
                    cfg = configuracion_desde_dict(nombre, candidato)
                except RondaNoValida:
                    continue
                if cfg.valores() in vistas or _parametros(cfg) > limite:
                    continue
                vistas.add(cfg.valores())
                nuevas.append((nombre, candidato))
                if len(nuevas) == por_ronda:
                    break
            if len(nuevas) == por_ronda:
                break
        if len(nuevas) == por_ronda:
            break
    if not nuevas:
        return None
    mejor = ranking[0]
    base = _a_dict(declaradas[mejor.configuracion])
    motivo = "la mejor configuración elegible" if mejor.elegible else "la de mejor media; aún no hay elegibles"
    comentario = (f"Ronda {numero} de {barrido}, generada por `signai-reconocedor automatico` a partir de "
                  f"{mejor.configuracion} ({motivo}).")
    return _toml(barrido, numero, comentario, base, nuevas)


# --- Efectos del ciclo ---------------------------------------------------------------------------


def confirmar_con_git(raiz: Path, ruta: Path, mensaje: str) -> None:
    """Commit local solo del fichero de la ronda, sin push y sin líneas de coautoría."""
    for argumentos in (("add", "--", str(ruta)), ("commit", "-q", "-m", mensaje, "--", str(ruta))):
        subprocess.run(["git", "-C", str(raiz), *argumentos], check=True, capture_output=True, text=True)


def preparar_copia() -> Callable[[], int]:
    """Pide una vez la clave de la copia de TT-21 y devuelve la copia entre rondas.

    `signai-copia actualizar` pide la clave en papel cada vez; el ciclo la pide al empezar, la guarda
    solo en memoria mientras dura y la reutiliza en cada copia, para no quedarse esperando.
    """
    from ..copia import cli as copia_cli

    raiz = Path(os.environ.get("SIGNAI_DATA_ROOT", Path.home() / "signai-data"))
    clave = copia_cli.crear_contexto(raiz).clave_tecleada(
        "Clave de la copia (desde el papel); se usará en cada copia entre rondas: "
    )

    def copiar() -> int:
        contexto = copia_cli.crear_contexto(raiz)
        contexto.pedir_clave = lambda _mensaje: clave
        return copia_cli.main(["actualizar"], contexto=contexto)

    return copiar
