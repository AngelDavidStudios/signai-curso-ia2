"""CLI `signai-reconocedor` con el protocolo por rotación (TT-22; contracts/cli.md C-1'…C-6').

    signai-reconocedor corrida <barrido> <configuracion> --semilla N --rotacion {a,b} [--permitir-sucio]
    signai-reconocedor ronda <barrido> <numero> [--permitir-sucio]
    signai-reconocedor resumen <barrido>
    signai-reconocedor cerrar <barrido>
    signai-reconocedor candidato <barrido> [--permitir-sucio]
    signai-reconocedor automatico <barrido> [--max-rondas N] [--permitir-sucio]

Cada configuración se entrena con las 3 semillas en las 2 rotaciones de la partición versión 2; el
candidato es el **modelo final**, entrenado con los dos pliegues hasta las épocas elegidas, sin validar.
Las rondas se leen de `config/reconocedor/<barrido>/` y el registro es el MLflow local
(`$SIGNAI_DATA_ROOT/mlflow`). Ninguna orden lee la prueba. Código de salida: 0 bien, 1 error de los
datos o del estado, 2 uso incorrecto.
"""

import argparse
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from . import SEMILLAS, automatico
from ..codigo import codigo_permitido as _codigo_permitido
from ..copia.errores import ErrorCopia
from ..particion.errores import ErrorParticion
from ..particion.lectura import para_entrenamiento_final, para_seleccion
from .bucle import dispositivo, entrenar, entrenar_final
from .candidato import VERSIONAR, escribir_candidato
from .configuracion import Configuracion
from .datos import ROTACIONES, DatosEntrenamiento, DatosFinal, cargar_final, cargar_rotacion
from .errores import (
    ArbolSucio,
    BarridoCerrado,
    DatosDistintos,
    ErrorReconocedor,
    RondaModificada,
    RondaNoValida,
    SemillaNoPermitida,
    SinCandidato,
)
from .modelo import KPoseFormer, contar_parametros, guardar
from .registro import CorridaRegistrada, EstadoCorrida, Registro, RegistroMlflow
from .resumen import generar_resumen
from .ronda import Ronda, buscar_configuracion, leer_barrido, ruta_barrido
from .seleccion import METRICAS, Agregado, ResultadoRotacion, agregar, elegir_candidato, estancado, mejores_por_ronda

CLAVE_JIRA = "SP-372"


@dataclass
class Contexto:
    raiz: Path
    registro: Registro
    lector_seleccion: Callable
    lector_final: Callable
    dev: torch.device
    confirmar: Callable[[Path, Path, str], None] = automatico.confirmar_con_git
    copiar: Callable[[], int] = lambda: 0


def _raiz_repo() -> Path:
    return Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signai-reconocedor",
        description="Entrenamiento del reconocedor KPoseFormer (TT-14, TT-22): selección por rotación y modelo final.",
    )
    ordenes = parser.add_subparsers(dest="orden", required=True)
    corrida = ordenes.add_parser(
        "corrida", help="Lanza una corrida de selección: una configuración, una semilla y una rotación."
    )
    corrida.add_argument("barrido")
    corrida.add_argument("configuracion")
    corrida.add_argument("--semilla", type=int, required=True)
    corrida.add_argument("--rotacion", choices=ROTACIONES, required=True,
                         help="a: entrena con el pliegue A y valida con el B; b: al revés.")
    corrida.add_argument("--permitir-sucio", action="store_true", help="Solo para pruebas; queda etiquetado.")
    ronda = ordenes.add_parser("ronda", help="Lanza las corridas pendientes de una ronda.")
    ronda.add_argument("barrido")
    ronda.add_argument("numero", type=int)
    ronda.add_argument("--permitir-sucio", action="store_true", help="Solo para pruebas; queda etiquetado.")
    ciclo = ordenes.add_parser(
        "automatico", help="Recorre el barrido solo: genera, confirma y lanza rondas hasta el cierre y el candidato."
    )
    ciclo.add_argument("barrido")
    ciclo.add_argument("--max-rondas", type=int, default=None, help="Detenerse tras lanzar N rondas.")
    ciclo.add_argument("--permitir-sucio", action="store_true", help="Solo para pruebas; queda etiquetado.")
    for orden, ayuda in (
        ("resumen", "Regenera el resumen del barrido."),
        ("cerrar", "Cierra el barrido si se ha estancado."),
    ):
        ordenes.add_parser(orden, help=ayuda).add_argument("barrido")
    candidato = ordenes.add_parser(
        "candidato", help="Entrena el modelo final del barrido cerrado y lo escribe en modelos/reconocedor/."
    )
    candidato.add_argument("barrido")
    candidato.add_argument("--permitir-sucio", action="store_true", help="Solo para pruebas; queda etiquetado.")
    return parser


# --- Código y comprobaciones previas -------------------------------------------------------------


def codigo_permitido(raiz: Path, permitir_sucio: bool) -> tuple[str, bool]:
    """El commit que se registra; sin cambios sin confirmar, salvo `--permitir-sucio` (solo pruebas)."""
    return _codigo_permitido(
        raiz, permitir_sucio, ArbolSucio, "Confirma los cambios, incluida la ronda, y vuelve a lanzar."
    )


def mismos_datos(ctx: Contexto, barrido: str, datos: DatosEntrenamiento | DatosFinal) -> None:
    """Un barrido compara corridas sobre los mismos datos (caso límite de la spec)."""
    for corrida in ctx.registro.corridas(barrido):
        for clave, valor in datos.huellas.items():
            registrado = corrida.etiquetas.get(clave)
            if registrado is not None and registrado != valor:
                raise DatosDistintos(
                    f"{clave} es {valor[:12]}… pero {corrida.nombre} se entrenó con {registrado[:12]}…: un barrido "
                    "compara corridas sobre los mismos datos. Restaura los datos con `uv run dvc checkout` o abre "
                    "otro barrido."
                )


def _barrido_abierto(ctx: Contexto, barrido: str) -> None:
    if ctx.registro.estado_barrido(barrido).get("barrido.estado") == "cerrado":
        raise BarridoCerrado(
            f"El barrido {barrido} está cerrado: no admite corridas ni rondas nuevas. Abre otro barrido."
        )


def _ronda_sin_cambios(ctx: Contexto, ronda: Ronda) -> None:
    for corrida in ctx.registro.corridas(ronda.barrido):
        if corrida.ronda == ronda.numero and corrida.etiquetas.get("ronda.huella") != ronda.huella:
            raise RondaModificada(
                f"{ronda.ruta} cambió después de lanzarse (huella distinta de la registrada en {corrida.nombre}). "
                "Una ronda lanzada no se edita: restaura el fichero y declara los cambios en una ronda nueva."
            )


# --- Una corrida ---------------------------------------------------------------------------------


def _etiquetas(ronda: Ronda, cfg: Configuracion, datos: DatosEntrenamiento | DatosFinal, ctx: Contexto, codigo) -> dict:
    parametros = contar_parametros(KPoseFormer(cfg.forma, clases=len(datos.clases)))
    commit, sucio = codigo
    return {
        "barrido": ronda.barrido,
        "ronda": str(ronda.numero),
        "ronda.huella": ronda.huella,
        "configuracion": cfg.nombre,
        "codigo.commit": commit,
        "codigo.sucio": str(sucio).lower(),
        "dispositivo": ctx.dev.type,
        "modelo.parametros": str(parametros),
        "modelo.tamano_kb": f"{parametros * 2 / 1024:.1f}",
        **datos.huellas,
    }


def ejecutar_corrida(ctx: Contexto, ronda: Ronda, cfg: Configuracion, semilla: int, rotacion: str,
                     datos: DatosEntrenamiento, codigo: tuple[str, bool], extra: dict[str, str] | None = None) -> bool:
    """Una corrida de selección: entrena un pliegue, valida el otro y registra la curva de cada época."""
    nombre = f"{cfg.nombre}-s{semilla}-{rotacion}"
    id_corrida = ctx.registro.abrir(
        ronda.barrido, nombre, {**cfg.aplanar(), "semilla": semilla},
        {**_etiquetas(ronda, cfg, datos, ctx, codigo), "fase": "seleccion", "rotacion": rotacion, **(extra or {})},
    )
    print(f"→ {nombre} (ronda {ronda.numero}, {ctx.dev.type})", flush=True)
    try:
        resultado = entrenar(datos, cfg, semilla, lambda e, m: ctx.registro.epoca(id_corrida, e, m), ctx.dev)
    except BaseException:
        ctx.registro.interrumpir(id_corrida)
        raise
    if resultado.estado is not EstadoCorrida.TERMINADA:
        ctx.registro.fallar(id_corrida, resultado.motivo)
        print(f"✗ {nombre}: {resultado.motivo}", file=sys.stderr)
        return False
    with tempfile.TemporaryDirectory() as temporal:
        por_clase = Path(temporal) / "por_clase.json"
        por_clase.write_text(json.dumps(dict(zip(datos.clases, resultado.por_clase)), ensure_ascii=False), encoding="utf-8")
        metricas = {f"elegida.{k}": v for k, v in resultado.metricas.items()}
        ctx.registro.terminar(
            id_corrida, {**metricas, "elegida.epoca": resultado.epoca_elegida, "duracion_s": resultado.duracion_s},
            [por_clase], {},
        )
    m = resultado.metricas
    print(
        f"✓ {nombre}: época elegida {resultado.epoca_elegida} · Top-1 {m['top1']:.3f} · F1 macro {m['f1_macro']:.3f} · "
        f"acierto 95 % {m['acierto_95']:.3f} · falsas 95 % {m['falsas_95']:.3f} ({resultado.duracion_s:.0f} s)"
    )
    return True


def _semilla_permitida(semilla: int) -> None:
    if semilla not in SEMILLAS:
        raise SemillaNoPermitida(
            f"La semilla {semilla} no es una de las fijas ({', '.join(map(str, SEMILLAS))}); research R-06."
        )


def _repeticion_del_candidato(
    ctx: Contexto, barrido: str, configuracion: str, semilla: int, rotacion: str
) -> dict[str, str]:
    """Sobre un barrido cerrado solo se repite la selección del candidato con su semilla final (SC-005 de
    TT-14). La repetición queda etiquetada con el run original y no cuenta en el resumen."""
    elegido = elegir_candidato(_analizar(ctx, barrido).agregados)
    if elegido is None or (elegido.configuracion, elegido.semilla_final) != (configuracion, semilla):
        candidato = f" ({elegido.configuracion} con la semilla {elegido.semilla_final})" if elegido else ""
        raise BarridoCerrado(
            f"El barrido {barrido} está cerrado: solo admite repetir la selección del candidato{candidato}. "
            "Para entrenar otra cosa, abre otro barrido."
        )
    original = next(c for c in elegido.corridas if (c.semilla, c.rotacion) == (semilla, rotacion))
    return {"repeticion.de": original.corrida_id}


def cmd_corrida(ctx: Contexto, barrido: str, configuracion: str, semilla: int, rotacion: str,
                permitir_sucio: bool) -> int:
    _semilla_permitida(semilla)
    ronda, cfg = buscar_configuracion(leer_barrido(ruta_barrido(ctx.raiz, barrido)), configuracion)
    cerrado = ctx.registro.estado_barrido(barrido).get("barrido.estado") == "cerrado"
    extra = _repeticion_del_candidato(ctx, barrido, configuracion, semilla, rotacion) if cerrado else {}
    _ronda_sin_cambios(ctx, ronda)
    codigo = codigo_permitido(ctx.raiz, permitir_sucio)
    datos = cargar_rotacion(ctx.raiz, rotacion, ctx.lector_seleccion)
    mismos_datos(ctx, barrido, datos)
    return 0 if ejecutar_corrida(ctx, ronda, cfg, semilla, rotacion, datos, codigo, extra) else 1


def cmd_ronda(ctx: Contexto, barrido: str, numero: int, permitir_sucio: bool) -> int:
    rondas = leer_barrido(ruta_barrido(ctx.raiz, barrido))
    if not 1 <= numero <= len(rondas):
        raise RondaNoValida(f"El barrido {barrido} no declara la ronda {numero}.")
    ronda = rondas[numero - 1]
    _barrido_abierto(ctx, barrido)
    interrumpidas = ctx.registro.marcar_interrumpidas(barrido)
    if interrumpidas:
        print(f"{len(interrumpidas)} corrida(s) que quedaron en curso se marcan como interrumpidas y se repiten.")
    _ronda_sin_cambios(ctx, ronda)
    terminadas = {
        (c.configuracion, c.semilla, c.rotacion) for c in ctx.registro.corridas(barrido)
        if c.estado is EstadoCorrida.TERMINADA and c.fase == "seleccion"
    }
    pendientes = [
        (cfg, s, r) for cfg in ronda.configuraciones for s in SEMILLAS for r in ROTACIONES
        if (cfg.nombre, s, r) not in terminadas
    ]
    print(f"Ronda {numero} de {barrido}: {len(pendientes)} corrida(s) pendientes.")
    fallidas = 0
    if pendientes:
        codigo = codigo_permitido(ctx.raiz, permitir_sucio)
        datos: dict[str, DatosEntrenamiento] = {}
        for rotacion in sorted({r for _, _, r in pendientes}):
            datos[rotacion] = cargar_rotacion(ctx.raiz, rotacion, ctx.lector_seleccion)
            mismos_datos(ctx, barrido, datos[rotacion])
        for cfg, semilla, rotacion in pendientes:
            fallidas += not ejecutar_corrida(ctx, ronda, cfg, semilla, rotacion, datos[rotacion], codigo)
    analisis = _analizar(ctx, barrido, rondas)
    _publicar_resumen(ctx, barrido, analisis)
    if estancado(analisis.mejores):
        print(f"El barrido se ha estancado: ciérralo con `uv run signai-reconocedor cerrar {barrido}`.")
    return 1 if fallidas else 0


# --- Resumen, cierre y candidato ---------------------------------------------------------------


@dataclass(frozen=True)
class Analisis:
    agregados: list[Agregado]
    mejores: list[tuple[int, float | None]]
    corridas: dict[str, CorridaRegistrada]


def _resultado(ctx: Contexto, corrida: CorridaRegistrada) -> ResultadoRotacion:
    """Una corrida de selección con sus curvas de validación por época, leídas del registro."""
    curvas = {
        m: tuple(valor for _, valor in sorted(ctx.registro.historia(corrida.id, f"validacion.{m}"))) for m in METRICAS
    }
    return ResultadoRotacion(
        corrida.id, corrida.nombre, corrida.configuracion, corrida.semilla, corrida.rotacion or "", corrida.ronda,
        int(corrida.etiquetas["modelo.parametros"]), curvas,
    )


def _analizar(ctx: Contexto, barrido: str, rondas: tuple[Ronda, ...] | None = None) -> Analisis:
    rondas = rondas or leer_barrido(ruta_barrido(ctx.raiz, barrido))
    orden = {cfg.nombre: (r.numero, i) for r in rondas for i, cfg in enumerate(r.configuraciones)}
    terminadas = sorted(
        (
            c for c in ctx.registro.corridas(barrido)
            if c.estado is EstadoCorrida.TERMINADA and c.fase == "seleccion" and "repeticion.de" not in c.etiquetas
        ),
        key=lambda c: c.inicio,
    )
    agregados = agregar([_resultado(ctx, c) for c in terminadas], orden)
    completas = {a.configuracion for a in agregados}
    rondas_completas = [r.numero for r in rondas if all(c.nombre in completas for c in r.configuraciones)]
    return Analisis(agregados, mejores_por_ronda(agregados, rondas_completas), {c.id: c for c in terminadas})


def _publicar_resumen(ctx: Contexto, barrido: str, analisis: Analisis) -> None:
    estado = ctx.registro.estado_barrido(barrido).get("barrido.estado", "abierto")
    texto = generar_resumen(barrido, analisis.agregados, analisis.mejores, estado,
                            elegir_candidato(analisis.agregados))
    ctx.registro.guardar_resumen(barrido, texto)
    print(texto, end="")


def cmd_resumen(ctx: Contexto, barrido: str) -> int:
    _publicar_resumen(ctx, barrido, _analizar(ctx, barrido))
    return 0


def cmd_cerrar(ctx: Contexto, barrido: str) -> int:
    _barrido_abierto(ctx, barrido)
    analisis = _analizar(ctx, barrido)
    if not any(a.elegible for a in analisis.agregados):
        raise SinCandidato(
            f"El barrido {barrido} no tiene ninguna configuración elegible (límite de parámetros); "
            "no se puede cerrar. Declara otra ronda."
        )
    if not estancado(analisis.mejores):
        raise ErrorReconocedor(
            f"El barrido {barrido} no se ha estancado: hacen falta 2 rondas completas seguidas que mejoren la mejor "
            f"media elegible menos de 0,5 puntos cada una (mejores: {analisis.mejores}). Declara otra ronda."
        )
    (ronda, despues), (_, antes) = analisis.mejores[-1], analisis.mejores[-3]
    ctx.registro.cerrar_barrido(
        barrido, {"cierre.ronda": str(ronda), "cierre.mejor_antes": f"{antes:.6f}", "cierre.mejor_despues": f"{despues:.6f}"}
    )
    print(f"Barrido {barrido} cerrado en la ronda {ronda}: mejor media {antes:.3f} → {despues:.3f}.")
    _publicar_resumen(ctx, barrido, analisis)
    return 0


def _final_existente(ctx: Contexto, barrido: str, agregado: Agregado) -> CorridaRegistrada | None:
    """El modelo final ya entrenado para esta configuración, semilla y épocas, si lo hay."""
    for corrida in ctx.registro.corridas(barrido):
        if (
            corrida.fase == "final" and corrida.estado is EstadoCorrida.TERMINADA
            and corrida.configuracion == agregado.configuracion and corrida.semilla == agregado.semilla_final
            and corrida.etiquetas.get("epocas_elegidas") == str(agregado.epocas_elegidas)
        ):
            return corrida
    return None


def _entrenar_final(ctx: Contexto, barrido: str, agregado: Agregado, permitir_sucio: bool) -> CorridaRegistrada:
    """Entrena el modelo final (research R-06) y lo registra como corrida `final` con su modelo."""
    rondas = leer_barrido(ruta_barrido(ctx.raiz, barrido))
    ronda, cfg = buscar_configuracion(rondas, agregado.configuracion)
    codigo = codigo_permitido(ctx.raiz, permitir_sucio)
    datos = cargar_final(ctx.raiz, ctx.lector_final)
    mismos_datos(ctx, barrido, datos)
    semilla, epocas = agregado.semilla_final, agregado.epocas_elegidas
    nombre = f"{cfg.nombre}-s{semilla}-final"
    id_corrida = ctx.registro.abrir(
        barrido, nombre, {**cfg.aplanar(), "semilla": semilla},
        {**_etiquetas(ronda, cfg, datos, ctx, codigo), "fase": "final", "epocas_elegidas": str(epocas),
         "final.clips": str(len(datos.x_entrenamiento))},
    )
    print(f"→ {nombre}: modelo final con {len(datos.x_entrenamiento)} clips y {epocas} época(s) ({ctx.dev.type})",
          flush=True)
    try:
        resultado = entrenar_final(datos, cfg, semilla, epocas, lambda e, m: ctx.registro.epoca(id_corrida, e, m),
                                   ctx.dev)
    except BaseException:
        ctx.registro.interrumpir(id_corrida)
        raise
    if resultado.estado is not EstadoCorrida.TERMINADA:
        ctx.registro.fallar(id_corrida, resultado.motivo)
        raise ErrorReconocedor(f"El modelo final {nombre} falló: {resultado.motivo}.")
    with tempfile.TemporaryDirectory() as temporal:
        modelo = guardar(
            Path(temporal) / "modelo.pt", resultado.modelo, cfg.forma, list(datos.clases), list(datos.glosas),
            corrida={"barrido": barrido, "run_id": id_corrida, "configuracion": cfg.nombre, "semilla": semilla,
                     "epoca": epocas, "fase": "final"},
            huellas=datos.huellas,
        )
        ctx.registro.terminar(id_corrida, {"final.epocas": epocas, "duracion_s": resultado.duracion_s}, [modelo], {})
    return next(c for c in ctx.registro.corridas(barrido) if c.id == id_corrida)


def cmd_candidato(ctx: Contexto, barrido: str, permitir_sucio: bool = False) -> int:
    if ctx.registro.estado_barrido(barrido).get("barrido.estado") != "cerrado":
        raise ErrorReconocedor(
            f"El barrido {barrido} sigue abierto: el candidato solo sale de un barrido cerrado. Usa "
            f"`uv run signai-reconocedor cerrar {barrido}` cuando se estanque."
        )
    agregado = elegir_candidato(_analizar(ctx, barrido).agregados)
    if agregado is None:
        raise SinCandidato(f"El barrido {barrido} no tiene ninguna configuración elegible.")
    final = _final_existente(ctx, barrido, agregado) or _entrenar_final(ctx, barrido, agregado, permitir_sucio)
    with tempfile.TemporaryDirectory() as temporal:
        modelo = ctx.registro.descargar(final.id, "modelo.pt", Path(temporal))
        destino = escribir_candidato(
            ctx.raiz, barrido, agregado, final, modelo, int(final.etiquetas["final.clips"]),
            float(final.metricas.get("duracion_s", 0.0)),
        )
    print(
        f"Candidato de {barrido}: {agregado.configuracion} con la semilla {agregado.semilla_final} y "
        f"{agregado.epocas_elegidas} época(s) (Top-1 de selección {agregado.medias['top1']:.3f}; rotaciones "
        f"{agregado.por_rotacion['a']:.3f} / {agregado.por_rotacion['b']:.3f}) → {destino.relative_to(ctx.raiz)}\n"
        f"{VERSIONAR}"
    )
    return 0


# --- Ciclo automático (FR-019; research R-16) ----------------------------------------------------


def _ronda_completa(ronda: Ronda, analisis: Analisis) -> bool:
    completas = {a.configuracion for a in analisis.agregados}
    return all(cfg.nombre in completas for cfg in ronda.configuraciones)


def _declarar(ctx: Contexto, barrido: str, numero: int, texto: str) -> None:
    ruta = ruta_barrido(ctx.raiz, barrido) / f"ronda-{numero:02d}.toml"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    ctx.confirmar(ctx.raiz, ruta, f"{CLAVE_JIRA} · Ronda {numero} de {barrido} declarada")
    print(f"Ronda {numero} de {barrido} declarada y confirmada: {ruta.relative_to(ctx.raiz)}", flush=True)


def _candidato_pendiente(ctx: Contexto, barrido: str) -> bool:
    ficha = ctx.raiz / "modelos" / "reconocedor" / "candidato.json"
    return not ficha.is_file() or json.loads(ficha.read_text(encoding="utf-8")).get("barrido") != barrido


def cmd_automatico(ctx: Contexto, barrido: str, max_rondas: int | None, permitir_sucio: bool) -> int:
    codigo_permitido(ctx.raiz, permitir_sucio)
    lanzadas = 0
    while True:
        if ctx.registro.estado_barrido(barrido).get("barrido.estado") == "cerrado":
            print(f"El barrido {barrido} está cerrado.")
            return cmd_candidato(ctx, barrido, permitir_sucio) if _candidato_pendiente(ctx, barrido) else 0
        directorio = ruta_barrido(ctx.raiz, barrido)
        rondas = leer_barrido(directorio) if any(directorio.glob("ronda-*.toml")) else ()
        if not rondas:
            _declarar(ctx, barrido, 1, automatico.generar_primera_ronda(barrido))
            continue
        analisis = _analizar(ctx, barrido, rondas)
        ultima = rondas[-1]
        if not _ronda_completa(ultima, analisis):
            if max_rondas is not None and lanzadas >= max_rondas:
                print(f"Alcanzado el máximo de {max_rondas} ronda(s); relanza la orden para seguir.")
                return 0
            resultado = cmd_ronda(ctx, barrido, ultima.numero, permitir_sucio)
            lanzadas += 1
            if ctx.copiar() != 0:
                print("Aviso: la copia de seguridad entre rondas falló; revisa `signai-copia`.", file=sys.stderr)
            if resultado != 0:
                print(f"Error: en la ronda {ultima.numero} fallaron corridas; revísalas antes de seguir.",
                      file=sys.stderr)
                return 1
            continue
        if estancado(analisis.mejores):
            cmd_cerrar(ctx, barrido)
            return cmd_candidato(ctx, barrido, permitir_sucio)
        texto = automatico.generar_siguiente_ronda(barrido, ultima.numero + 1, rondas, analisis.agregados)
        if texto is None:
            print(f"Error: el vecindario de búsqueda de {barrido} está agotado sin estancarse; declara una ronda a "
                  "mano o amplía el vecindario.", file=sys.stderr)
            return 1
        _declarar(ctx, barrido, ultima.numero + 1, texto)


def main(
    argv: Sequence[str] | None = None,
    raiz: Path | None = None,
    registro: Registro | None = None,
    lector_seleccion: Callable | None = None,
    lector_final: Callable | None = None,
    dev: torch.device | None = None,
    confirmar: Callable[[Path, Path, str], None] | None = None,
    copiar: Callable[[], int] | None = None,
) -> int:
    argumentos = _parser().parse_args(argv)
    ctx = Contexto(
        raiz=raiz or _raiz_repo(),
        registro=registro or RegistroMlflow(),
        lector_seleccion=lector_seleccion or para_seleccion,
        lector_final=lector_final or para_entrenamiento_final,
        dev=dev or dispositivo(),
        confirmar=confirmar or automatico.confirmar_con_git,
    )
    try:
        if argumentos.orden == "automatico":
            codigo_permitido(ctx.raiz, argumentos.permitir_sucio)  # antes de pedir la clave de la copia
            ctx.copiar = copiar or automatico.preparar_copia()
            return cmd_automatico(ctx, argumentos.barrido, argumentos.max_rondas, argumentos.permitir_sucio)
        if argumentos.orden == "corrida":
            return cmd_corrida(ctx, argumentos.barrido, argumentos.configuracion, argumentos.semilla,
                               argumentos.rotacion, argumentos.permitir_sucio)
        if argumentos.orden == "ronda":
            return cmd_ronda(ctx, argumentos.barrido, argumentos.numero, argumentos.permitir_sucio)
        if argumentos.orden == "candidato":
            return cmd_candidato(ctx, argumentos.barrido, argumentos.permitir_sucio)
        ordenes = {"resumen": cmd_resumen, "cerrar": cmd_cerrar}
        return ordenes[argumentos.orden](ctx, argumentos.barrido)
    except (ErrorReconocedor, ErrorParticion, ErrorCopia) as error:
        print(f"Error: {error}", file=sys.stderr)
        return error.codigo_salida
