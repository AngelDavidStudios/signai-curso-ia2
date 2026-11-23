"""Configuración de una corrida: todo lo que fija qué y cómo se entrena, salvo la semilla.

data-model «Configuración»; spec FR-006, FR-009; contracts/ronda.md. Los valores por defecto son la
base de research R-07 (forma de ADR-0013 y valores de partida del laboratorio).
"""

import re
from dataclasses import asdict, dataclass, fields
from typing import Any

from .errores import RondaNoValida

NOMBRE = re.compile(r"[a-z0-9-]+")
OPTIMIZADORES = ("adamw",)
PROGRAMACIONES = ("coseno", "constante")
# Tipo de aumento → rango abierto/cerrado de su intensidad (research R-08).
TIPOS_AUMENTO: dict[str, tuple[float, float]] = {
    "desplazamiento": (0.0, 0.5),
    "escala": (0.0, 0.5),
    "rotacion": (0.0, 0.5),
    "ocultacion": (0.0, 0.5),
    "velocidad": (0.0, 0.5),
    "recorte_temporal": (0.1, 1.0),
}

Escalar = str | int | float | bool


@dataclass(frozen=True)
class Forma:
    capas: int = 4
    dimension: int = 98
    cabezas: int = 7
    intermedia: int = 256


@dataclass(frozen=True)
class Entrenamiento:
    optimizador: str = "adamw"
    tasa: float = 1e-3
    programacion: str = "coseno"
    regularizacion: float = 0.05
    dropout: float = 0.05
    recorte_gradiente: float = 1.0
    lote: int = 32
    epocas: int = 200
    suavizado_etiquetas: float = 0.0


@dataclass(frozen=True)
class Aumento:
    probabilidad: float
    intensidad: float


@dataclass(frozen=True)
class Configuracion:
    nombre: str
    forma: Forma
    entrenamiento: Entrenamiento
    aumento: tuple[tuple[str, Aumento], ...]

    def aplanar(self) -> dict[str, Escalar]:
        plano: dict[str, Escalar] = {f"forma.{k}": v for k, v in asdict(self.forma).items()}
        plano |= {f"entrenamiento.{k}": v for k, v in asdict(self.entrenamiento).items()}
        for tipo, aumento in self.aumento:
            plano[f"aumento.{tipo}.probabilidad"] = aumento.probabilidad
            plano[f"aumento.{tipo}.intensidad"] = aumento.intensidad
        return plano

    def valores(self) -> tuple:
        """Todo menos el nombre: dos configuraciones con los mismos valores son la misma."""
        return (self.forma, self.entrenamiento, self.aumento)


def _error(ruta: str, motivo: str) -> RondaNoValida:
    return RondaNoValida(f"{ruta}: {motivo}. Corrige la declaración de la ronda (contracts/ronda.md).")


def _numero(ruta: str, valor: Any, entero: bool) -> float:
    if isinstance(valor, bool) or not isinstance(valor, int | float) or (entero and not isinstance(valor, int)):
        raise _error(ruta, f"se esperaba un {'entero' if entero else 'número'} y llegó {valor!r}")
    return valor


def _sin_sobrantes(ruta: str, datos: dict, permitidas: set[str]) -> None:
    sobrantes = sorted(set(datos) - permitidas)
    if sobrantes:
        prefijo = f"{ruta}." if ruta else ""
        raise _error(f"{prefijo}{sobrantes[0]}", "clave desconocida")


def _forma(datos: dict) -> Forma:
    _sin_sobrantes("forma", datos, {f.name for f in fields(Forma)})
    forma = Forma(**datos)
    for clave, minimo in (("capas", 1), ("dimension", 2), ("cabezas", 1), ("intermedia", 1)):
        if _numero(f"forma.{clave}", getattr(forma, clave), entero=True) < minimo:
            raise _error(f"forma.{clave}", f"debe ser ≥ {minimo}")
    if forma.dimension % forma.cabezas:
        raise _error("forma.cabezas", f"{forma.cabezas} no divide a la dimensión {forma.dimension}")
    return forma


def _entrenamiento(datos: dict) -> Entrenamiento:
    _sin_sobrantes("entrenamiento", datos, {f.name for f in fields(Entrenamiento)})
    e = Entrenamiento(**datos)
    if e.optimizador not in OPTIMIZADORES:
        raise _error("entrenamiento.optimizador", f"admite {', '.join(OPTIMIZADORES)}")
    if e.programacion not in PROGRAMACIONES:
        raise _error("entrenamiento.programacion", f"admite {', '.join(PROGRAMACIONES)}")
    reglas = (
        ("tasa", False, lambda v: v > 0, "debe ser > 0"),
        ("regularizacion", False, lambda v: v >= 0, "debe ser ≥ 0"),
        ("dropout", False, lambda v: 0 <= v < 1, "debe estar en [0, 1)"),
        ("recorte_gradiente", False, lambda v: v >= 0, "debe ser ≥ 0"),
        ("lote", True, lambda v: v >= 1, "debe ser ≥ 1"),
        ("epocas", True, lambda v: v >= 1, "debe ser ≥ 1"),
        ("suavizado_etiquetas", False, lambda v: 0 <= v < 1, "debe estar en [0, 1)"),
    )
    for clave, entero, valido, motivo in reglas:
        if not valido(_numero(f"entrenamiento.{clave}", getattr(e, clave), entero)):
            raise _error(f"entrenamiento.{clave}", motivo)
    return e


def _aumento(datos: dict) -> tuple[tuple[str, Aumento], ...]:
    resultado = []
    for tipo in sorted(datos, key=lambda t: list(TIPOS_AUMENTO).index(t) if t in TIPOS_AUMENTO else -1):
        ruta = f"aumento.{tipo}"
        if tipo not in TIPOS_AUMENTO:
            raise _error(ruta, f"tipo desconocido; admite {', '.join(TIPOS_AUMENTO)}")
        valores = datos[tipo]
        _sin_sobrantes(ruta, valores, {"probabilidad", "intensidad"})
        for clave in ("probabilidad", "intensidad"):
            if clave not in valores:
                raise _error(f"{ruta}.{clave}", "falta")
        probabilidad = _numero(f"{ruta}.probabilidad", valores["probabilidad"], entero=False)
        if not 0 <= probabilidad <= 1:
            raise _error(f"{ruta}.probabilidad", "debe estar en [0, 1]")
        minimo, maximo = TIPOS_AUMENTO[tipo]
        intensidad = _numero(f"{ruta}.intensidad", valores["intensidad"], entero=False)
        if not minimo < intensidad < maximo if tipo != "recorte_temporal" else not minimo <= intensidad < maximo:
            raise _error(f"{ruta}.intensidad", f"fuera de rango ({minimo}, {maximo})")
        resultado.append((tipo, Aumento(float(probabilidad), float(intensidad))))
    return tuple(resultado)


def configuracion_desde_dict(nombre: str, datos: dict) -> Configuracion:
    """Valida y construye una configuración ya fusionada con la base de su ronda."""
    if not isinstance(nombre, str) or not NOMBRE.fullmatch(nombre):
        raise _error("nombre", f"{nombre!r} no cumple [a-z0-9-]+")
    _sin_sobrantes("", datos, {"forma", "entrenamiento", "aumento"})
    try:
        return Configuracion(
            nombre=nombre,
            forma=_forma(dict(datos.get("forma", {}))),
            entrenamiento=_entrenamiento(dict(datos.get("entrenamiento", {}))),
            aumento=_aumento(dict(datos.get("aumento", {}))),
        )
    except TypeError as error:  # p. ej. una sección que no es una tabla
        raise _error(nombre, f"declaración mal formada ({error})") from error
