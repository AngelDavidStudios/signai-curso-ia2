"""Selección por rotación, elegibilidad y cierre del barrido (TT-22, research R-05; data-model «Agregado»).

Funciones puras sobre las corridas de selección terminadas; nunca ven la prueba. Cada configuración se
entrena con las 3 semillas en las 2 rotaciones (entrena un pliegue y valida el otro): 6 corridas.

- **Curva** de la configuración: la media, época a época, del Top-1 de validación de sus 6 corridas.
  **Épocas elegidas (`E*`)**: el máximo de esa curva; empates por acierto al 95 % medio y después la
  época menor. Las cifras de la configuración son las medias de sus 6 corridas en `E*`.
- **Diferencia entre rotaciones**: |media de la rotación a − media de la b| del Top-1 en `E*`. Se
  informa y desempata; no quita la elegibilidad (SC-002, tasks T037).
- **Elegible**: parámetros ≤ `LIMITE_PARAMETROS`. El acierto al 95 % se informa y desempata, pero no
  descarta: con un solo pliegue de entrenamiento (una grabación por clase) el umbral de TT-14 no es
  alcanzable y la exigencia es el Top-1 en la prueba (aclaración de la spec; research R-05, enmienda
  del 2026-11-21).
- **Candidato**: la configuración elegible de mayor Top-1 medio; desempates: menor diferencia entre
  rotaciones, mayor acierto al 95 %, mayor F1 macro y la declarada antes.
- **Semilla del modelo final**: la de mayor media de sus dos rotaciones en `E*` (a igualdad, la primera).
- **Estancamiento**: las 2 últimas rondas completas mejoran la mejor media elegible en menos de
  `MEJORA_MINIMA` cada una, como en TT-14.
"""

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from . import LIMITE_PARAMETROS, MEJORA_MINIMA, RONDAS_SIN_MEJORA, SEMILLAS

METRICAS = ("top1", "f1_macro", "acierto_95", "falsas_95")
ROTACIONES = ("a", "b")


@dataclass(frozen=True)
class ResultadoRotacion:
    """Una corrida de selección terminada y sus curvas de validación (época 1 en la posición 0)."""

    corrida_id: str
    nombre: str
    configuracion: str
    semilla: int
    rotacion: str
    ronda: int
    parametros: int
    curvas: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class Agregado:
    configuracion: str
    ronda: int
    orden: tuple[int, int]
    corridas: tuple[ResultadoRotacion, ...]
    epocas_elegidas: int
    medias: dict[str, float]
    por_rotacion: dict[str, float]
    diferencia_rotaciones: float
    parametros: int
    elegible: bool
    motivo: str | None
    semilla_final: int


def _motivo(parametros: int, limite: int) -> str | None:
    if parametros > limite:
        return f"más de {limite:,} parámetros ({parametros:,})".replace(",", ".")
    return None


def _media(corridas: Iterable[ResultadoRotacion], metrica: str, epoca: int) -> float:
    return statistics.fmean(c.curvas[metrica][epoca - 1] for c in corridas)


def _agregado(
    configuracion: str, filas: tuple[ResultadoRotacion, ...], orden: tuple[int, int], semillas: Sequence[int],
    limite: int,
) -> Agregado | None:
    epocas = min(len(f.curvas["top1"]) for f in filas)
    if epocas == 0:
        return None
    elegidas = max(
        range(1, epocas + 1),
        key=lambda e: (_media(filas, "top1", e), _media(filas, "acierto_95", e), -e),
    )
    medias = {m: _media(filas, m, elegidas) for m in METRICAS}
    por_rotacion = {r: _media((f for f in filas if f.rotacion == r), "top1", elegidas) for r in ROTACIONES}
    por_semilla = {s: _media((f for f in filas if f.semilla == s), "top1", elegidas) for s in semillas}
    semilla_final = min(semillas, key=lambda s: (-por_semilla[s], semillas.index(s)))
    parametros = filas[0].parametros
    motivo = _motivo(parametros, limite)
    return Agregado(
        configuracion, filas[0].ronda, orden, filas, elegidas, medias, por_rotacion,
        abs(por_rotacion["a"] - por_rotacion["b"]), parametros, motivo is None, motivo, semilla_final,
    )


def agregar(
    resultados: Iterable[ResultadoRotacion],
    orden: Mapping[str, tuple[int, int]],
    semillas: Sequence[int] = SEMILLAS,
    limite: int = LIMITE_PARAMETROS,
) -> list[Agregado]:
    """Una fila por configuración con sus 6 corridas (la primera de cada semilla y rotación); ignora las
    incompletas."""
    semillas = tuple(semillas)
    por_configuracion: dict[str, dict[tuple[int, str], ResultadoRotacion]] = {}
    for resultado in resultados:
        por_configuracion.setdefault(resultado.configuracion, {}).setdefault(
            (resultado.semilla, resultado.rotacion), resultado
        )
    agregados = []
    for configuracion, por_clave in por_configuracion.items():
        claves = [(s, r) for s in semillas for r in ROTACIONES]
        if not all(k in por_clave for k in claves):
            continue
        agregado = _agregado(
            configuracion, tuple(por_clave[k] for k in claves), tuple(orden.get(configuracion, (10**9, 0))),
            semillas, limite,
        )
        if agregado is not None:
            agregados.append(agregado)
    return sorted(agregados, key=lambda a: a.orden)


def clave_configuracion(a: Agregado) -> tuple:
    return (-a.medias["top1"], a.diferencia_rotaciones, -a.medias["acierto_95"], -a.medias["f1_macro"], a.orden)


def elegir_candidato(agregados: Sequence[Agregado]) -> Agregado | None:
    elegibles = [a for a in agregados if a.elegible]
    return min(elegibles, key=clave_configuracion) if elegibles else None


def mejores_por_ronda(agregados: Sequence[Agregado], rondas: Sequence[int]) -> list[tuple[int, float | None]]:
    """La mejor media de Top-1 elegible acumulada al terminar cada ronda (en orden)."""
    resultado = []
    for ronda in rondas:
        medias = [a.medias["top1"] for a in agregados if a.elegible and a.ronda <= ronda]
        resultado.append((ronda, max(medias) if medias else None))
    return resultado


def estancado(mejores: Sequence[tuple[int, float | None]]) -> bool:
    valores = [valor for _, valor in mejores]
    if len(valores) < RONDAS_SIN_MEJORA + 1 or valores[-1] is None:
        return False
    for anterior, actual in zip(valores[-RONDAS_SIN_MEJORA - 1 : -1], valores[-RONDAS_SIN_MEJORA:]):
        if anterior is None or actual - anterior >= MEJORA_MINIMA:
            return False
    return True
