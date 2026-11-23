"""Consultas a la prueba (spec FR-005, SC-003; research R-05; ADR-0011).

Puro. Cada evaluación sobre la prueba es una consulta numerada. El candidato se identifica por la
huella de sus pesos:

- si ya tiene una consulta terminada, no se abre otra;
- si la tiene a medias (fallida, interrumpida o abandonada en curso), se reutiliza su número, porque
  no llegó a verse ninguna cifra;
- si no, es la siguiente a la mayor registrada.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class EstadoConsulta(StrEnum):
    TERMINADA = "terminada"
    FALLIDA = "fallida"
    INTERRUMPIDA = "interrumpida"
    EN_CURSO = "en curso"


@dataclass(frozen=True)
class EvaluacionRegistrada:
    run_id: str
    consulta: int
    modelo_sha256: str
    estado: EstadoConsulta
    decision: str | None = None
    metricas: dict[str, float] = field(default_factory=dict)
    etiquetas: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Siguiente:
    numero: int
    terminada: EvaluacionRegistrada | None
    reutiliza: bool


def total(evaluaciones: Sequence[EvaluacionRegistrada]) -> int:
    """Cuántas consultas a la prueba hay: la mayor registrada (0 si ninguna)."""
    return max((e.consulta for e in evaluaciones), default=0)


def siguiente(evaluaciones: Sequence[EvaluacionRegistrada], modelo_sha256: str) -> Siguiente:
    propias = [e for e in evaluaciones if e.modelo_sha256 == modelo_sha256]
    terminadas = [e for e in propias if e.estado is EstadoConsulta.TERMINADA]
    if terminadas:
        return Siguiente(numero=terminadas[0].consulta, terminada=terminadas[0], reutiliza=False)
    if propias:
        return Siguiente(numero=propias[0].consulta, terminada=None, reutiliza=True)
    return Siguiente(numero=total(evaluaciones) + 1, terminada=None, reutiliza=False)
