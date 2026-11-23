"""Paridad del modelo exportado frente al original (spec FR-012, SC-005; research R-08; Clarifications 2026-11-16).

Puro. En cada conjunto, el exportado da los mismos resultados que el original si:

- la clase Top-1 coincide en el 100 % de los casos;
- el número de aciertos es el mismo, cuando hay etiquetas;
- `max |p_original − p_exportado| ≤ 0,01` en todas las clases de todos los casos.

Los casos que cruzan el umbral del 95 % de la app (ADR-0020) son el redondeo de los 16 bits: se
cuentan en cada sentido y no invalidan la paridad.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..reconocedor import UMBRAL_CONFIANZA
from . import TOLERANCIA_PARIDAD


@dataclass(frozen=True)
class InformeConjunto:
    casos: int
    clase_distinta: tuple[int, ...]
    max_diferencia: float
    media_diferencia: float
    aciertos_original: int | None
    aciertos_exportado: int | None
    cambios_95: dict[str, int]
    tolerancia: float

    @property
    def superado(self) -> bool:
        return (
            not self.clase_distinta
            and self.aciertos_original == self.aciertos_exportado
            and self.max_diferencia <= self.tolerancia
        )

    def a_dict(self) -> dict:
        return {
            "casos": self.casos,
            "clase_distinta": list(self.clase_distinta),
            "max_diferencia": self.max_diferencia,
            "media_diferencia": self.media_diferencia,
            "aciertos_original": self.aciertos_original,
            "aciertos_exportado": self.aciertos_exportado,
            "cambios_95": dict(self.cambios_95),
            "superado": self.superado,
        }


def comparar(
    p_original: np.ndarray,
    p_exportado: np.ndarray,
    etiquetas: np.ndarray | None = None,
    tolerancia: float = TOLERANCIA_PARIDAD,
    umbral: float = UMBRAL_CONFIANZA,
) -> InformeConjunto:
    original, exportado = np.asarray(p_original, dtype=np.float64), np.asarray(p_exportado, dtype=np.float64)
    if original.shape != exportado.shape or original.ndim != 2:
        raise ValueError(f"Probabilidades de formas distintas: {original.shape} y {exportado.shape}.")
    if len(original) == 0:
        raise ValueError("No hay casos que comparar: una paridad sin casos no demuestra nada.")
    if etiquetas is not None and len(etiquetas) != len(original):
        raise ValueError(f"{len(etiquetas)} etiquetas para {len(original)} casos.")
    clase_original, clase_exportada = original.argmax(axis=1), exportado.argmax(axis=1)
    diferencia_por_caso = np.abs(original - exportado).max(axis=1)
    seguro_original, seguro_exportado = original.max(axis=1) >= umbral, exportado.max(axis=1) >= umbral
    aciertos = (
        (None, None)
        if etiquetas is None
        else (int((clase_original == etiquetas).sum()), int((clase_exportada == etiquetas).sum()))
    )
    return InformeConjunto(
        casos=len(original),
        clase_distinta=tuple(int(i) for i in np.flatnonzero(clase_original != clase_exportada)),
        max_diferencia=float(diferencia_por_caso.max()),
        media_diferencia=float(diferencia_por_caso.mean()),
        aciertos_original=aciertos[0],
        aciertos_exportado=aciertos[1],
        cambios_95={
            "a_aceptada": int((~seguro_original & seguro_exportado).sum()),
            "a_descartada": int((seguro_original & ~seguro_exportado).sum()),
        },
        tolerancia=tolerancia,
    )


@dataclass(frozen=True)
class InformeParidad:
    unidades: dict[str, dict[str, InformeConjunto]]
    tolerancia: float

    @property
    def superada(self) -> bool:
        return all(informe.superado for conjuntos in self.unidades.values() for informe in conjuntos.values())

    def a_dict(self) -> dict:
        """El contenido de `paridad.json` (data-model, «Informe de paridad»)."""
        return {
            "tolerancia": self.tolerancia,
            "superada": self.superada,
            "unidades": {
                unidad: {conjunto: informe.a_dict() for conjunto, informe in conjuntos.items()}
                for unidad, conjuntos in self.unidades.items()
            },
        }


def informe_paridad(
    resultados: Mapping[str, Mapping[str, InformeConjunto]], tolerancia: float = TOLERANCIA_PARIDAD
) -> InformeParidad:
    if not resultados or any(not conjuntos for conjuntos in resultados.values()):
        raise ValueError("El informe de paridad necesita al menos un conjunto en cada unidad de cómputo.")
    return InformeParidad({u: dict(c) for u, c in resultados.items()}, tolerancia)
