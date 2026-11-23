"""Umbral de aceptación ≥85 % Top-1 (constitución IV; ADR-0011; spec FR-004; research R-04).

Bloqueante y en aritmética entera: se acepta si `aciertos × 100 ≥ 85 × total`. Con coma flotante,
`0.85 * 20` da 16,999…, y un redondeo no puede decidir en el borde. Ninguna otra métrica acepta ni
rechaza.
"""

from enum import StrEnum

from . import UMBRAL_TOP1
from .errores import ErrorAceptacion


class Decision(StrEnum):
    ACEPTADO = "aceptado"
    RECHAZADO = "rechazado"


def _comprobar(aciertos: int, total: int) -> None:
    if total <= 0 or not 0 <= aciertos <= total:
        raise ErrorAceptacion(f"Recuento imposible: {aciertos} aciertos de {total} clips.")


def decidir(aciertos: int, total: int) -> Decision:
    _comprobar(aciertos, total)
    numerador, denominador = UMBRAL_TOP1
    return Decision.ACEPTADO if aciertos * denominador >= numerador * total else Decision.RECHAZADO


def umbral_aciertos(total: int) -> int:
    """Los aciertos mínimos para aceptar sobre `total` clips (1.530 sobre 1.800)."""
    numerador, denominador = UMBRAL_TOP1
    return -(-numerador * total // denominador)
