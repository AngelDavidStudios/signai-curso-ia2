"""Reparto de las grabaciones de una clase (TT-22, research R-01; ADR-0011, enmienda del 2026-11-20).

En el orden del sorteo, la primera grabación va a prueba y las siguientes a los pliegues A y B,
alternando. Hacen falta al menos 3: una de prueba y una para cada pliegue, sin las cuales no hay
rotación con la que validar.
"""

from . import GRABACIONES_MINIMAS, Lado
from .errores import IndiceNoValido

_PLIEGUES = (Lado.PLIEGUE_A, Lado.PLIEGUE_B)


def lados_por_posicion(n: int, clase: str) -> list[Lado]:
    """El lado de cada grabación de `clase`, en el orden del sorteo: 3 → prueba, A, B; 4 → prueba, A, B, A."""
    if n < GRABACIONES_MINIMAS:
        raise IndiceNoValido(
            f"La clase {clase} tiene {n} grabaciones y hacen falta al menos {GRABACIONES_MINIMAS}: una para "
            "la prueba y una para cada pliegue de la selección. Revisa el índice del corpus procesado."
        )
    return [Lado.PRUEBA] + [_PLIEGUES[i % 2] for i in range(n - 1)]
