"""Regla de aceptación ≥85 % Top-1, entera y bloqueante (spec FR-004; research R-04; ADR-0011)."""

import pytest

from signai_ml.aceptacion.decision import Decision, decidir, umbral_aciertos
from signai_ml.aceptacion.errores import ErrorAceptacion


@pytest.mark.parametrize(
    ("aciertos", "esperada"),
    [(1530, Decision.ACEPTADO), (1529, Decision.RECHAZADO), (1800, Decision.ACEPTADO), (0, Decision.RECHAZADO)],
)
def test_sobre_los_1800_clips(aciertos: int, esperada: Decision) -> None:
    assert decidir(aciertos, 1800) is esperada


def test_el_borde_exacto_se_acepta_sin_coma_flotante() -> None:
    # 17/20 = 0,85 exacto, pero 0.85 * 20 = 16.999999999999996 en coma flotante.
    assert decidir(17, 20) is Decision.ACEPTADO
    assert decidir(16, 20) is Decision.RECHAZADO


def test_umbral_en_aciertos() -> None:
    assert umbral_aciertos(1800) == 1530
    assert umbral_aciertos(12) == 11  # 85 % de 12 = 10,2 → hacen falta 11
    assert umbral_aciertos(20) == 17


@pytest.mark.parametrize(("aciertos", "total"), [(1801, 1800), (-1, 1800), (0, 0), (1, -5)])
def test_recuentos_imposibles(aciertos: int, total: int) -> None:
    with pytest.raises(ErrorAceptacion):
        decidir(aciertos, total)


def test_los_valores_de_la_decision() -> None:
    assert [d.value for d in Decision] == ["aceptado", "rechazado"]
    assert Decision("aceptado") is Decision.ACEPTADO
