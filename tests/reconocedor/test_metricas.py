"""Métricas de validación (spec FR-007; research R-09)."""

import numpy as np
import pytest

from signai_ml.reconocedor.metricas import (
    acierto_95,
    calcular,
    f1_macro,
    falsas_95,
    top1,
    top1_por_clase,
)

# 4 clips, 3 clases. Predicciones: 0 (bien, 0,97), 1 (bien, 0,60), 1 (mal, 0,96), 2 (bien, 0,95).
PROB = np.array(
    [[0.97, 0.02, 0.01], [0.30, 0.60, 0.10], [0.02, 0.96, 0.02], [0.03, 0.02, 0.95]], dtype=np.float32
)
Y = np.array([0, 1, 0, 2])


def test_top1() -> None:
    assert top1(PROB, Y) == pytest.approx(0.75)


def test_f1_macro() -> None:
    # clase 0: tp1 fp0 fn1 → 2/3; clase 1: tp1 fp1 fn0 → 2/3; clase 2: tp1 → 1
    assert f1_macro(PROB, Y, 3) == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_clase_sin_predicciones_ni_ejemplos_aporta_cero() -> None:
    assert f1_macro(PROB, Y, 4) == pytest.approx((2 / 3 + 2 / 3 + 1 + 0) / 4)


def test_top1_por_clase() -> None:
    assert top1_por_clase(PROB, Y, 3) == pytest.approx([0.5, 1.0, 1.0])
    assert top1_por_clase(PROB, Y, 4)[3] == 0.0


def test_acierto_y_falsas_al_95() -> None:
    # 0,95 exacto cuenta como ≥
    assert acierto_95(PROB, Y) == pytest.approx(2 / 4)
    assert falsas_95(PROB, Y) == pytest.approx(1 / 4)


def test_calcular_devuelve_las_cuatro() -> None:
    assert set(calcular(PROB, Y, 3)) == {"top1", "f1_macro", "acierto_95", "falsas_95"}
