"""Métricas de la prueba sobre probabilidades conocidas (spec FR-003; research R-03)."""

import numpy as np
import pytest
import torch

from signai_ml.aceptacion.evaluacion import ResultadoEvaluacion, evaluar, probabilidades
from signai_ml.reconocedor import metricas
from signai_ml.reconocedor.configuracion import Forma
from signai_ml.reconocedor.modelo import KPoseFormer

# 6 clips de 3 clases: aciertan 4 (dos con confianza ≥95 %) y fallan 2 (uno con confianza ≥95 %).
PROBABILIDADES = np.array(
    [
        [0.97, 0.02, 0.01],  # clase 0 → 0, seguro
        [0.60, 0.30, 0.10],  # clase 0 → 0
        [0.10, 0.85, 0.05],  # clase 1 → 1
        [0.96, 0.03, 0.01],  # clase 1 → 0, falsa segura
        [0.01, 0.01, 0.98],  # clase 2 → 2, seguro
        [0.20, 0.50, 0.30],  # clase 2 → 1
    ]
)
ETIQUETAS = np.array([0, 0, 1, 1, 2, 2])


def test_recuento_y_top1() -> None:
    resultado = evaluar(PROBABILIDADES, ETIQUETAS, 3)

    assert isinstance(resultado, ResultadoEvaluacion)
    assert resultado.aciertos == 4
    assert resultado.total == 6
    assert resultado.top1 == pytest.approx(4 / 6)


def test_mismas_definiciones_que_tt14() -> None:
    resultado = evaluar(PROBABILIDADES, ETIQUETAS, 3)

    assert resultado.f1_macro == pytest.approx(metricas.f1_macro(PROBABILIDADES, ETIQUETAS, 3))
    assert resultado.acierto_95 == pytest.approx(metricas.acierto_95(PROBABILIDADES, ETIQUETAS))
    assert resultado.falsas_95 == pytest.approx(metricas.falsas_95(PROBABILIDADES, ETIQUETAS))
    assert resultado.acierto_95 == pytest.approx(2 / 6)
    assert resultado.falsas_95 == pytest.approx(1 / 6)
    assert resultado.top1_por_clase == pytest.approx((1.0, 0.5, 0.5))
    assert len(resultado.f1_por_clase) == 3


def test_matriz_de_confusion() -> None:
    confusion = evaluar(PROBABILIDADES, ETIQUETAS, 3).confusion

    assert confusion.dtype == np.int64
    assert confusion.shape == (3, 3)
    assert confusion.tolist() == [[2, 0, 0], [1, 1, 0], [0, 1, 1]]  # filas = clase real
    assert confusion.sum() == 6
    assert np.trace(confusion) == 4


def test_las_metricas_para_el_registro() -> None:
    assert evaluar(PROBABILIDADES, ETIQUETAS, 3).metricas() == {
        "aciertos": 4.0,
        "total": 6.0,
        "top1": pytest.approx(4 / 6),
        "f1_macro": pytest.approx(metricas.f1_macro(PROBABILIDADES, ETIQUETAS, 3)),
        "acierto_95": pytest.approx(2 / 6),
        "falsas_95": pytest.approx(1 / 6),
    }


def test_evaluar_es_pura() -> None:
    p, y = PROBABILIDADES.copy(), ETIQUETAS.copy()

    evaluar(p, y, 3)

    assert np.array_equal(p, PROBABILIDADES)
    assert np.array_equal(y, ETIQUETAS)


def test_entradas_incoherentes() -> None:
    with pytest.raises(ValueError):
        evaluar(PROBABILIDADES, ETIQUETAS[:-1], 3)
    with pytest.raises(ValueError):
        evaluar(PROBABILIDADES, ETIQUETAS, 4)


def test_probabilidades_del_modelo_en_cpu_por_lotes() -> None:
    torch.manual_seed(0)
    modelo = KPoseFormer(Forma(capas=1, dimension=16, cabezas=2, intermedia=16), clases=3).eval()
    x = np.random.default_rng(0).random((5, 75, 96), dtype=np.float32)

    p = probabilidades(modelo, x, lote=2)

    with torch.no_grad():
        esperadas = torch.softmax(modelo(torch.from_numpy(x)), dim=1).numpy()
    assert p.shape == (5, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p, esperadas, atol=1e-6)
    assert np.allclose(p, probabilidades(modelo, x, lote=5), atol=1e-7)  # el lote no cambia el resultado
