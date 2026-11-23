"""Criterio de paridad, puro (spec FR-012, SC-005; research R-08; Clarifications 2026-11-16).

El exportado da los mismos resultados que el original si en cada caso coincide la clase Top-1, si el
número de aciertos es el mismo y si `max |p_original − p_exportado| ≤ 0,01` en todas las clases. Los
cambios de decisión al 95 % se cuentan, pero no invalidan la paridad.
"""

import json

import numpy as np
import pytest

from signai_ml.aceptacion import TOLERANCIA_PARIDAD
from signai_ml.aceptacion.paridad import InformeConjunto, comparar, informe_paridad

ORIGINAL = np.array(
    [
        [0.70, 0.20, 0.10],
        [0.10, 0.80, 0.10],
        [0.05, 0.15, 0.80],
    ]
)
ETIQUETAS = np.array([0, 1, 2])


def _desplazar(p: np.ndarray, fila: int, delta: float) -> np.ndarray:
    """Pasa `delta` de probabilidad de la clase Top-1 de `fila` a la siguiente, sin cambiar la suma."""
    q = p.copy()
    top = int(q[fila].argmax())
    q[fila, top] -= delta
    q[fila, (top + 1) % q.shape[1]] += delta
    return q


def test_la_tolerancia_es_la_acordada() -> None:
    assert TOLERANCIA_PARIDAD == pytest.approx(1e-2)


def test_probabilidades_identicas_superan() -> None:
    informe = comparar(ORIGINAL, ORIGINAL.copy(), ETIQUETAS)

    assert isinstance(informe, InformeConjunto)
    assert informe.superado
    assert informe.casos == 3
    assert informe.clase_distinta == ()
    assert informe.max_diferencia == 0.0
    assert informe.media_diferencia == 0.0
    assert informe.aciertos_original == informe.aciertos_exportado == 3


def test_por_debajo_de_la_tolerancia_supera() -> None:
    informe = comparar(ORIGINAL, _desplazar(ORIGINAL, 0, 0.0099), ETIQUETAS)

    assert informe.superado
    assert informe.max_diferencia == pytest.approx(0.0099)


def test_por_encima_de_la_tolerancia_falla() -> None:
    informe = comparar(ORIGINAL, _desplazar(ORIGINAL, 0, 0.0101), ETIQUETAS)

    assert not informe.superado
    assert informe.clase_distinta == ()
    assert informe.max_diferencia == pytest.approx(0.0101)


def test_la_media_es_la_de_la_mayor_diferencia_de_cada_caso() -> None:
    exportado = _desplazar(_desplazar(ORIGINAL, 0, 0.004), 2, 0.002)

    informe = comparar(ORIGINAL, exportado, ETIQUETAS)

    assert informe.media_diferencia == pytest.approx((0.004 + 0.0 + 0.002) / 3)


def test_una_sola_clase_distinta_falla_y_se_senala() -> None:
    exportado = ORIGINAL.copy()
    exportado[1] = [0.80, 0.10, 0.10]

    informe = comparar(ORIGINAL, exportado, ETIQUETAS)

    assert not informe.superado
    assert informe.clase_distinta == (1,)
    assert informe.aciertos_original == 3
    assert informe.aciertos_exportado == 2


def test_los_cambios_al_95_se_cuentan_y_no_invalidan() -> None:
    original = np.array([[0.951, 0.049], [0.948, 0.052], [0.990, 0.010], [0.600, 0.400]])
    exportado = np.array([[0.949, 0.051], [0.952, 0.048], [0.991, 0.009], [0.601, 0.399]])

    informe = comparar(original, exportado, np.array([0, 0, 0, 0]))

    assert informe.superado
    assert informe.cambios_95 == {"a_aceptada": 1, "a_descartada": 1}


def test_sin_etiquetas_no_cuenta_aciertos() -> None:
    informe = comparar(ORIGINAL, _desplazar(ORIGINAL, 2, 0.003))

    assert informe.superado
    assert informe.aciertos_original is None
    assert informe.aciertos_exportado is None


def test_formas_distintas_son_un_error() -> None:
    with pytest.raises(ValueError):
        comparar(ORIGINAL, ORIGINAL[:2], ETIQUETAS)
    with pytest.raises(ValueError):
        comparar(ORIGINAL, ORIGINAL, ETIQUETAS[:2])


def test_sin_casos_es_un_error() -> None:
    with pytest.raises(ValueError):
        comparar(np.empty((0, 3)), np.empty((0, 3)))


def test_comparar_no_modifica_sus_entradas() -> None:
    original, exportado = ORIGINAL.copy(), _desplazar(ORIGINAL, 0, 0.005)
    copia_original, copia_exportado = original.copy(), exportado.copy()

    comparar(original, exportado, ETIQUETAS)

    assert np.array_equal(original, copia_original)
    assert np.array_equal(exportado, copia_exportado)


def test_el_informe_se_supera_solo_si_todo_se_supera() -> None:
    bien = comparar(ORIGINAL, ORIGINAL.copy(), ETIQUETAS)
    mal = comparar(ORIGINAL, _desplazar(ORIGINAL, 0, 0.05), ETIQUETAS)

    assert informe_paridad({"cpu": {"prueba": bien, "referencia": bien}, "todas": {"prueba": bien}}).superada
    assert not informe_paridad({"cpu": {"prueba": bien}, "todas": {"prueba": bien, "referencia": mal}}).superada


def test_un_informe_vacio_es_un_error() -> None:
    with pytest.raises(ValueError):
        informe_paridad({})
    with pytest.raises(ValueError):
        informe_paridad({"cpu": {}})


def test_el_informe_se_serializa_como_paridad_json() -> None:
    bien = comparar(ORIGINAL, ORIGINAL.copy(), ETIQUETAS)
    mal = comparar(ORIGINAL, _desplazar(ORIGINAL, 0, 0.05))

    datos = json.loads(json.dumps(informe_paridad({"cpu": {"prueba": bien}, "todas": {"referencia": mal}}).a_dict()))

    assert datos["tolerancia"] == pytest.approx(TOLERANCIA_PARIDAD)
    assert datos["superada"] is False
    assert set(datos["unidades"]) == {"cpu", "todas"}
    prueba = datos["unidades"]["cpu"]["prueba"]
    assert set(prueba) == {
        "casos", "clase_distinta", "max_diferencia", "media_diferencia",
        "aciertos_original", "aciertos_exportado", "cambios_95", "superado",
    }
    assert prueba["clase_distinta"] == []
    assert prueba["cambios_95"] == {"a_aceptada": 0, "a_descartada": 0}
    assert datos["unidades"]["todas"]["referencia"]["superado"] is False
    assert datos["unidades"]["todas"]["referencia"]["aciertos_original"] is None
