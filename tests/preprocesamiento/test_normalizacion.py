"""Normalización de la ventana, paso a paso (contrato `normalizacion.md` 2.0.0, N-1…N-7).

Los valores esperados se calculan a mano a partir del texto del contrato, no con la implementación.
El contrato adopta la preparación validada del laboratorio KPoseFormer (`09_preparar.py`).
"""

import inspect

import numpy as np
import pytest

from signai_ml.preproceso.normalizacion import (
    Imputacion,
    NormalizacionHombros,
    Normalizador,
    RespaldoMuneca,
    Remuestreo,
    Secuencia,
    secuencia_desde_keypoints,
)

AUS = -1.0


def keypoints(t: int) -> np.ndarray:
    """(T, 96) con todos los puntos ausentes."""
    return np.full((t, 96), AUS, dtype=np.float32)


def poner(kp: np.ndarray, punto: int, x, y) -> None:
    kp[:, 2 * punto] = x
    kp[:, 2 * punto + 1] = y


def hombros(kp: np.ndarray, izq=(0.6, 0.5), der=(0.4, 0.5)) -> None:
    poner(kp, 0, *izq)
    poner(kp, 1, *der)


def secuencia(kp: np.ndarray) -> Secuencia:
    return secuencia_desde_keypoints(kp)


# ── N-1 · Entrada ─────────────────────────────────────────────────────────────────────────


def test_un_punto_es_presente_si_su_x_es_distinta_de_menos_uno() -> None:
    kp = keypoints(2)
    poner(kp, 0, [0.4, AUS], [0.5, AUS])
    s = secuencia(kp)
    assert s.valores.shape == (2, 48, 2)
    assert s.presente[:, 0].tolist() == [True, False]


# ── N-3 · Respaldo de muñeca ──────────────────────────────────────────────────────────────


def test_la_muneca_de_pose_ausente_toma_la_de_la_mano_del_mismo_lado() -> None:
    kp = keypoints(3)
    poner(kp, 4, [0.70, AUS, AUS], [0.40, AUS, AUS])  # muñeca izquierda de pose
    poner(kp, 6, [0.71, 0.72, AUS], [0.41, 0.42, AUS])  # muñeca de la mano izquierda
    poner(kp, 27, [0.31, 0.32, 0.33], [0.41, 0.42, 0.43])  # muñeca de la mano derecha
    s = RespaldoMuneca().aplicar(secuencia(kp))
    np.testing.assert_allclose(s.valores[:, 4], [[0.70, 0.40], [0.72, 0.42], [AUS, AUS]], atol=1e-6)
    np.testing.assert_allclose(s.valores[:, 5], [[0.31, 0.41], [0.32, 0.42], [0.33, 0.43]], atol=1e-6)
    assert s.presente[:, 4].tolist() == [True, True, False]


# ── N-4 · Imputación ──────────────────────────────────────────────────────────────────────


def test_hueco_interior_con_la_recta_y_extremos_con_el_vecino() -> None:
    kp = keypoints(5)
    poner(kp, 30, [AUS, 0.2, AUS, AUS, 0.5], [AUS, 0.3, AUS, AUS, 0.6])
    s = Imputacion().aplicar(secuencia(kp))
    np.testing.assert_allclose(s.valores[:, 30, 0], [0.2, 0.2, 0.3, 0.4, 0.5], atol=1e-6)
    np.testing.assert_allclose(s.valores[:, 30, 1], [0.3, 0.3, 0.4, 0.5, 0.6], atol=1e-6)


def test_una_mano_que_no_aparece_se_ancla_a_la_muneca_de_pose_de_su_lado() -> None:
    kp = keypoints(3)
    hombros(kp)
    poner(kp, 4, 0.7, 0.4)  # muñeca izquierda de pose
    poner(kp, 5, [0.3, AUS, 0.3], [0.4, AUS, 0.6])  # muñeca derecha de pose, con un hueco
    s = Imputacion().aplicar(secuencia(kp))
    for k in range(6, 27):  # mano izquierda, toda ausente → muñeca 4
        np.testing.assert_allclose(s.valores[:, k], [[0.7, 0.4]] * 3, atol=1e-6)
    for k in range(27, 48):  # mano derecha, toda ausente → muñeca 5 ya imputada
        np.testing.assert_allclose(s.valores[:, k], [[0.3, 0.4], [0.3, 0.5], [0.3, 0.6]], atol=1e-6)


def test_sin_muneca_el_ancla_es_el_centro_de_hombros_y_sin_hombros_el_medio() -> None:
    kp = keypoints(2)
    hombros(kp, izq=([0.6, AUS], [0.5, AUS]), der=([0.4, AUS], [0.5, AUS]))
    s = Imputacion().aplicar(secuencia(kp))
    # Punto 10 (mano izquierda): ancla 4 ausente → centro de hombros (0,5; 0,5) en f0; en f1 no hay
    # centro (los hombros faltaban antes de imputar) → (0,5; 0,5) por defecto.
    np.testing.assert_allclose(s.valores[:, 10], [[0.5, 0.5], [0.5, 0.5]], atol=1e-6)
    kp2 = keypoints(1)
    hombros(kp2, izq=(0.8, 0.2), der=(0.6, 0.2))
    s2 = Imputacion().aplicar(secuencia(kp2))
    np.testing.assert_allclose(s2.valores[0, 40], [0.7, 0.2], atol=1e-6)  # centro de hombros


def test_tras_imputar_ningun_punto_vale_menos_uno() -> None:
    rng = np.random.default_rng(3)
    kp = rng.uniform(0.2, 0.8, size=(20, 96)).astype(np.float32)
    kp.reshape(20, 48, 2)[rng.random((20, 48)) < 0.4] = AUS
    s = Imputacion().aplicar(RespaldoMuneca().aplicar(secuencia(kp)))
    assert not (s.valores == AUS).any()


# ── N-5 · Normalización por hombros ──────────────────────────────────────────────────────


def test_centro_de_hombros_por_fotograma_y_ancho_mediano() -> None:
    kp = keypoints(3)
    # hombros que se desplazan y cambian de ancho: anchos 0,2 · 0,4 · 0,3 → mediana 0,3
    poner(kp, 0, [0.6, 0.7, 0.65], 0.5)
    poner(kp, 1, [0.4, 0.3, 0.35], 0.5)
    poner(kp, 10, [0.5, 0.5, 0.8], [0.8, 0.5, 0.5])
    s = NormalizacionHombros().aplicar(Imputacion().aplicar(secuencia(kp)))
    np.testing.assert_allclose(s.valores[:, 0, 0], [0.1 / 0.3, 0.2 / 0.3, 0.15 / 0.3], atol=1e-6)
    np.testing.assert_allclose(s.valores[:, 10], [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]], atol=1e-6)
    assert not s.sin_normalizar


def test_sin_ancho_valido_no_se_normaliza() -> None:
    kp = keypoints(4)
    poner(kp, 0, 0.5, 0.5)
    poner(kp, 1, 0.5, 0.5)  # los dos hombros en el mismo sitio: ancho 0
    poner(kp, 7, 0.3, 0.9)
    s = NormalizacionHombros().aplicar(Imputacion().aplicar(secuencia(kp)))
    assert s.sin_normalizar
    np.testing.assert_allclose(s.valores[:, 7], [[0.3, 0.9]] * 4, atol=1e-6)


# ── N-6 · Remuestreo ──────────────────────────────────────────────────────────────────────


def test_remuestreo_de_75_fotogramas_es_la_identidad_exacta() -> None:
    rng = np.random.default_rng(1)
    s = secuencia(rng.uniform(0.1, 0.9, size=(75, 96)).astype(np.float32))
    assert np.array_equal(Remuestreo(75).aplicar(s).valores, s.valores)


def test_remuestreo_interpola_en_k_por_t_menos_uno_entre_74_y_repite_un_fotograma() -> None:
    kp = keypoints(3)
    poner(kp, 0, [0.0, 0.5, 1.0], 0.2)
    r = Remuestreo(75).aplicar(secuencia(kp))
    np.testing.assert_allclose(r.valores[:, 0, 0], [k * 2 / 74 * 0.5 for k in range(75)], atol=1e-12)
    uno = keypoints(1)
    poner(uno, 0, 0.4, 0.5)
    np.testing.assert_allclose(Remuestreo(75).aplicar(secuencia(uno)).valores[:, 0, 0], 0.4)


# ── Normalizador completo · N-2 y N-7 ────────────────────────────────────────────────────


def test_hombros_quedan_en_mas_y_menos_medio() -> None:
    kp = keypoints(41)
    hombros(kp)  # izquierdo a la derecha de la imagen, sin espejo
    v = Normalizador().normalizar(kp)
    np.testing.assert_allclose(v.valores[:, 0:4], np.tile([0.5, 0.0, -0.5, 0.0], (75, 1)), atol=1e-6)


def test_salida_75_por_96_float32_finita() -> None:
    kp = keypoints(18)
    hombros(kp)
    poner(kp, 4, 0.7, 0.4)
    poner(kp, 5, 0.3, 0.4)
    v = Normalizador().normalizar(kp)
    assert v.valores.shape == (75, 96) and v.valores.dtype == np.float32
    assert np.isfinite(v.valores).all()
    # mano izquierda anclada a su muñeca de pose: (0,7 − 0,5)/0,2 = 1; (0,4 − 0,5)/0,2 = −0,5
    np.testing.assert_allclose(v.valores[0, 12:14], [1.0, -0.5], atol=1e-6)


def test_todo_ausente_da_una_ventana_finita_sin_normalizar() -> None:
    v = Normalizador().normalizar(keypoints(30))
    np.testing.assert_allclose(v.valores, 0.5)
    assert v.sin_normalizar


def test_coordenadas_fuera_de_cero_uno_se_tratan_como_cualquier_otra() -> None:
    kp = keypoints(75)
    hombros(kp, izq=(1.02, 0.5), der=(0.82, 0.5))
    v = Normalizador().normalizar(kp)
    np.testing.assert_allclose(v.valores[0, 0:4], [0.5, 0.0, -0.5, 0.0], atol=1e-6)


def test_el_orden_de_los_pasos_es_fijo_y_no_se_configura_desde_fuera() -> None:
    assert [type(p) for p in Normalizador.PASOS] == [RespaldoMuneca, Imputacion, NormalizacionHombros, Remuestreo]
    assert list(inspect.signature(Normalizador).parameters) == []


@pytest.mark.parametrize("t", [1, 18, 75, 203])
def test_cualquier_longitud_da_75_fotogramas(t: int) -> None:
    kp = keypoints(t)
    hombros(kp)
    assert Normalizador().normalizar(kp).valores.shape == (75, 96)
