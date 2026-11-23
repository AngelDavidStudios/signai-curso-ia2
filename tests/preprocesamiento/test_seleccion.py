"""Selección de los 48 keypoints (E-2…E-6; FR-006…FR-008; research R-05).

Port uno a uno de `signai-ios/signai-iosTests/Captura/KeypointSelectorTests.swift` (20 casos), con
las mismas detecciones fabricadas en coordenadas del detector (origen abajo a la izquierda). Así la
selección del corpus y la de la app se prueban con los mismos casos. Al final, los casos propios de
la cadena del corpus: fotograma fallido, forma de la salida y coordenadas sin recortar.
"""

import numpy as np
import pytest

from signai_ml.preproceso.deteccion import (
    ORDEN_MANO,
    ORDEN_POSE,
    CuerpoDetectado,
    Deteccion,
    FotogramaDetectado,
    ManoDetectada,
    PuntoDetector,
)
from signai_ml.preproceso.seleccion import Selector, seleccionar_clip

P = PuntoDetector


# ── Fixtures, como PoseDetectionFixture.swift ─────────────────────────────────────────────


def cuerpo(puntos: dict[str, PuntoDetector]) -> CuerpoDetectado:
    return CuerpoDetectado(puntos)


def cuerpo_completo(centro: PuntoDetector = P(0.5, 0.6), separacion: float = 0.1) -> CuerpoDetectado:
    cx, cy, s = centro.x, centro.y, separacion
    return CuerpoDetectado({
        "leftShoulder": P(cx + s, cy), "rightShoulder": P(cx - s, cy),
        "leftElbow": P(cx + 2 * s, cy - 0.1), "rightElbow": P(cx - 2 * s, cy - 0.1),
        "leftWrist": P(cx + 2 * s, cy - 0.2), "rightWrist": P(cx - 2 * s, cy - 0.2),
    })


def mano(lado: str | None, muneca: PuntoDetector, escala: float = 1) -> ManoDetectada:
    puntos = {
        nombre: P(muneca.x + 0.001 * i * escala, muneca.y + 0.002 * i * escala)
        for i, nombre in enumerate(ORDEN_MANO)
    }
    return ManoDetectada(lado, puntos)


def sin(m: ManoDetectada, nombre: str) -> ManoDetectada:
    return ManoDetectada(m.lado, {k: v for k, v in m.puntos.items() if k != nombre})


def desplazar(p: PuntoDetector, delta: float = 0.01) -> PuntoDetector:
    return P(p.x + delta, p.y + delta)


def seleccionar(cuerpos=(), manos=()) -> np.ndarray:
    return Selector().seleccionar(FotogramaDetectado(0, False, tuple(cuerpos), tuple(manos)))


# Índices de E-2
I_HOMBRO_IZQ, I_HOMBRO_DER, I_CODO_IZQ, I_CODO_DER, I_MUNECA_IZQ, I_MUNECA_DER = range(6)
MANO_IZQ, MANO_DER = 6, 27


def punto(valores: np.ndarray, indice: int) -> tuple[float, float]:
    return float(valores[2 * indice]), float(valores[2 * indice + 1])


def ausente(valores: np.ndarray, indice: int) -> bool:
    return punto(valores, indice) == (-1.0, -1.0)


def bloque_ausente(valores: np.ndarray, inicio: int) -> bool:
    return all(ausente(valores, inicio + k) for k in range(21))


def assert_volteado(valores: np.ndarray, indice: int, p: PuntoDetector) -> None:
    x, y = punto(valores, indice)
    assert x == pytest.approx(p.x, abs=1e-6)
    assert y == pytest.approx(1 - p.y, abs=1e-6)


# ── Orden (D-10) ──────────────────────────────────────────────────────────────────────────


def test_cuerpo_y_dos_manos_caen_en_sus_indices_del_esquema() -> None:
    c = cuerpo_completo()
    izq = mano("left", P(0.7, 0.3))
    der = mano("right", P(0.3, 0.3))
    v = seleccionar([c], [izq, der])
    assert v.shape == (96,)
    assert_volteado(v, 0, c.puntos["leftShoulder"])
    assert_volteado(v, 5, c.puntos["rightWrist"])
    assert_volteado(v, 6, izq.puntos["wrist"])
    assert_volteado(v, 26, izq.puntos["littleTip"])
    assert_volteado(v, 27, der.puntos["wrist"])
    assert_volteado(v, 31, der.puntos["thumbTip"])
    assert_volteado(v, 47, der.puntos["littleTip"])


# ── Ausencia y confianza (D-11, D-16) ────────────────────────────────────────────────────


def test_un_punto_que_falta_en_el_cuerpo_es_menos_uno() -> None:
    puntos = dict(cuerpo_completo().puntos)
    del puntos["rightElbow"]
    v = seleccionar([cuerpo(puntos)])
    assert ausente(v, I_CODO_DER)
    assert not ausente(v, I_CODO_IZQ)


def test_sin_manos_los_dos_bloques_de_mano_son_menos_uno() -> None:
    v = seleccionar([cuerpo_completo()])
    assert (v[12:] == -1).all() and len(v[12:]) == 84


def test_nadie_da_48_puntos_ausentes() -> None:
    assert (seleccionar() == -1).all()


def test_un_punto_de_dedo_que_falta_es_menos_uno_y_el_resto_de_la_mano_queda() -> None:
    m = sin(mano("left", P(0.7, 0.3)), "indexTip")
    v = seleccionar([], [m])
    assert ausente(v, MANO_IZQ + 8)  # punta del índice
    assert not ausente(v, MANO_IZQ + 7)  # DIP del índice


def test_todo_punto_devuelto_se_conserva() -> None:
    v = seleccionar([cuerpo_completo()], [mano("right", P(0.3, 0.3))])
    assert all(not ausente(v, i) for i in range(6))
    assert all(not ausente(v, MANO_DER + k) for k in range(21))


# ── Eje vertical (D-12) ──────────────────────────────────────────────────────────────────


def test_el_eje_vertical_se_invierte_y_el_horizontal_se_conserva() -> None:
    c = cuerpo({"leftShoulder": P(0.3, 0.2), "rightShoulder": P(0.6, 0.0), "leftElbow": P(0.9, 1.0)})
    v = seleccionar([c])
    assert punto(v, I_HOMBRO_IZQ) == pytest.approx((0.3, 0.8), abs=1e-6)
    assert punto(v, I_HOMBRO_DER)[1] == pytest.approx(1.0, abs=1e-6)
    assert punto(v, I_CODO_IZQ)[1] == pytest.approx(0.0, abs=1e-6)


def test_un_punto_ausente_no_se_invierte() -> None:
    v = seleccionar([cuerpo({"leftShoulder": P(0.3, 0.2)})])
    assert punto(v, I_HOMBRO_DER) == (-1.0, -1.0)


# ── Persona (D-13, E-5) ──────────────────────────────────────────────────────────────────


def test_gana_el_cuerpo_mas_grande_aunque_llegue_segundo() -> None:
    lejos = cuerpo_completo(P(0.2, 0.6), 0.02)
    cerca = cuerpo_completo(P(0.6, 0.6), 0.1)
    v = seleccionar([lejos, cerca])
    assert_volteado(v, I_HOMBRO_IZQ, cerca.puntos["leftShoulder"])


def test_a_igualdad_se_queda_el_primero() -> None:
    primero = cuerpo_completo(P(0.3, 0.6))
    segundo = cuerpo_completo(P(0.7, 0.6))
    v = seleccionar([primero, segundo])
    assert_volteado(v, I_HOMBRO_IZQ, primero.puntos["leftShoulder"])


def test_un_cuerpo_con_un_punto_gana_a_uno_sin_ninguno() -> None:
    vacio = CuerpoDetectado({})
    solo = cuerpo({"rightWrist": P(0.4, 0.4)})
    v = seleccionar([vacio, solo])
    assert_volteado(v, I_MUNECA_DER, solo.puntos["rightWrist"])


# ── Manos (D-14, E-6) ────────────────────────────────────────────────────────────────────


def test_con_lados_distintos_manda_el_lado_aunque_esten_cruzadas() -> None:
    c = cuerpo_completo()
    izq_cruzada = mano("left", c.puntos["rightWrist"])
    der_cruzada = mano("right", c.puntos["leftWrist"])
    v = seleccionar([c], [izq_cruzada, der_cruzada])
    assert_volteado(v, MANO_IZQ, izq_cruzada.puntos["wrist"])
    assert_volteado(v, MANO_DER, der_cruzada.puntos["wrist"])


def test_dos_manos_del_mismo_lado_se_reparten_por_la_muneca_de_pose_mas_cercana() -> None:
    c = cuerpo_completo()
    cerca_izq = mano("right", desplazar(c.puntos["leftWrist"]))
    cerca_der = mano("right", desplazar(c.puntos["rightWrist"]))
    v = seleccionar([c], [cerca_der, cerca_izq])
    assert_volteado(v, MANO_IZQ, cerca_izq.puntos["wrist"])
    assert_volteado(v, MANO_DER, cerca_der.puntos["wrist"])


def test_una_mano_sin_lado_va_a_la_muneca_de_pose_mas_cercana() -> None:
    c = cuerpo_completo()
    con_lado = mano("left", desplazar(c.puntos["leftWrist"]))
    sin_lado = mano(None, desplazar(c.puntos["rightWrist"]))
    v = seleccionar([c], [sin_lado, con_lado])
    assert_volteado(v, MANO_IZQ, con_lado.puntos["wrist"])
    assert_volteado(v, MANO_DER, sin_lado.puntos["wrist"])


def test_cuando_decide_la_cercania_el_lado_informado_no_cuenta() -> None:
    c = cuerpo_completo()
    marcada_izq = mano("left", desplazar(c.puntos["rightWrist"]))
    sin_lado = mano(None, desplazar(c.puntos["leftWrist"]))
    v = seleccionar([c], [marcada_izq, sin_lado])
    assert_volteado(v, MANO_DER, marcada_izq.puntos["wrist"])
    assert_volteado(v, MANO_IZQ, sin_lado.puntos["wrist"])


def test_sin_muneca_la_posicion_de_la_mano_es_el_centro_de_sus_puntos() -> None:
    c = cuerpo_completo()
    completa = mano("right", desplazar(c.puntos["leftWrist"]))
    sin_muneca = sin(completa, "wrist")
    otra = mano("right", desplazar(c.puntos["rightWrist"]))
    v = seleccionar([c], [otra, sin_muneca])
    assert ausente(v, MANO_IZQ)
    assert_volteado(v, MANO_IZQ + 1, sin_muneca.puntos["thumbCMC"])
    assert_volteado(v, MANO_DER, otra.puntos["wrist"])


def test_sin_munecas_de_pose_la_mano_mayor_se_queda_el_lado() -> None:
    solo_hombros = cuerpo({"leftShoulder": P(0.6, 0.6), "rightShoulder": P(0.4, 0.6)})
    pequena = mano("left", P(0.2, 0.2), escala=1)
    grande = mano("left", P(0.6, 0.2), escala=3)
    v = seleccionar([solo_hombros], [pequena, grande])
    assert_volteado(v, MANO_IZQ, grande.puntos["wrist"])
    assert bloque_ausente(v, MANO_DER)


def test_sin_pose_una_mano_sin_lado_se_descarta() -> None:
    con_lado = mano("right", P(0.3, 0.3))
    sin_lado = mano(None, P(0.7, 0.3))
    v = seleccionar([], [sin_lado, con_lado])
    assert_volteado(v, MANO_DER, con_lado.puntos["wrist"])
    assert bloque_ausente(v, MANO_IZQ)


def test_las_distancias_se_miden_antes_de_invertir_el_eje() -> None:
    c = cuerpo({"leftWrist": P(0.6, 0.2), "rightWrist": P(0.4, 0.8)})
    m = mano(None, P(0.6, 0.25))
    v = seleccionar([c], [m])
    assert_volteado(v, MANO_IZQ, m.puntos["wrist"])
    assert bloque_ausente(v, MANO_DER)


# ── Determinismo ─────────────────────────────────────────────────────────────────────────


def test_la_misma_deteccion_da_los_mismos_puntos() -> None:
    c = cuerpo_completo()
    manos = [mano("right", desplazar(c.puntos["leftWrist"])), mano("right", desplazar(c.puntos["rightWrist"]))]
    assert np.array_equal(seleccionar([c], manos), seleccionar([c], manos))


# ── Propios de la cadena del corpus (E-4, FR-006) ────────────────────────────────────────


def test_un_fotograma_fallido_da_96_menos_uno() -> None:
    v = Selector().seleccionar(FotogramaDetectado(0, True, (cuerpo_completo(),), ()))
    assert (v == -1).all()


def test_el_orden_de_los_nombres_es_el_de_e2() -> None:
    assert ORDEN_POSE == ("leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist")
    assert ORDEN_MANO[0] == "wrist" and ORDEN_MANO[4] == "thumbTip" and ORDEN_MANO[8] == "indexTip"
    assert ORDEN_MANO[20] == "littleTip" and len(ORDEN_MANO) == 21


def test_una_coordenada_fuera_de_cero_uno_se_conserva_sin_recortar() -> None:
    v = seleccionar([cuerpo({"leftShoulder": P(1.02, -0.01)})])
    assert punto(v, I_HOMBRO_IZQ) == pytest.approx((1.02, 1.01), abs=1e-6)


def test_seleccionar_clip_da_t_por_96_float32() -> None:
    fotogramas = tuple(FotogramaDetectado(2 * i, False, (cuerpo_completo(),), ()) for i in range(5))
    det = Deteccion(
        version_extractor="x", macos="27.2", md5="0" * 32, fotogramas_decodificados=10, cada=2,
        fotogramas=fotogramas,
    )
    m = seleccionar_clip(det)
    assert m.shape == (5, 96) and m.dtype == np.float32
    assert ((m == -1) | np.isfinite(m)).all()
