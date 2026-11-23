"""Selección por rotación, elegibilidad y cierre (TT-22, research R-05; data-model «Agregado»). Sin disco."""

import pytest

from signai_ml.reconocedor import LIMITE_PARAMETROS
from signai_ml.reconocedor.seleccion import (
    ResultadoRotacion,
    agregar,
    elegir_candidato,
    estancado,
    mejores_por_ronda,
)

SEMILLAS = (215, 216, 217)
ORDEN = {"x": (1, 0), "y": (1, 1), "z": (2, 0)}


def r(cfg: str, semilla: int, rotacion: str, top1: list[float], acierto: list[float] | None = None,
      ronda: int = 1, parametros: int = 381_121, f1: list[float] | None = None) -> ResultadoRotacion:
    acierto = acierto if acierto is not None else [0.8] * len(top1)
    curvas = {
        "top1": tuple(top1), "f1_macro": tuple(f1 if f1 is not None else top1),
        "acierto_95": tuple(acierto), "falsas_95": tuple(0.01 for _ in top1),
    }
    return ResultadoRotacion(f"id-{cfg}-{semilla}-{rotacion}", f"{cfg}-s{semilla}-{rotacion}", cfg, semilla,
                             rotacion, ronda, parametros, curvas)


def seis(cfg: str, a: list[float], b: list[float], **kw) -> list[ResultadoRotacion]:
    """Las 6 corridas de una configuración: la rotación a con la curva `a` y la b con la `b`."""
    return [r(cfg, s, rot, curva, **kw) for s in SEMILLAS for rot, curva in (("a", a), ("b", b))]


def test_solo_agrega_configuraciones_con_sus_6_corridas() -> None:
    resultados = seis("x", [0.5, 0.8], [0.5, 0.8]) + [r("y", 215, "a", [0.9, 0.9])]
    (agregado,) = agregar(resultados, ORDEN)
    assert agregado.configuracion == "x" and len(agregado.corridas) == 6


def test_curva_media_y_epocas_elegidas() -> None:
    # Media por época: [0,6 · 0,8 · 0,7] → E* = 2.
    (agregado,) = agregar(seis("x", [0.5, 0.9, 0.7], [0.7, 0.7, 0.7]), ORDEN)
    assert agregado.epocas_elegidas == 2
    assert agregado.medias["top1"] == pytest.approx(0.8)
    assert set(agregado.medias) == {"top1", "f1_macro", "acierto_95", "falsas_95"}
    assert agregado.por_rotacion == {"a": pytest.approx(0.9), "b": pytest.approx(0.7)}
    assert agregado.diferencia_rotaciones == pytest.approx(0.2)


def test_epocas_elegidas_empate_por_acierto_y_despues_la_menor() -> None:
    curva = [0.8, 0.8, 0.8]
    (por_acierto,) = agregar(seis("x", curva, curva, acierto=[0.7, 0.9, 0.9]), ORDEN)
    assert por_acierto.epocas_elegidas == 2
    (la_menor,) = agregar(seis("x", curva, curva), ORDEN)
    assert la_menor.epocas_elegidas == 1


def test_curvas_de_distinta_longitud_usan_las_epocas_comunes() -> None:
    resultados = seis("x", [0.5, 0.8], [0.5, 0.8])
    resultados[0] = r("x", 215, "a", [0.5, 0.8, 0.99])
    (agregado,) = agregar(resultados, ORDEN)
    assert agregado.epocas_elegidas == 2


def test_semilla_final_la_de_mejor_media_de_sus_dos_rotaciones() -> None:
    resultados = seis("x", [0.8], [0.8])
    resultados[2] = r("x", 216, "a", [0.9])  # 216: (0,9 + 0,8) / 2 = 0,85
    (agregado,) = agregar(resultados, ORDEN)
    assert agregado.semilla_final == 216
    (empate,) = agregar(seis("x", [0.8], [0.8]), ORDEN)
    assert empate.semilla_final == 215  # a igualdad, la primera de las fijas


def test_elegibilidad_solo_por_parametros() -> None:
    resultados = (
        seis("x", [0.8], [0.8])
        + seis("y", [0.9], [0.9], parametros=LIMITE_PARAMETROS + 1)
        + seis("z", [0.65], [0.65], acierto=[0.45], ronda=2)
    )
    por_nombre = {a.configuracion: a for a in agregar(resultados, ORDEN)}
    assert por_nombre["x"].elegible and por_nombre["x"].motivo is None
    assert not por_nombre["y"].elegible and "parámetros" in por_nombre["y"].motivo
    # el acierto al 95 % se informa, no descarta (research R-05, enmienda del 2026-11-21)
    assert por_nombre["z"].elegible and por_nombre["z"].motivo is None


def test_la_diferencia_entre_rotaciones_no_quita_la_elegibilidad() -> None:
    # SC-002 se informa; no descarta (research R-05, tasks T037).
    (agregado,) = agregar(seis("x", [0.95], [0.60]), ORDEN)
    assert agregado.diferencia_rotaciones == pytest.approx(0.35) and agregado.elegible


def test_candidato_por_top1_medio() -> None:
    resultados = seis("x", [0.80], [0.84]) + seis("y", [0.83], [0.83])
    assert elegir_candidato(agregar(resultados, ORDEN)).configuracion == "y"  # 0,83 > 0,82


def test_desempates_entre_configuraciones() -> None:
    # mismo Top-1 medio → menor diferencia entre rotaciones
    empate = seis("x", [0.9], [0.7]) + seis("y", [0.8], [0.8])
    assert elegir_candidato(agregar(empate, ORDEN)).configuracion == "y"
    # … → mayor acierto al 95 %
    empate = seis("x", [0.8], [0.8], acierto=[0.8]) + seis("y", [0.8], [0.8], acierto=[0.85])
    assert elegir_candidato(agregar(empate, ORDEN)).configuracion == "y"
    # … → mayor F1 macro
    empate = seis("x", [0.8], [0.8], f1=[0.7]) + seis("y", [0.8], [0.8], f1=[0.75])
    assert elegir_candidato(agregar(empate, ORDEN)).configuracion == "y"
    # … → la declarada antes
    empate = seis("y", [0.8], [0.8]) + seis("x", [0.8], [0.8])
    assert elegir_candidato(agregar(empate, ORDEN)).configuracion == "x"


def test_sin_elegibles_no_hay_candidato() -> None:
    assert elegir_candidato(agregar(seis("x", [0.9], [0.9], parametros=LIMITE_PARAMETROS + 1), ORDEN)) is None


def test_mejor_media_elegible_acumulada_por_ronda() -> None:
    resultados = seis("x", [0.80], [0.80]) + seis("z", [0.79], [0.79], ronda=2)
    assert mejores_por_ronda(agregar(resultados, ORDEN), [1, 2]) == [(1, pytest.approx(0.80)), (2, pytest.approx(0.80))]


@pytest.mark.parametrize(
    ("mejores", "cerrar"),
    [
        ([(1, 0.80), (2, 0.803), (3, 0.804)], True),
        ([(1, 0.80), (2, 0.803)], False),
        ([(1, 0.80), (2, 0.81), (3, 0.812)], False),
        ([(1, 0.80), (2, 0.802), (3, 0.812)], False),
        ([(1, None), (2, None), (3, None)], False),
        ([(1, None), (2, 0.80), (3, 0.801)], False),
        ([(1, None), (2, 0.80), (3, 0.801), (4, 0.802)], True),
        ([], False),
    ],
)
def test_estancamiento(mejores: list, cerrar: bool) -> None:
    assert estancado(mejores) is cerrar
