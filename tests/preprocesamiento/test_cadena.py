"""Orquestador: clip → detección → keypoints → ventana (FR-002, FR-015, FR-022, FR-023)."""

import numpy as np
import pytest
from pathlib import Path

from signai_ml.preproceso.almacen import Almacen, Versiones
from signai_ml.preproceso.cadena import Cadena
from signai_ml.preproceso.clips import leer_etiquetas
from signai_ml.preproceso.errores import ErrorPreproceso

from .dobles import MACOS, ExtractorDoble, crear_corpus

V = Versiones("signai-ml/preproceso/1", "signai-ml/keypoints/1", "1.0.0", "1.0.0", MACOS)
CLIPS = [("1-Yo__IMG_1008__rep01", 82), ("1-Yo__IMG_1008__rep02", 35), ("2-Tu__IMG_1010__rep01", 203)]


@pytest.fixture
def entorno(tmp_path: Path):
    base = crear_corpus(tmp_path, CLIPS)
    clips = leer_etiquetas(base)
    doble = ExtractorDoble(dict(CLIPS))
    almacen = Almacen(tmp_path / "datos")
    cadena = Cadena(doble, almacen, tmp_path / "cache", V)
    return clips, doble, almacen, cadena


def test_cada_clip_da_keypoints_y_ventana(entorno) -> None:
    clips, _, almacen, cadena = entorno
    resumen = cadena.procesar(clips, procesos=2)
    assert sorted(resumen.procesados) == sorted(c.clip_id for c in clips)
    assert resumen.errores == {}
    for c in clips:
        r = resumen.resultados[c.clip_id]
        assert r.keypoints.shape == (-(-c.n_fotogramas // 2), 96)
        assert r.ventana.valores.shape == (75, 96)
        assert almacen.vigente(c, V)


def test_clip_alterado_se_senala_y_no_produce_resultado(entorno) -> None:
    clips, doble, almacen, cadena = entorno
    clips[0].ruta.write_bytes(b"otro contenido")
    resumen = cadena.procesar(clips, procesos=1)
    assert clips[0].clip_id in resumen.errores and "md5" in resumen.errores[clips[0].clip_id]
    assert not almacen.vigente(clips[0], V)
    assert clips[0].clip_id not in doble.llamadas


def test_los_vigentes_se_saltan_y_el_resultado_es_el_mismo(entorno) -> None:
    clips, doble, almacen, cadena = entorno
    cadena.procesar(clips[:1], procesos=1)  # «interrupción» tras el primero
    antes = almacen.leer_keypoints(clips[0]).copy()
    doble.llamadas.clear()

    resumen = cadena.procesar(clips, procesos=2)

    assert resumen.saltados == [clips[0].clip_id]
    assert clips[0].clip_id not in doble.llamadas
    assert np.array_equal(almacen.leer_keypoints(clips[0]), antes)


def test_dos_pasadas_dan_ficheros_identicos(entorno, tmp_path: Path) -> None:
    clips, doble, _, cadena = entorno
    cadena.procesar(clips, procesos=1)
    uno = cadena.ensamblar(clips, {})
    datos_uno = uno.read_bytes()

    otra = Cadena(doble, Almacen(tmp_path / "otra"), tmp_path / "cache-otra", V)
    otra.procesar(clips, procesos=3)
    dos = otra.ensamblar(clips, {})
    assert dos.read_bytes() == datos_uno


def test_solo_cuenta_los_fotogramas_que_pasan_por_vision(entorno, tmp_path: Path) -> None:
    clips, doble, _, cadena = entorno
    primera = cadena.procesar(clips, procesos=1)
    assert primera.fotogramas_extraidos == sum(-(-c.n_fotogramas // 2) for c in clips)
    from dataclasses import replace
    otra = Cadena(doble, Almacen(tmp_path / "otra"), tmp_path / "cache", replace(V, normalizacion="9.9.9"))
    segunda = otra.procesar(clips, procesos=1)  # keypoints no vigentes, detecciones en caché
    assert sorted(segunda.procesados) == sorted(c.clip_id for c in clips)
    assert segunda.fotogramas_extraidos == 0


def test_ensamblar_exige_todos_los_clips_vigentes(entorno) -> None:
    clips, _, almacen, cadena = entorno
    cadena.procesar(clips[:2], procesos=1)
    with pytest.raises(ErrorPreproceso, match="2-Tu__IMG_1010__rep01"):
        cadena.ensamblar(clips, {})
    assert not (almacen.dir_ventanas / "ventanas.npy").exists()


def test_ensamblar_apila_las_ventanas_en_el_orden_de_los_clips(entorno) -> None:
    clips, _, almacen, cadena = entorno
    resumen = cadena.procesar(clips, procesos=2)
    ruta = cadena.ensamblar(clips, {})
    ventanas = np.load(ruta)
    assert ventanas.shape == (3, 75, 96) and ventanas.dtype == np.float32
    for i, c in enumerate(clips):
        assert np.array_equal(ventanas[i], resumen.resultados[c.clip_id].ventana.valores)


def test_un_macos_distinto_del_esperado_es_error(entorno, tmp_path: Path) -> None:
    clips, doble, _, _ = entorno
    from dataclasses import replace
    cadena = Cadena(doble, Almacen(tmp_path / "x"), tmp_path / "c", replace(V, macos="26.0.0"))
    resumen = cadena.procesar(clips[:1], procesos=1)
    assert "macOS" in resumen.errores[clips[0].clip_id]
