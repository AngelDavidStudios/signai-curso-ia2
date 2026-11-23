"""Verificación del resultado versionado (SC-001…SC-003, SC-006; contracts/cli.md C-4)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from signai_ml.preproceso.almacen import Almacen, Versiones
from signai_ml.preproceso.cadena import Cadena
from signai_ml.preproceso.clips import leer_etiquetas
from signai_ml.preproceso.verificacion import verificar

from .dobles import MACOS, ExtractorDoble, crear_corpus

V = Versiones("signai-ml/preproceso/1", "signai-ml/keypoints/1", "1.0.0", "1.0.0", MACOS)
CLIPS = [("1-Yo__IMG_1__rep01", 20), ("1-Yo__IMG_1__rep02", 15), ("2-Tu__IMG_2__rep01", 30)]


@pytest.fixture
def corpus(tmp_path: Path):
    clips = leer_etiquetas(crear_corpus(tmp_path, CLIPS))
    almacen = Almacen(tmp_path / "datos")
    cadena = Cadena(ExtractorDoble(dict(CLIPS)), almacen, tmp_path / "cache", V)
    cadena.procesar(clips)
    cadena.ensamblar(clips, {})
    return clips, almacen


def test_un_resultado_correcto_no_tiene_problemas(corpus) -> None:
    clips, almacen = corpus
    assert verificar(clips, almacen) == []


def test_detecta_una_ventana_de_menos(corpus) -> None:
    clips, almacen = corpus
    extra = replace(clips[0], clip_id="3-Ustedes__IMG_9__rep01", clase="3-Ustedes")
    assert any("3-Ustedes__IMG_9__rep01" in p for p in verificar([*clips, extra], almacen))


def test_detecta_valores_no_finitos_y_huella_distinta(corpus) -> None:
    clips, almacen = corpus
    ruta = almacen.dir_ventanas / "ventanas.npy"
    v = np.load(ruta)
    v[0, 0, 0] = np.nan
    np.save(ruta, v)
    problemas = verificar(clips, almacen)
    assert any("finit" in p for p in problemas) and any("huella" in p for p in problemas)


def test_detecta_keypoints_con_un_ausente_a_medias_y_acepta_fuera_de_cero_uno(corpus) -> None:
    clips, almacen = corpus
    kp = almacen.leer_keypoints(clips[0])
    kp[0, 0] = 1.02  # fuera de [0, 1]: no es error
    kp[1, 2] = -1  # x a −1 con y real: error
    np.save(almacen.ruta_keypoints(clips[0]), kp)
    problemas = verificar(clips, almacen)
    assert len(problemas) == 1 and "a medias" in problemas[0]


def test_detecta_fotogramas_que_no_son_la_mitad(corpus) -> None:
    clips, almacen = corpus
    np.save(almacen.ruta_keypoints(clips[0]), almacen.leer_keypoints(clips[0])[:-1])
    assert any("⌈n/2⌉" in p for p in verificar(clips, almacen))


def test_detecta_versiones_mezcladas(corpus) -> None:
    clips, almacen = corpus
    ruta = almacen.ruta_keypoints(clips[1]).with_suffix(".meta.json")
    meta = json.loads(ruta.read_text())
    meta["macos"] = "27.3.0"
    ruta.write_text(json.dumps(meta))
    assert any("macos" in p for p in verificar(clips, almacen))


def test_sin_ventanas_lo_dice(tmp_path: Path) -> None:
    clips = leer_etiquetas(crear_corpus(tmp_path, CLIPS))
    assert any("ventanas.npy" in p for p in verificar(clips, Almacen(tmp_path / "vacio")))
