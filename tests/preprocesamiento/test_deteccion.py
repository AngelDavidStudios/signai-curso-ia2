"""Lectura de la salida del extractor y su envoltorio (contracts/extractor.md X-4, X-5)."""

import gzip
import json
from pathlib import Path

import pytest

from signai_ml.preproceso import VERSION_EXTRACTOR
from signai_ml.preproceso.clips import ClipEtiquetado
from signai_ml.preproceso.deteccion import (
    ExtractorKeypoints,
    ManoDetectada,
    PuntoDetector,
    leer_deteccion,
)
from signai_ml.preproceso.errores import ErrorExtraccion, FaltaHerramienta

MD5 = "a" * 32


def clip(n: int = 5) -> ClipEtiquetado:
    return ClipEtiquetado("1-Yo__IMG_1008__rep01", "1-Yo", 1, "yo", "IMG_1008", 1, n, MD5, Path("x.mp4"))


def salida(n: int = 5, **cambios) -> dict:
    datos = {
        "version_extractor": VERSION_EXTRACTOR,
        "detector": {"pose_revision": 2, "manos_revision": 1, "max_manos": 2},
        "macos": "27.2",
        "md5": MD5,
        "fotogramas_decodificados": n,
        "cada": 2,
        "imagen": {"ancho": 720, "alto": 1280, "color": "bt709-sdr-8bit"},
        "fotogramas": [
            {
                "origen": i, "fallido": i == 2,
                "cuerpos": [{"puntos": {"leftShoulder": [0.6, 0.6], "rightShoulder": [0.4, 0.6]}}],
                "manos": [{"lado": "left", "puntos": {"wrist": [0.7, 0.3]}}, {"lado": None, "puntos": {}}],
            }
            for i in range(0, n, 2)
        ],
    }
    datos.update(cambios)
    return datos


def escribir(tmp_path: Path, datos: dict) -> Path:
    ruta = tmp_path / "det.json.gz"
    with gzip.open(ruta, "wt", encoding="utf-8") as f:
        json.dump(datos, f)
    return ruta


def test_lee_la_salida_completa(tmp_path: Path) -> None:
    det = leer_deteccion(escribir(tmp_path, salida()), clip())
    assert det.version_extractor == VERSION_EXTRACTOR
    assert det.macos == "27.2"
    assert det.md5 == MD5
    assert det.fotogramas_decodificados == 5
    assert [f.origen for f in det.fotogramas] == [0, 2, 4]
    f0 = det.fotogramas[0]
    assert not f0.fallido
    assert f0.cuerpos[0].puntos["leftShoulder"] == PuntoDetector(0.6, 0.6)
    assert f0.manos[0] == ManoDetectada("left", {"wrist": PuntoDetector(0.7, 0.3)})
    assert f0.manos[1].lado is None
    assert det.fotogramas[1].fallido


@pytest.mark.parametrize(
    ("cambios", "motivo"),
    [
        ({"version_extractor": "signai-ml/keypoints/0"}, "versión"),
        ({"detector": {"pose_revision": 1, "manos_revision": 1, "max_manos": 2}}, "revisiones"),
        ({"md5": "b" * 32}, "md5"),
        ({"fotogramas_decodificados": 6}, "fotogramas"),
        ({"cada": 3}, "cada"),
    ],
)
def test_rechaza_una_salida_que_no_es_de_este_clip_o_de_esta_version(
    tmp_path: Path, cambios: dict, motivo: str
) -> None:
    with pytest.raises(ErrorExtraccion, match=motivo):
        leer_deteccion(escribir(tmp_path, salida(**cambios)), clip())


def test_rechaza_origenes_que_no_son_la_decimacion(tmp_path: Path) -> None:
    datos = salida()
    datos["fotogramas"][1]["origen"] = 3
    with pytest.raises(ErrorExtraccion, match="decimación"):
        leer_deteccion(escribir(tmp_path, datos), clip())


def test_rechaza_un_nombre_de_articulacion_desconocido(tmp_path: Path) -> None:
    datos = salida()
    datos["fotogramas"][0]["cuerpos"][0]["puntos"]["nose"] = [0.5, 0.5]
    with pytest.raises(ErrorExtraccion, match="nose"):
        leer_deteccion(escribir(tmp_path, datos), clip())


def test_el_extractor_sin_binario_avisa_de_como_compilarlo(tmp_path: Path) -> None:
    with pytest.raises(FaltaHerramienta, match="swift build"):
        ExtractorKeypoints(tmp_path / "no-existe").extraer(Path("x.mp4"), tmp_path / "s.json.gz", cada=2)


def test_un_fallo_del_extractor_es_error_de_extraccion_con_su_mensaje(tmp_path: Path) -> None:
    falso = tmp_path / "falso"
    falso.write_text("#!/bin/sh\necho 'sin pista de vídeo' >&2\nexit 1\n")
    falso.chmod(0o755)
    with pytest.raises(ErrorExtraccion, match="sin pista de vídeo"):
        ExtractorKeypoints(falso).extraer(tmp_path / "x.mp4", tmp_path / "s.json.gz", cada=2)
