"""Contrato del extractor real `keypoints-vision` (Swift + Apple Vision) sobre un clip sintético.

El extractor se compila aquí, una vez por sesión, y se ejecuta sobre un vídeo **sin personas** con el
formato de los clips del corpus: Vision no detecta a nadie, y lo que se comprueba es la salida X-4,
la decimación y que, tras la selección, los 48 puntos valen −1. Nunca se prueba con vídeo de
señantes (constitución, Seguridad; FR-010).
"""

import gzip
import json
import platform
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from signai_ml.preproceso import VERSION_EXTRACTOR
from signai_ml.preproceso.clips import ClipEtiquetado, md5_fichero
from signai_ml.preproceso.deteccion import ExtractorKeypoints, leer_deteccion
from signai_ml.preproceso.errores import ErrorExtraccion
from signai_ml.preproceso.seleccion import seleccionar_clip

from .conftest import ClipSintetico

PAQUETE = Path(__file__).parents[2] / "herramientas" / "keypoints-vision"


@pytest.fixture(scope="session")
def extractor(tmp_path_factory: pytest.TempPathFactory) -> ExtractorKeypoints:
    if platform.system() != "Darwin" or shutil.which("swift") is None:
        pytest.fail("El extractor necesita macOS y Swift (Vision solo existe en macOS)")
    construccion = tmp_path_factory.mktemp("swift-build-keypoints")
    subprocess.run(
        ["swift", "build", "-c", "release", "--package-path", str(PAQUETE), "--build-path", str(construccion)],
        check=True, capture_output=True,
    )
    return ExtractorKeypoints(construccion / "release" / "keypoints-vision")


def _clip(sintetico: ClipSintetico) -> ClipEtiquetado:
    return ClipEtiquetado(
        "0-Ref__SINT__rep01", "0-Ref", 0, "ref", "SINT", 1, sintetico.n, md5_fichero(sintetico.ruta),
        sintetico.ruta,
    )


def test_salida_completa_sobre_un_clip_sin_personas(
    extractor: ExtractorKeypoints, clip_sintetico: Callable[..., ClipSintetico], tmp_path: Path
) -> None:
    sintetico = clip_sintetico(n=24)
    destino = tmp_path / "det.json.gz"

    extractor.extraer(sintetico.ruta, destino, cada=2)

    with gzip.open(destino, "rt", encoding="utf-8") as f:
        crudo = json.load(f)
    assert crudo["version_extractor"] == VERSION_EXTRACTOR
    assert crudo["detector"] == {"pose_revision": 2, "manos_revision": 1, "max_manos": 2}
    assert crudo["imagen"] == {"ancho": 720, "alto": 1280, "color": "bt709-sdr-8bit"}
    assert crudo["macos"].split(".")[0].isdigit()
    assert crudo["fotogramas_decodificados"] == 24

    det = leer_deteccion(destino, _clip(sintetico))
    assert [f.origen for f in det.fotogramas] == list(range(0, 24, 2))
    assert all(f.cuerpos == () and f.manos == () for f in det.fotogramas)

    keypoints = seleccionar_clip(det)
    assert keypoints.shape == (12, 96)
    assert (keypoints == -1).all()


def test_volcar_escribe_la_imagen_que_recibe_vision(
    extractor: ExtractorKeypoints, clip_sintetico: Callable[..., ClipSintetico], tmp_path: Path
) -> None:
    sintetico = clip_sintetico(n=6)
    carpeta = tmp_path / "png"

    extractor.extraer(sintetico.ruta, tmp_path / "det.json.gz", cada=2, volcar=carpeta)

    pngs = sorted(carpeta.glob("*.png"))
    assert [p.name for p in pngs] == ["000000.png", "000002.png", "000004.png"]
    sonda = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(pngs[0])],
        check=True, capture_output=True, text=True,
    )
    assert sonda.stdout.strip() == "720,1280"


def test_rechaza_un_video_con_rotacion_en_los_metadatos(
    extractor: ExtractorKeypoints, clip_sintetico: Callable[..., ClipSintetico], tmp_path: Path
) -> None:
    sintetico = clip_sintetico(n=4, rotacion=-90)
    with pytest.raises(ErrorExtraccion, match="rotación"):
        extractor.extraer(sintetico.ruta, tmp_path / "det.json.gz", cada=2)
