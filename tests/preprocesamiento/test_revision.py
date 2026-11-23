"""Vídeos de revisión con el esqueleto dibujado (FR-020; research R-12)."""

import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np

from signai_ml.preproceso.revision import ALTO, ANCHO, COLORES, dibujar, generar_video

from .conftest import ClipSintetico


def fila(x: float = -1, y: float = -1) -> np.ndarray:
    f = np.full(96, -1, dtype=np.float32)
    return f


def test_dibuja_cada_bloque_con_su_color_y_no_dibuja_los_ausentes() -> None:
    lienzo = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    f = fila()
    f[0:2] = (0.5, 0.5)  # hombro izquierdo (pose)
    f[12:14] = (0.25, 0.75)  # muñeca de la mano izquierda
    f[54:56] = (0.75, 0.25)  # muñeca de la mano derecha

    salida = dibujar(lienzo, f)

    assert tuple(salida[640, 360]) == COLORES["pose"]
    assert tuple(salida[960, 180]) == COLORES["mano_izq"]
    assert tuple(salida[320, 540]) == COLORES["mano_der"]
    assert len({COLORES["pose"], COLORES["mano_izq"], COLORES["mano_der"]}) == 3
    assert (lienzo == 0).all()  # no modifica la entrada
    assert (dibujar(lienzo, fila())[100:] == 0).all()  # todo −1: nada dibujado (salvo la leyenda arriba)


def test_un_punto_fuera_de_la_imagen_no_rompe_el_dibujo() -> None:
    f = fila()
    f[0:2] = (1.02, -0.01)
    dibujar(np.zeros((ALTO, ANCHO, 3), dtype=np.uint8), f)


def test_genera_un_video_h264_720x1280_con_un_fotograma_por_fila(
    clip_sintetico: Callable[..., ClipSintetico], tmp_path: Path
) -> None:
    sintetico = clip_sintetico(n=8)
    keypoints = np.full((4, 96), -1, dtype=np.float32)
    destino = tmp_path / "rev.mp4"

    generar_video(sintetico.ruta, keypoints, destino)

    sonda = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_name,width,height,nb_read_frames,r_frame_rate", "-of", "csv=p=0", str(destino)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert sonda == "h264,720,1280,30/1,4"
