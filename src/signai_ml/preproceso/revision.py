"""Vídeos de revisión con el esqueleto dibujado (FR-020; research R-12).

ffmpeg decodifica el clip a 720×1280 conservando los fotogramas pares (los mismos que vio Vision);
NumPy dibuja los puntos presentes, cada bloque en su color; ffmpeg codifica en H.264 a 30 fps. Los
vídeos son material desechable: no se versionan y no son corpus.

Colores (la leyenda, arriba a la izquierda, en el mismo orden): pose amarillo, mano izquierda **de
la persona** cian, mano derecha **de la persona** magenta.
"""

import subprocess
from pathlib import Path

import numpy as np

from .errores import ErrorPreproceso

ANCHO, ALTO = 720, 1280
RADIO = 4
COLORES = {"pose": (255, 215, 0), "mano_izq": (0, 200, 255), "mano_der": (255, 0, 200)}
BLOQUES = (("pose", range(0, 6)), ("mano_izq", range(6, 27)), ("mano_der", range(27, 48)))


def dibujar(fotograma: np.ndarray, fila: np.ndarray) -> np.ndarray:
    """Copia de `fotograma` (ALTO, ANCHO, 3) con los puntos presentes de `fila` (96) y la leyenda."""
    salida = fotograma.copy()
    for k, (nombre, _) in enumerate(BLOQUES):
        salida[10:30, 10 + 30 * k:30 + 30 * k] = COLORES[nombre]
    for nombre, puntos in BLOQUES:
        for p in puntos:
            x, y = float(fila[2 * p]), float(fila[2 * p + 1])
            if x == -1 and y == -1:
                continue
            col, fil = int(round(x * ANCHO)), int(round(y * ALTO))
            if not (0 <= col < ANCHO and 0 <= fil < ALTO):
                continue
            salida[max(0, fil - RADIO):fil + RADIO + 1, max(0, col - RADIO):col + RADIO + 1] = COLORES[nombre]
    return salida


def generar_video(clip: Path, keypoints: np.ndarray, destino: Path, cada: int = 2, ffmpeg: str = "ffmpeg") -> None:
    lectura = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-i", str(clip),
         "-vf", f"select=not(mod(n\\,{cada})),scale={ANCHO}:{ALTO}", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=False,
    )
    if lectura.returncode != 0:
        raise ErrorPreproceso(f"{clip.name}: ffmpeg no pudo decodificar el clip: {lectura.stderr.decode()[-300:]}")
    tamano = ANCHO * ALTO * 3
    n = len(lectura.stdout) // tamano
    if n != keypoints.shape[0]:
        raise ErrorPreproceso(f"{clip.name}: {n} fotogramas decodificados y {keypoints.shape[0]} filas de keypoints")
    fotogramas = np.frombuffer(lectura.stdout, dtype=np.uint8).reshape(n, ALTO, ANCHO, 3)
    dibujados = b"".join(dibujar(f, fila).tobytes() for f, fila in zip(fotogramas, keypoints, strict=True))
    destino.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{ANCHO}x{ALTO}", "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "23", str(destino)],
        input=dibujados, check=True,
    )
