"""Dobles de la suite `preprocesamiento`: un corpus falso y un extractor sin Vision.

El corpus falso son ficheros con bytes aleatorios (no vídeos): basta para la huella y para el
recorrido de la cadena. El extractor doble escribe una salida X-4 válida para el clip, con un cuerpo
y dos manos que se mueven poco a poco, o lo que se le indique por clip.
"""

import csv
import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from signai_ml.preproceso import VERSION_EXTRACTOR
from signai_ml.preproceso.deteccion import ORDEN_MANO, ORDEN_POSE

from .conftest import COLUMNAS_ETIQUETAS, fila_etiqueta

MACOS = "27.2.0"


def crear_corpus(raiz: Path, clips: Sequence[tuple[str, int]]) -> Path:
    """`clips` = [(clip_id, n_fotogramas)] → `raiz/datos/clips` con sus ficheros y `etiquetas.csv`."""
    base = raiz / "datos" / "clips"
    filas = []
    for i, (clip_id, n) in enumerate(clips):
        clase = clip_id.split("__")[0]
        ruta = base / clase / f"{clip_id}.mp4"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(f"{clip_id}-{i}".encode() * 50)
        filas.append(fila_etiqueta(clip_id, n, hashlib.md5(ruta.read_bytes()).hexdigest()))
    with (base / "etiquetas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_ETIQUETAS)
        w.writeheader()
        w.writerows(filas)
    return base


def _fotograma(origen: int, sin_manos: bool = False, sin_cuerpo: bool = False, fallido: bool = False) -> dict:
    if fallido:
        return {"origen": origen, "fallido": True, "cuerpos": [], "manos": []}
    d = 0.001 * origen
    pose = {
        "leftShoulder": [0.6 + d, 0.6], "rightShoulder": [0.4 + d, 0.6],
        "leftElbow": [0.7, 0.5], "rightElbow": [0.3, 0.5],
        "leftWrist": [0.7, 0.4 + d], "rightWrist": [0.3, 0.4 + d],
    }
    manos = [] if sin_manos else [
        {"lado": "left", "puntos": {n: [0.7 + 0.001 * k, 0.4 + d + 0.002 * k] for k, n in enumerate(ORDEN_MANO)}},
        {"lado": "right", "puntos": {n: [0.3 + 0.001 * k, 0.4 + d + 0.002 * k] for k, n in enumerate(ORDEN_MANO)}},
    ]
    assert set(pose) == set(ORDEN_POSE)
    return {"origen": origen, "fallido": False, "cuerpos": [] if sin_cuerpo else [{"puntos": pose}], "manos": manos}


class ExtractorDoble:
    """Mismo interfaz que `ExtractorKeypoints`. `perfil[clip_id]` cambia lo que «ve»."""

    def __init__(self, n_por_clip: dict[str, int], perfil: dict[str, str] | None = None) -> None:
        self.n_por_clip = n_por_clip
        self.perfil = perfil or {}
        self.llamadas: list[str] = []

    def extraer(self, video: Path, destino: Path, cada: int, volcar: Path | None = None) -> None:
        clip_id = Path(video).stem
        self.llamadas.append(clip_id)
        n = self.n_por_clip[clip_id]
        modo = self.perfil.get(clip_id, "")
        fotogramas = [
            _fotograma(
                o, sin_manos=modo == "sin_manos", sin_cuerpo=modo == "sin_cuerpo",
                fallido=modo == "fallidos" and o % 4 == 0,
            )
            for o in range(0, n, cada)
        ]
        datos = {
            "version_extractor": VERSION_EXTRACTOR,
            "detector": {"pose_revision": 2, "manos_revision": 1, "max_manos": 2},
            "macos": MACOS, "md5": hashlib.md5(Path(video).read_bytes()).hexdigest(),
            "fotogramas_decodificados": n, "cada": cada,
            "imagen": {"ancho": 720, "alto": 1280, "color": "bt709-sdr-8bit"},
            "fotogramas": fotogramas,
        }
        destino.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destino, "wt", encoding="utf-8") as f:
            json.dump(datos, f)
