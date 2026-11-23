"""Dónde y cómo se guardan los resultados (research R-09, R-10).

Un `Almacen` cuelga de una base: `datos/` para el corpus completo y `datos/validacion/` para el lote.

    <base>/keypoints/<clase>/<clip_id>.npy        (T, 96) float32, −1 en ausencias (FR-009)
    <base>/keypoints/<clase>/<clip_id>.meta.json   vigencia: md5 del clip y versiones (R-10)
    <base>/ventanas/ventanas.npy                   (N, 75, 96) float32
    <base>/ventanas/indice.csv                     una fila por ventana, en el mismo orden
    <base>/ventanas/manifiesto.json                versiones, aprobación del lote y huella

Todo se escribe en un temporal y se renombra: un corte deja el fichero sin escribir, nunca a medias.
"""

import csv
import hashlib
import io
import json
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import ESQUEMA, NORMALIZACION, VERSION_CADENA, VERSION_EXTRACTOR
from .clips import ClipEtiquetado

COLUMNAS_INDICE = (
    "fila", "clip_id", "clase", "numero", "glosa", "grabacion", "repeticion",
    "n_fotogramas_60", "n_fotogramas_30", "md5_clip",
)


@dataclass(frozen=True)
class Versiones:
    version_cadena: str
    version_extractor: str
    esquema: str
    normalizacion: str
    macos: str

    @classmethod
    def actuales(cls) -> "Versiones":
        return cls(VERSION_CADENA, VERSION_EXTRACTOR, ESQUEMA, NORMALIZACION, macos_actual())


def macos_actual() -> str:
    partes = (platform.mac_ver()[0] or "0").split(".")
    return ".".join((partes + ["0", "0"])[:3])


def escribir_atomico(ruta: Path, escribir: Callable[[Path], None]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(f".{ruta.name}.tmp-{os.getpid()}")
    try:
        escribir(temporal)
        os.replace(temporal, ruta)
    finally:
        temporal.unlink(missing_ok=True)


def escribir_texto(ruta: Path, texto: str) -> None:
    escribir_atomico(ruta, lambda t: t.write_text(texto, encoding="utf-8"))


def sha256_fichero(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as f:
        while bloque := f.read(1 << 20):
            resumen.update(bloque)
    return resumen.hexdigest()


def _guardar_npy(ruta: Path, matriz: np.ndarray) -> None:
    def escribir(temporal: Path) -> None:
        with temporal.open("wb") as f:
            np.save(f, matriz, allow_pickle=False)

    escribir_atomico(ruta, escribir)


class Almacen:
    def __init__(self, base: Path) -> None:
        self.base = Path(base)
        self.dir_keypoints = self.base / "keypoints"
        self.dir_ventanas = self.base / "ventanas"

    # ── Keypoints por clip ──

    def ruta_keypoints(self, clip: ClipEtiquetado) -> Path:
        return self.dir_keypoints / clip.clase / f"{clip.clip_id}.npy"

    def _ruta_meta(self, clip: ClipEtiquetado) -> Path:
        return self.ruta_keypoints(clip).with_suffix(".meta.json")

    def guardar_keypoints(
        self, clip: ClipEtiquetado, keypoints: np.ndarray, versiones: Versiones, fallidos: int = 0
    ) -> None:
        meta = {
            "clip_id": clip.clip_id, "md5_clip": clip.md5, **asdict(versiones),
            "n_fotogramas_60": clip.n_fotogramas, "n_fotogramas_30": int(keypoints.shape[0]),
            "fallidos": fallidos,
        }
        _guardar_npy(self.ruta_keypoints(clip), keypoints.astype(np.float32))
        escribir_texto(self._ruta_meta(clip), json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))

    def leer_keypoints(self, clip: ClipEtiquetado) -> np.ndarray:
        return np.load(self.ruta_keypoints(clip), allow_pickle=False)

    def leer_meta(self, clip: ClipEtiquetado) -> dict:
        return json.loads(self._ruta_meta(clip).read_text(encoding="utf-8"))

    def vigente(self, clip: ClipEtiquetado, versiones: Versiones) -> bool:
        if not self.ruta_keypoints(clip).is_file() or not self._ruta_meta(clip).is_file():
            return False
        meta = self.leer_meta(clip)
        esperado = {"md5_clip": clip.md5, **asdict(versiones)}
        return all(meta.get(k) == v for k, v in esperado.items())

    # ── Ventanas ensambladas ──

    def escribir_ventanas(
        self, clips: Sequence[ClipEtiquetado], ventanas: np.ndarray, versiones: Versiones, extra: Mapping
    ) -> Path:
        ruta = self.dir_ventanas / "ventanas.npy"
        _guardar_npy(ruta, ventanas.astype(np.float32))
        salida = io.StringIO()
        w = csv.DictWriter(salida, fieldnames=COLUMNAS_INDICE, lineterminator="\n")
        w.writeheader()
        for fila, clip in enumerate(clips):
            w.writerow({
                "fila": fila, "clip_id": clip.clip_id, "clase": clip.clase, "numero": clip.numero,
                "glosa": clip.glosa, "grabacion": clip.grabacion, "repeticion": clip.repeticion,
                "n_fotogramas_60": clip.n_fotogramas, "n_fotogramas_30": -(-clip.n_fotogramas // 2),
                "md5_clip": clip.md5,
            })
        escribir_texto(self.dir_ventanas / "indice.csv", salida.getvalue())
        manifiesto = {
            **asdict(versiones), **dict(extra),
            "ventanas": int(ventanas.shape[0]), "forma": list(ventanas.shape), "dtype": "float32",
            "sha256_ventanas": sha256_fichero(ruta),
            "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        escribir_texto(
            self.dir_ventanas / "manifiesto.json", json.dumps(manifiesto, ensure_ascii=False, indent=2, sort_keys=True)
        )
        return ruta
