"""Fábricas sintéticas de la suite `preprocesamiento` (TT-12; constitución, Seguridad).

Nada de aquí lee clips reales: los vídeos se generan en la prueba y **no contienen personas**.

- `clip_sintetico`: un clip con el formato de los del corpus (1080×1920 vertical, 60/1 fps
  constantes, HEVC Main10 HLG/BT.2020), sin rotación salvo que se pida.
- `escribir_etiquetas`: un `etiquetas.csv` con las columnas de TT-11 para una lista de clips.
"""

import csv
import hashlib
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

COLUMNAS_ETIQUETAS = (
    "clip_id", "clase", "numero", "glosa", "grabacion", "repeticion", "frame_inicio", "frame_fin",
    "n_fotogramas", "duracion_s", "md5", "fecha_aprobacion", "codificacion",
)


@dataclass(frozen=True)
class ClipSintetico:
    ruta: Path
    n: int


@pytest.fixture(scope="session")
def ffmpeg_disponible() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.fail("La suite preprocesamiento necesita ffmpeg y ffprobe (scripts/bootstrap.sh)")


def crear_clip(destino: Path, n: int, rotacion: int | None = None) -> ClipSintetico:
    destino.parent.mkdir(parents=True, exist_ok=True)
    intermedio = destino.with_name(destino.stem + ".sin-rotar.mp4")
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=60",
            "-frames:v", str(n), "-fps_mode", "passthrough",
            "-c:v", "libx265", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p10le",
            "-x265-params", "log-level=error:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
            "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc",
            "-tag:v", "hvc1", str(intermedio),
        ],
        check=True, stdin=subprocess.DEVNULL,
    )
    if rotacion is None:
        intermedio.replace(destino)
    else:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-display_rotation", str(rotacion),
             "-i", str(intermedio), "-c", "copy", str(destino)],
            check=True, stdin=subprocess.DEVNULL,
        )
        intermedio.unlink()
    return ClipSintetico(destino, n)


@pytest.fixture
def clip_sintetico(tmp_path: Path, ffmpeg_disponible: None) -> Callable[..., ClipSintetico]:
    """`clip_sintetico(n=24, rotacion=None, ruta=None)` → ClipSintetico."""

    def crear(n: int = 24, rotacion: int | None = None, ruta: Path | None = None) -> ClipSintetico:
        destino = ruta or tmp_path / "clips" / "0-Ref" / f"0-Ref__SINT__rep{n:02d}.mp4"
        return crear_clip(destino, n, rotacion)

    return crear


def md5(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def fila_etiqueta(clip_id: str, n: int, huella: str, **extra: object) -> dict[str, object]:
    clase, grabacion, rep = clip_id.split("__")
    numero, glosa = clase.split("-", 1)
    fila: dict[str, object] = {
        "clip_id": clip_id, "clase": clase, "numero": int(numero), "glosa": glosa.lower(),
        "grabacion": grabacion, "repeticion": int(rep.removeprefix("rep")), "frame_inicio": 0,
        "frame_fin": n - 1, "n_fotogramas": n, "duracion_s": f"{n / 60:.3f}", "md5": huella,
        "fecha_aprobacion": "2026-10-09", "codificacion": "hevc-main10-crf14-v1",
    }
    fila.update(extra)
    return fila


@pytest.fixture
def escribir_etiquetas() -> Callable[[Path, Sequence[dict[str, object]]], Path]:
    """`escribir_etiquetas(raiz_clips, filas)` escribe `raiz_clips/etiquetas.csv`."""

    def escribir(raiz: Path, filas: Sequence[dict[str, object]]) -> Path:
        raiz.mkdir(parents=True, exist_ok=True)
        ruta = raiz / "etiquetas.csv"
        with ruta.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNAS_ETIQUETAS)
            w.writeheader()
            w.writerows(filas)
        return ruta

    return escribir
