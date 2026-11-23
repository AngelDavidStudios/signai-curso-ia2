"""La entrada de la cadena: los clips de TT-11 descritos por `etiquetas.csv` (FR-001, FR-002).

La cadena procesa exactamente los clips de este fichero, ni más ni menos, y antes de procesar uno
comprueba que su contenido es el que se aprobó (md5).
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .errores import ClipAlterado, ErrorPreproceso


@dataclass(frozen=True)
class ClipEtiquetado:
    clip_id: str
    clase: str
    numero: int
    glosa: str
    grabacion: str
    repeticion: int
    n_fotogramas: int  # a 60 fps
    md5: str
    ruta: Path


def leer_etiquetas(raiz_clips: Path) -> tuple[ClipEtiquetado, ...]:
    """Los clips de `raiz_clips/etiquetas.csv`, ordenados por `clip_id`."""
    fichero = raiz_clips / "etiquetas.csv"
    if not fichero.is_file():
        raise ErrorPreproceso(
            f"No existe {fichero}. Trae los clips con `uv run dvc pull datos/clips.dvc`."
        )
    clips: dict[str, ClipEtiquetado] = {}
    with fichero.open(newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            clip_id = fila["clip_id"]
            if clip_id in clips:
                raise ErrorPreproceso(f"{clip_id}: clip_id duplicado en etiquetas.csv")
            n = int(fila["n_fotogramas"])
            if n < 1:
                raise ErrorPreproceso(f"{clip_id}: n_fotogramas = {n}; debe ser ≥ 1")
            clips[clip_id] = ClipEtiquetado(
                clip_id=clip_id, clase=fila["clase"], numero=int(fila["numero"]),
                glosa=fila["glosa"], grabacion=fila["grabacion"],
                repeticion=int(fila["repeticion"]), n_fotogramas=n, md5=fila["md5"],
                ruta=raiz_clips / fila["clase"] / f"{clip_id}.mp4",
            )
    return tuple(clips[k] for k in sorted(clips))


def md5_fichero(ruta: Path) -> str:
    resumen = hashlib.md5()
    with ruta.open("rb") as f:
        while bloque := f.read(1 << 20):
            resumen.update(bloque)
    return resumen.hexdigest()


def comprobar_md5(clip: ClipEtiquetado) -> None:
    obtenido = md5_fichero(clip.ruta)
    if obtenido != clip.md5:
        raise ClipAlterado(clip.clip_id, clip.md5, obtenido)
