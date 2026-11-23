"""Lectura y validación del corpus procesado de TT-12 (data-model «Entrada»; spec FR-011).

Lee `datos/ventanas/indice.csv` y `datos/ventanas/manifiesto.json` sin modificarlos. La generación se
detiene, sin escribir nada, si el índice no permite una partición de las 100 clases.
"""

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import CLASES_ESPERADAS
from .errores import IndiceNoValido
from .huella import sha256_fichero

RESTAURAR_VENTANAS = "uv run dvc checkout datos/ventanas.dvc"


@dataclass(frozen=True)
class CorpusProcesado:
    filas: tuple[dict[str, str], ...]
    sha256_indice: str
    sha256_ventanas: str
    n: int


def dir_ventanas(raiz: Path) -> Path:
    return Path(raiz) / "datos" / "ventanas"


def _exigir(ruta: Path) -> Path:
    if not ruta.is_file():
        raise IndiceNoValido(
            f"Falta {ruta.relative_to(ruta.parents[2])}: restaura el corpus procesado con `{RESTAURAR_VENTANAS}`."
        )
    return ruta


def _leer_filas(ruta: Path) -> tuple[dict[str, str], ...]:
    with ruta.open(encoding="utf-8", newline="") as f:
        return tuple(csv.DictReader(f))


def _comprobar_numeracion(filas: tuple[dict[str, str], ...]) -> None:
    for esperada, fila in enumerate(filas):
        if fila.get("fila") != str(esperada):
            raise IndiceNoValido(
                f"Las filas del índice no son 0…N−1: en la posición {esperada} la fila dice {fila.get('fila')!r}."
            )


def _comprobar_duplicados(filas: tuple[dict[str, str], ...]) -> None:
    repetidos = sorted(clip_id for clip_id, veces in Counter(f["clip_id"] for f in filas).items() if veces > 1)
    if repetidos:
        raise IndiceNoValido(f"Clips repetidos en el índice: {', '.join(repetidos)}. Un clip no puede estar dos veces.")


def _comprobar_grabaciones(filas: tuple[dict[str, str], ...]) -> None:
    """La grabación es la unidad de partición (TT-22, research R-01): todo clip tiene la suya."""
    if filas and "grabacion" not in filas[0]:
        raise IndiceNoValido(
            "El índice no tiene la columna grabacion, que la partición necesita para apartar grabaciones "
            f"enteras. Restaura el corpus procesado con `{RESTAURAR_VENTANAS}`."
        )
    sin_grabacion = [f["clip_id"] for f in filas if not (f.get("grabacion") or "").strip()]
    if sin_grabacion:
        raise IndiceNoValido(f"Clips sin grabación en el índice: {', '.join(sin_grabacion[:5])}.")


def _comprobar_clases(filas: tuple[dict[str, str], ...], esperadas: int) -> None:
    clases = {f["clase"] for f in filas}
    if len(clases) != esperadas:
        raise IndiceNoValido(
            f"El índice tiene {len(clases)} clases y deben ser {esperadas}: no se genera una partición incompleta."
        )


def leer_corpus(raiz: Path, clases_esperadas: int = CLASES_ESPERADAS) -> CorpusProcesado:
    ruta_indice = _exigir(dir_ventanas(raiz) / "indice.csv")
    ruta_manifiesto = _exigir(dir_ventanas(raiz) / "manifiesto.json")
    manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
    filas = _leer_filas(ruta_indice)

    if len(filas) != manifiesto["ventanas"]:
        raise IndiceNoValido(
            f"El índice tiene {len(filas)} filas y el manifiesto de las ventanas dice {manifiesto['ventanas']}. "
            f"Restaura el corpus procesado con `{RESTAURAR_VENTANAS}`."
        )
    _comprobar_numeracion(filas)
    _comprobar_duplicados(filas)
    _comprobar_grabaciones(filas)
    _comprobar_clases(filas, clases_esperadas)
    return CorpusProcesado(
        filas=filas,
        sha256_indice=sha256_fichero(ruta_indice),
        sha256_ventanas=manifiesto["sha256_ventanas"],
        n=len(filas),
    )
