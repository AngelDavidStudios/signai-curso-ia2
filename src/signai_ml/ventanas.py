"""Las ventanas del corpus procesado (TT-12), tal como las leen el entrenamiento y la aceptación.

Una sola lectura para TT-14 y TT-15, para que las dos calculen las mismas etiquetas y las mismas
huellas (research R-02 de TT-15). Aquí no se transforma ninguna coordenada: las ventanas ya están
normalizadas. Qué clips se leen lo decide el contrato de lectura de la partición, no este módulo.
"""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESTAURAR_VENTANAS = "uv run dvc checkout datos/ventanas.dvc"


class VentanasNoValidas(ValueError):
    """El corpus procesado o el vocabulario no son coherentes; cada paquete lo traduce a su error."""


def _leer_csv(ruta: Path) -> list[dict[str, str]]:
    with open(ruta, newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def _dir_ventanas(raiz: Path) -> Path:
    return Path(raiz) / "datos" / "ventanas"


def vocabulario(raiz: Path) -> list[dict[str, str]]:
    """`config/corpus/vocabulario.csv` ordenado por número de clase; los números deben ser 1…N."""
    filas = sorted(_leer_csv(Path(raiz) / "config" / "corpus" / "vocabulario.csv"), key=lambda f: int(f["numero"]))
    if [int(f["numero"]) for f in filas] != list(range(1, len(filas) + 1)):
        raise VentanasNoValidas("config/corpus/vocabulario.csv: los números de clase no son 1…N.")
    return filas


def huellas_corpus(raiz: Path) -> dict[str, str]:
    """Las huellas que las corridas registran: la de `ventanas.npy` (del manifiesto) y la del índice."""
    ventanas = _dir_ventanas(raiz)
    manifiesto = json.loads((ventanas / "manifiesto.json").read_text(encoding="utf-8"))
    return {
        "corpus.sha256_ventanas": manifiesto["sha256_ventanas"],
        "corpus.sha256_indice": hashlib.sha256((ventanas / "indice.csv").read_bytes()).hexdigest(),
    }


@dataclass(frozen=True)
class Corpus:
    raiz: Path
    clases: tuple[str, ...]
    glosas: tuple[str, ...]
    etiquetas: np.ndarray
    huellas: dict[str, str]

    def tomar(self, filas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`(x float32 (len(filas), 75, 96), y int64)` de las filas pedidas, sin leer el resto."""
        ventanas = np.load(_dir_ventanas(self.raiz) / "ventanas.npy", mmap_mode="r")
        return np.array(ventanas[filas], dtype=np.float32), self.etiquetas[filas]


def abrir_corpus(raiz: Path) -> Corpus:
    """Comprueba que el índice corresponde al vocabulario y deja el corpus listo para leer filas."""
    raiz = Path(raiz)
    indice = _leer_csv(_dir_ventanas(raiz) / "indice.csv")
    filas_vocabulario = vocabulario(raiz)
    clase_de = {int(f["numero"]): f["clase"] for f in filas_vocabulario}
    for fila in indice:
        if clase_de.get(int(fila["numero"])) != fila["clase"]:
            raise VentanasNoValidas(
                f"El vocabulario no corresponde al índice: la clase {fila['numero']} es «{fila['clase']}» en "
                f"datos/ventanas/indice.csv. Restaura los datos con `{RESTAURAR_VENTANAS}`."
            )
    etiquetas = np.array([int(f["numero"]) - 1 for f in indice], dtype=np.int64)
    etiquetas.setflags(write=False)
    return Corpus(
        raiz=raiz,
        clases=tuple(f["clase"] for f in filas_vocabulario),
        glosas=tuple(f["glosa"] for f in filas_vocabulario),
        etiquetas=etiquetas,
        huellas=huellas_corpus(raiz),
    )
