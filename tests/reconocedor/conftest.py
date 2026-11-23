"""Fixtures de la suite `reconocedor` (TT-14; protocolo por rotación de TT-22).

Corpus pequeño e inventado en `tmp_path` (4 clases × 8 clips), lectores falsos de la partición
versión 2 (dos pliegues y su unión, sin prueba) y una forma de modelo diminuta para entrenar en
segundos sobre CPU. Nada de aquí lee `datos/` real ni el MLflow real (`tests/conftest.py` fija
`SIGNAI_DATA_ROOT` a un temporal).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from signai_ml.reconocedor.configuracion import Forma

from .sinteticos import escribir_corpus

N_CLASES = 4
POR_CLASE = 8
HUELLA_PARTICION = "f" * 64


@dataclass(frozen=True)
class RotacionFalsa:
    entrenamiento: np.ndarray
    validacion: np.ndarray


@dataclass(frozen=True)
class RotacionesFalsas:
    a: RotacionFalsa
    b: RotacionFalsa
    huella: str


@dataclass(frozen=True)
class FinalFalso:
    entrenamiento: np.ndarray
    huella: str


def pliegues_sinteticos() -> tuple[np.ndarray, np.ndarray]:
    """Pliegue A: la primera mitad de los clips de cada clase; pliegue B: la segunda."""
    a, b = [], []
    for clase in range(N_CLASES):
        filas = list(range(clase * POR_CLASE, (clase + 1) * POR_CLASE))
        a += filas[: POR_CLASE // 2]
        b += filas[POR_CLASE // 2 :]
    return np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)


def lector_seleccion_sintetico(_raiz: Path) -> RotacionesFalsas:
    a, b = pliegues_sinteticos()
    return RotacionesFalsas(RotacionFalsa(a, b), RotacionFalsa(b, a), HUELLA_PARTICION)


def lector_final_sintetico(_raiz: Path) -> FinalFalso:
    a, b = pliegues_sinteticos()
    return FinalFalso(np.sort(np.concatenate([a, b])), HUELLA_PARTICION)


@pytest.fixture
def raiz_sintetica(tmp_path: Path) -> Path:
    return escribir_corpus(tmp_path / "repo", N_CLASES, POR_CLASE)


@pytest.fixture
def lector_seleccion():
    return lector_seleccion_sintetico


@pytest.fixture
def lector_final():
    return lector_final_sintetico


@pytest.fixture
def forma_diminuta() -> Forma:
    return Forma(capas=1, dimension=98, cabezas=2, intermedia=16)
