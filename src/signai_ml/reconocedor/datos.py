"""Datos de la selección por rotación y del modelo final (TT-22; research R-02, R-05, R-06).

La única vía hacia los clips es el contrato de lectura de la partición versión 2: `para_seleccion`
entrega las dos rotaciones (entrena un pliegue y valida el otro) y `para_entrenamiento_final` los dos
pliegues juntos, sin validación. Ninguno da acceso a la prueba, y los dos comprueban antes la ligadura
con el corpus. Las ventanas son las de TT-12, ya normalizadas: aquí no se transforma ninguna coordenada.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..particion.lectura import para_entrenamiento_final, para_seleccion
from ..ventanas import Corpus, VentanasNoValidas, abrir_corpus
from .errores import ErrorReconocedor

ROTACIONES = ("a", "b")


@dataclass(frozen=True)
class DatosEntrenamiento:
    """Una rotación de la selección: entrena con un pliegue y valida con el otro."""

    x_entrenamiento: torch.Tensor
    y_entrenamiento: torch.Tensor
    x_validacion: torch.Tensor
    y_validacion: torch.Tensor
    clases: tuple[str, ...]
    glosas: tuple[str, ...]
    huellas: dict[str, str]


@dataclass(frozen=True)
class DatosFinal:
    """El modelo final: los dos pliegues juntos, sin validación (FR-004a)."""

    x_entrenamiento: torch.Tensor
    y_entrenamiento: torch.Tensor
    clases: tuple[str, ...]
    glosas: tuple[str, ...]
    huellas: dict[str, str]


def _corpus(raiz: Path) -> Corpus:
    try:
        return abrir_corpus(raiz)
    except VentanasNoValidas as error:
        raise ErrorReconocedor(str(error)) from error


def _tomar(corpus: Corpus, filas: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = corpus.tomar(filas)
    return torch.from_numpy(x), torch.from_numpy(np.array(y))


def cargar_rotacion(raiz: Path, rotacion: str, lector: Callable = para_seleccion) -> DatosEntrenamiento:
    if rotacion not in ROTACIONES:
        raise ErrorReconocedor(f"La rotación {rotacion!r} no existe: son {', '.join(ROTACIONES)}.")
    raiz = Path(raiz)
    rotaciones = lector(raiz)
    lados = getattr(rotaciones, rotacion)
    corpus = _corpus(raiz)
    x_entrenamiento, y_entrenamiento = _tomar(corpus, lados.entrenamiento)
    x_validacion, y_validacion = _tomar(corpus, lados.validacion)
    return DatosEntrenamiento(
        x_entrenamiento, y_entrenamiento, x_validacion, y_validacion,
        clases=corpus.clases, glosas=corpus.glosas,
        huellas={**corpus.huellas, "particion.huella": rotaciones.huella},
    )


def cargar_final(raiz: Path, lector: Callable = para_entrenamiento_final) -> DatosFinal:
    raiz = Path(raiz)
    lado = lector(raiz)
    corpus = _corpus(raiz)
    x_entrenamiento, y_entrenamiento = _tomar(corpus, lado.entrenamiento)
    return DatosFinal(
        x_entrenamiento, y_entrenamiento, clases=corpus.clases, glosas=corpus.glosas,
        huellas={**corpus.huellas, "particion.huella": lado.huella},
    )
