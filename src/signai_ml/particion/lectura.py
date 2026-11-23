"""Lectura de la partición versión 2 por consumidor (contracts/lectura.md L-1a, L-1b, L-2; TT-22).

Es la única vía por la que el entrenamiento y la aceptación obtienen sus clips. Una interfaz por
consumidor (ISP):

- la **selección** recibe las dos rotaciones (entrena un pliegue y valida el otro) y no tiene a mano
  la prueba;
- el **entrenamiento final** recibe los dos pliegues juntos, sin validación ni prueba;
- la **aceptación** recibe solo la prueba.

Las tres comprueban antes la ligadura con el corpus procesado y devuelven la huella, que las corridas
registran en MLflow como `particion.huella`.

    rotaciones = para_seleccion(raiz)
    x_entrenamiento = ventanas[rotaciones.a.entrenamiento]
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import Lado
from .almacen import ParticionLeida
from .verificacion import comprobar_ligadura


@dataclass(frozen=True)
class Rotacion:
    entrenamiento: np.ndarray
    validacion: np.ndarray


@dataclass(frozen=True)
class Rotaciones:
    a: Rotacion
    b: Rotacion
    huella: str


@dataclass(frozen=True)
class LadoFinal:
    entrenamiento: np.ndarray
    huella: str


@dataclass(frozen=True)
class LadoPrueba:
    prueba: np.ndarray
    huella: str


def _filas(leida: ParticionLeida, *lados: Lado) -> np.ndarray:
    filas = np.array(sorted(a.fila for a in leida.asignaciones if a.lado in lados), dtype=np.int64)
    filas.setflags(write=False)
    return filas


def para_seleccion(raiz: Path) -> Rotaciones:
    leida = comprobar_ligadura(raiz)
    a, b = _filas(leida, Lado.PLIEGUE_A), _filas(leida, Lado.PLIEGUE_B)
    return Rotaciones(
        a=Rotacion(entrenamiento=a, validacion=b),
        b=Rotacion(entrenamiento=b, validacion=a),
        huella=leida.manifiesto["huella"],
    )


def para_entrenamiento_final(raiz: Path) -> LadoFinal:
    leida = comprobar_ligadura(raiz)
    return LadoFinal(
        entrenamiento=_filas(leida, Lado.PLIEGUE_A, Lado.PLIEGUE_B), huella=leida.manifiesto["huella"]
    )


def para_evaluacion(raiz: Path) -> LadoPrueba:
    leida = comprobar_ligadura(raiz)
    return LadoPrueba(prueba=_filas(leida, Lado.PRUEBA), huella=leida.manifiesto["huella"])
