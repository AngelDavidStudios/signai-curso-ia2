"""Ejecución del artefacto Core ML en el Mac, como lo ejecuta la app (research R-08).

La ventana se escribe en el tipo que el artefacto declara (16 bits) y la salida se lee en el suyo,
igual que `CoreMLSignRecognizer`. La app aplica el *softmax* a los `logits`, y aquí se hace en
`float64`.

Dos unidades de cómputo:
- `cpu` (`CPU_ONLY`): la referencia estable;
- `todas` (`ALL`): la configuración de la app, que en Apple Silicon usa el Neural Engine, como el
  iPhone.
"""

import json
from pathlib import Path
from typing import Protocol

import coremltools as ct
import numpy as np
from coremltools.proto import Model_pb2

from . import ENTRADA, SALIDA

UNIDADES = ("cpu", "todas")
_COMPUTO = {"cpu": ct.ComputeUnit.CPU_ONLY, "todas": ct.ComputeUnit.ALL}
_TIPOS = {
    Model_pb2.ArrayFeatureType.FLOAT16: np.float16,
    Model_pb2.ArrayFeatureType.FLOAT32: np.float32,
    Model_pb2.ArrayFeatureType.DOUBLE: np.float64,
}


class Ejecutor(Protocol):
    unidad: str

    def probabilidades(self, paquete: Path, ventanas: np.ndarray) -> np.ndarray: ...


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    exponenciales = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exponenciales / exponenciales.sum(axis=-1, keepdims=True)


class EjecutorCoreml:
    def __init__(self, unidad: str) -> None:
        if unidad not in _COMPUTO:
            raise ValueError(f"Unidad de cómputo desconocida: {unidad!r}; son {', '.join(UNIDADES)}.")
        self.unidad = unidad

    def probabilidades(self, paquete: Path, ventanas: np.ndarray) -> np.ndarray:
        """`softmax(logits)` del artefacto para cada ventana, como `float64 (N, clases)`."""
        modelo = ct.models.MLModel(str(paquete), compute_units=_COMPUTO[self.unidad])
        entrada = {e.name: e.type.multiArrayType.dataType for e in modelo.get_spec().description.input}[ENTRADA]
        tipo = _TIPOS[entrada]
        filas = [
            softmax(np.asarray(modelo.predict({ENTRADA: ventana[None].astype(tipo)})[SALIDA]).reshape(-1))
            for ventana in np.asarray(ventanas)
        ]
        return np.stack(filas)


def casos_referencia(ruta: Path) -> np.ndarray:
    """Las ventanas `salida` de los casos de referencia del contrato de normalización (sintéticos)."""
    casos = json.loads(Path(ruta).read_text(encoding="utf-8"))["casos"]
    return np.array([caso["salida"] for caso in casos], dtype=np.float32)
