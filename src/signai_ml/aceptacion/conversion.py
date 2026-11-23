"""Conversión del modelo aceptado a Core ML (spec FR-011; research R-06; ADR-0013, ADR-0014).

Formato de programa (`mlprogram`) y nunca el heredado, que traduce mal la atención sin dar error.
Cómputo, entrada y salida en 16 bits. Objetivo iOS 17 (especificación 8), como el artefacto
provisional que la app ya carga. La conversión no decide si el artefacto vale: lo decide la paridad.
Los avisos de coremltools se devuelven para registrarlos.
"""

import shutil
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import coremltools as ct
import numpy as np
import torch

from ..reconocedor import FOTOGRAMAS, RASGOS
from . import ENTRADA, NOMBRE_PAQUETE, SALIDA
from .errores import ArtefactoNoValido

FORMA_ENTRADA = (1, FOTOGRAMAS, RASGOS)
OBJETIVO = "iOS17"


@dataclass(frozen=True)
class Conversion:
    paquete: Path
    avisos: list[str]


class Convertidor(Protocol):
    def convertir(self, modelo: torch.nn.Module, clases: Sequence[str], destino: Path) -> Conversion: ...


class ConvertidorCoreml:
    def convertir(self, modelo: torch.nn.Module, clases: Sequence[str], destino: Path) -> Conversion:
        """Escribe `destino/KPoseFormer.mlpackage`; el modelo de entrada no se modifica."""
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        modelo = modelo.eval()
        ejemplo = torch.zeros(FORMA_ENTRADA)
        with torch.no_grad():
            salidas = modelo(ejemplo).shape[-1]
        if salidas != len(clases):
            raise ArtefactoNoValido(f"El modelo da {salidas} salidas y hay {len(clases)} clases.")
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            try:
                with torch.no_grad():
                    trazado = torch.jit.trace(modelo, ejemplo)
                convertido = ct.convert(
                    trazado,
                    inputs=[ct.TensorType(name=ENTRADA, shape=FORMA_ENTRADA, dtype=np.float16)],
                    outputs=[ct.TensorType(name=SALIDA, dtype=np.float16)],
                    convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=getattr(ct.target, OBJETIVO),
                )
            except Exception as error:
                raise ArtefactoNoValido(f"La conversión a Core ML falló: {error}") from error
        paquete = destino / NOMBRE_PAQUETE
        if paquete.exists():
            shutil.rmtree(paquete)
        convertido.save(str(paquete))
        return Conversion(paquete=paquete, avisos=sorted({str(aviso.message) for aviso in avisos}))
