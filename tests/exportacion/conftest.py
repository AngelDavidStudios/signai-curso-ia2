"""Fixtures de la suite `exportacion` (TT-15 · SP-338).

Un KPoseFormer diminuto, entrenado unos pasos sobre ventanas sintéticas para que sus predicciones
sean claras. Así el redondeo de 16 bits no puede empatar dos clases, y la paridad mide la conversión,
no un empate. Ninguna fixture lee `datos/`, `modelos/` ni el MLflow reales.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from signai_ml.reconocedor.configuracion import Forma
from signai_ml.reconocedor.modelo import KPoseFormer
from reconocedor.sinteticos import nombre_clase, ventanas_sinteticas

CLASES_DIMINUTO = 10
SEMILLA = 338
PASOS = 60


@pytest.fixture(scope="session")
def forma_diminuta() -> Forma:
    return Forma(capas=1, dimension=16, cabezas=2, intermedia=16)


@pytest.fixture(scope="session")
def clases_diminuto() -> list[str]:
    return [nombre_clase(n) for n in range(1, CLASES_DIMINUTO + 1)]


@pytest.fixture(scope="session")
def modelo_diminuto(forma_diminuta: Forma) -> KPoseFormer:
    """Modelo de 10 clases en `eval()`, compartido por la sesión.

    No se modifica: quien necesite alterarlo hace `copy.deepcopy`.
    """
    torch.manual_seed(SEMILLA)
    modelo = KPoseFormer(forma_diminuta, clases=CLASES_DIMINUTO)
    x, y = ventanas_sinteticas(CLASES_DIMINUTO, 6, semilla=SEMILLA)
    entrada, etiquetas = torch.from_numpy(x), torch.from_numpy(y)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
    for _ in range(PASOS):
        optimizador.zero_grad()
        torch.nn.functional.cross_entropy(modelo(entrada), etiquetas).backward()
        optimizador.step()
    return modelo.eval()


@pytest.fixture(scope="session")
def ventanas_prueba() -> tuple[np.ndarray, np.ndarray]:
    """20 ventanas `(75, 96)` `float32` que el modelo no vio, con algunos puntos a `-1`, y sus etiquetas."""
    return ventanas_sinteticas(CLASES_DIMINUTO, 2, semilla=SEMILLA + 1)


@pytest.fixture(scope="session")
def ruta_casos_referencia(raiz_repo: Path) -> Path:
    """Copia versionada del contrato de normalización (12 casos sintéticos, sin datos de señantes)."""
    return raiz_repo / "tests" / "referencias" / "preprocesamiento" / "normalizacion-casos.json"
