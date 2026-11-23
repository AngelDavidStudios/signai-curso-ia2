"""Fixtures de la suite `evaluacion` (TT-15).

Corpus y candidato inventados en `tmp_path` y un lector de la prueba falso. Nada de aquí lee `datos/`,
`modelos/` ni el MLflow reales: `tests/conftest.py` fija `SIGNAI_DATA_ROOT` a un temporal.
"""

from pathlib import Path

import pytest

from .sinteticos import escribir_raiz_prueba, lector_prueba


@pytest.fixture
def raiz_prueba(tmp_path: Path) -> Path:
    return escribir_raiz_prueba(tmp_path / "repo")


@pytest.fixture
def lector_prueba_falso():
    return lector_prueba
