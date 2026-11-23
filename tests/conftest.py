"""Fixtures comunes a todas las suites.

Ninguna prueba toca ~/signai-data: cada una recibe su propia raíz de datos temporal.
"""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def raiz_datos(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    raiz = tmp_path_factory.mktemp("signai-data")
    monkeypatch.setenv("SIGNAI_DATA_ROOT", str(raiz))
    return raiz


@pytest.fixture(scope="session")
def raiz_repo() -> Path:
    salida = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    )
    return Path(salida.stdout.strip())
