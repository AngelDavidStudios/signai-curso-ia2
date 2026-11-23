"""Fixtures de la suite `particion` (TT-13).

El corpus sintético se escribe una sola vez por sesión y cada prueba recibe su propia copia, que
puede alterar sin afectar a las demás. Nada de aquí lee `datos/` real.
"""

import shutil
from pathlib import Path

import pytest

from .sinteticos import RECUENTOS_CORPUS, escribir_corpus, indice_sintetico


@pytest.fixture(scope="session")
def _plantilla_sintetica(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return escribir_corpus(tmp_path_factory.mktemp("plantilla-particion"), indice_sintetico(RECUENTOS_CORPUS))


@pytest.fixture
def raiz_sintetica(tmp_path: Path, _plantilla_sintetica: Path) -> Path:
    raiz = tmp_path / "repo"
    shutil.copytree(_plantilla_sintetica, raiz)
    return raiz


@pytest.fixture
def raiz_con_particion(raiz_sintetica: Path) -> Path:
    """Corpus sintético con la partición ya generada por la CLI."""
    from signai_ml.particion.cli import main

    assert main(["generar"], raiz=raiz_sintetica) == 0
    return raiz_sintetica
