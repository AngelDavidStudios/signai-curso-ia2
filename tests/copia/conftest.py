"""Fixtures de la suite de la copia (contracts/canal-copia.md).

Todo es sintético: ninguna fixture lee `~/signai-data` ni las grabaciones crudas, y ninguna clave es
fija. Los repositorios de restic viven en `tmp_path`.
"""

import os
import random
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from signai_ml.copia import clave


def base_mlflow(ruta: Path, corridas: int = 2, en_curso: int = 0) -> Path:
    """Una base SQLite con las dos tablas de MLflow que la copia lee (research R-11, R-12)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    artefactos = (ruta.parent / "artifacts").as_uri()
    with closing(sqlite3.connect(ruta)) as conexion, conexion:
        conexion.execute("CREATE TABLE experiments (experiment_id INTEGER, name TEXT, artifact_location TEXT)")
        conexion.execute(
            "CREATE TABLE runs (run_uuid TEXT, experiment_id INTEGER, status TEXT, artifact_uri TEXT)"
        )
        conexion.execute("INSERT INTO experiments VALUES (1, 'prueba', ?)", (f"{artefactos}/1",))
        estados = ["FINISHED"] * corridas + ["RUNNING"] * en_curso
        for indice, estado in enumerate(estados):
            conexion.execute(
                "INSERT INTO runs VALUES (?, 1, ?, ?)", (f"r{indice}", estado, f"{artefactos}/1/r{indice}")
            )
    return ruta


@pytest.fixture
def raiz(tmp_path: Path) -> Path:
    """Una raíz de datos sintética: almacén de DVC, seguimiento y los directorios excluidos."""
    raiz = tmp_path / "signai-data"
    generador = random.Random(21)
    for indice in range(20):
        huella = f"{generador.getrandbits(128):032x}"
        objeto = raiz / "dvc-store" / "files" / "md5" / huella[:2] / huella[2:]
        objeto.parent.mkdir(parents=True, exist_ok=True)
        objeto.write_bytes(generador.randbytes(1024 * (1 + indice * 10)))
    base_mlflow(raiz / "mlflow" / "mlflow.db")
    for excluido in ("revision", "cierre", "studio"):
        (raiz / excluido).mkdir(parents=True)
        (raiz / excluido / "temporal.bin").write_bytes(b"desechable")
    return raiz


@pytest.fixture
def clave_prueba() -> str:
    return clave.generar()


@pytest.fixture
def restic_disponible() -> str:
    """Las integraciones necesitan restic. En el CI su falta es un fallo; fuera, se saltan (R-13)."""
    binario = shutil.which("restic")
    if binario is None:
        mensaje = "falta restic en el PATH (brew install restic)"
        if os.environ.get("CI") == "true":
            pytest.fail(mensaje)
        pytest.skip(mensaje)
    return binario
