"""Lecturas del seguimiento de MLflow restaurado: recuento, integridad y reubicación (R-11, R-12)."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from signai_ml.copia.errores import RechazoDominio
from signai_ml.copia.seguimiento import corridas_en_curso, integridad, recuento, reubicar

from .conftest import base_mlflow


def _uris(db: Path) -> list[str]:
    with closing(sqlite3.connect(db)) as conexion:
        experimentos = [fila[0] for fila in conexion.execute("SELECT artifact_location FROM experiments")]
        corridas = [fila[0] for fila in conexion.execute("SELECT artifact_uri FROM runs")]
    return experimentos + corridas


def test_cuenta_experimentos_y_corridas(tmp_path: Path) -> None:
    db = base_mlflow(tmp_path / "mlflow.db", corridas=3, en_curso=1)
    assert recuento(db) == (1, 4)


def test_integridad_de_una_base_sana(tmp_path: Path) -> None:
    assert integridad(base_mlflow(tmp_path / "mlflow.db")) == "ok"


def test_reubica_el_prefijo_y_es_idempotente(tmp_path: Path) -> None:
    db = base_mlflow(tmp_path / "mlflow.db", corridas=2)
    de = (tmp_path / "artifacts").as_uri()
    a = "file:///Users/otro/signai-data/mlflow/artifacts"

    assert reubicar(db, de, a) == 3
    assert all(uri.startswith(a) for uri in _uris(db))
    assert reubicar(db, de, a) == 0


def test_no_toca_las_uris_con_otro_prefijo(tmp_path: Path) -> None:
    db = base_mlflow(tmp_path / "mlflow.db", corridas=1)
    with closing(sqlite3.connect(db)) as conexion, conexion:
        conexion.execute("UPDATE runs SET artifact_uri = 's3://otro/sitio'")

    reubicar(db, (tmp_path / "artifacts").as_uri(), "file:///nuevo")
    assert "s3://otro/sitio" in _uris(db)


def test_reubica_tambien_los_modelos_registrados_si_existen(tmp_path: Path) -> None:
    db = base_mlflow(tmp_path / "mlflow.db", corridas=1)
    de = (tmp_path / "artifacts").as_uri()
    with closing(sqlite3.connect(db)) as conexion, conexion:
        conexion.execute("CREATE TABLE logged_models (model_id TEXT, artifact_location TEXT)")
        conexion.execute("INSERT INTO logged_models VALUES ('m1', ?)", (f"{de}/m1",))

    assert reubicar(db, de, "file:///nuevo") == 3


@pytest.mark.parametrize("contenido", [b"no es una base", b""])
def test_un_archivo_que_no_es_de_mlflow_se_rechaza(tmp_path: Path, contenido: bytes) -> None:
    falsa = tmp_path / "falsa.db"
    falsa.write_bytes(contenido)
    for operacion in (recuento, integridad):
        with pytest.raises(RechazoDominio, match="MLflow"):
            operacion(falsa)
    with pytest.raises(RechazoDominio, match="MLflow"):
        reubicar(falsa, "file:///a", "file:///b")


def test_una_base_inexistente_se_rechaza(tmp_path: Path) -> None:
    with pytest.raises(RechazoDominio, match="MLflow"):
        recuento(tmp_path / "no-existe.db")


def test_las_lecturas_no_modifican_la_base(tmp_path: Path) -> None:
    db = base_mlflow(tmp_path / "mlflow.db")
    antes = db.read_bytes()
    recuento(db)
    integridad(db)
    assert db.read_bytes() == antes


# ── Corridas en curso (FR-012, R-11) ───────────────────────────────────────────────────────────


def test_cuenta_las_corridas_en_curso(tmp_path: Path) -> None:
    assert corridas_en_curso(base_mlflow(tmp_path / "mlflow.db", corridas=2, en_curso=1)) == 1


def test_sin_corridas_en_curso(tmp_path: Path) -> None:
    assert corridas_en_curso(base_mlflow(tmp_path / "mlflow.db", corridas=2)) == 0


def test_sin_base_todavia_no_hay_corridas(tmp_path: Path) -> None:
    assert corridas_en_curso(tmp_path / "no-existe.db") == 0
