"""El seguimiento de MLflow visto desde la copia (research R-11, R-12).

Solo lee la base SQLite —en modo solo lectura— para contar corridas y comprobar su integridad tras
restaurar, y reescribe el prefijo `file://` de las rutas de artefactos cuando se restaura en otra
raíz. No usa MLflow: así no abre la base con un esquema que la migre.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from .errores import RechazoDominio

# Tablas con rutas de artefactos. `logged_models` solo existe a partir de MLflow 3.
_RUTAS_DE_ARTEFACTOS = (("experiments", "artifact_location"), ("runs", "artifact_uri"), ("logged_models", "artifact_location"))


def _abrir(db: Path, *, escritura: bool = False) -> sqlite3.Connection:
    db = Path(db)
    if not db.is_file():
        raise RechazoDominio(f"{db.name} no es una base de MLflow: no existe")
    modo = "rw" if escritura else "ro"
    conexion = sqlite3.connect(f"{db.as_uri()}?mode={modo}", uri=True)
    try:
        tablas = {fila[0] for fila in conexion.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    except sqlite3.DatabaseError as error:
        conexion.close()
        raise RechazoDominio(f"{db.name} no es una base de MLflow: {error}") from error
    if not {"experiments", "runs"} <= tablas:
        conexion.close()
        raise RechazoDominio(f"{db.name} no es una base de MLflow: faltan las tablas experiments y runs")
    return conexion


def recuento(db: Path) -> tuple[int, int]:
    """(experimentos, corridas)."""
    with closing(_abrir(db)) as conexion:
        experimentos = conexion.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        corridas = conexion.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    return experimentos, corridas


def integridad(db: Path) -> str:
    with closing(_abrir(db)) as conexion:
        return conexion.execute("PRAGMA integrity_check").fetchone()[0]


def reubicar(db: Path, de: str, a: str) -> int:
    """Cambia el prefijo `de` por `a` en las rutas de artefactos. Devuelve las filas cambiadas."""
    with closing(_abrir(db, escritura=True)) as conexion, conexion:
        tablas = {fila[0] for fila in conexion.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        cambiadas = 0
        for tabla, columna in _RUTAS_DE_ARTEFACTOS:
            if tabla in tablas:
                cursor = conexion.execute(
                    f"UPDATE {tabla} SET {columna} = ? || substr({columna}, ?) WHERE substr({columna}, 1, ?) = ?",
                    (a, len(de) + 1, len(de), de),
                )
                cambiadas += cursor.rowcount
    return cambiadas


def corridas_en_curso(db: Path) -> int:
    """Corridas con estado `RUNNING`. Mientras haya alguna, la base puede estar escribiéndose (FR-012)."""
    if not Path(db).exists():
        return 0
    with closing(_abrir(db)) as conexion:
        return conexion.execute("SELECT COUNT(*) FROM runs WHERE status = 'RUNNING'").fetchone()[0]
