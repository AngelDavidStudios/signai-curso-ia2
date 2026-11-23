"""Registro de corridas en el seguimiento de experimentos local (MLflow).

Cada corrida guarda su configuración, su semilla, sus métricas y las versiones de las bibliotecas
(constitución, Principio VI; ADR-0010). El almacén vive fuera del repositorio, en
``$SIGNAI_DATA_ROOT/mlflow`` (por defecto ``~/signai-data/mlflow``).
"""

import math
import os
import platform
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag

Escalar = str | int | float | bool

_BIBLIOTECAS = ("torch", "coremltools", "numpy", "mlflow", "dvc")


def _raiz_datos(raiz: Path | None) -> Path:
    if raiz is not None:
        return Path(raiz)
    return Path(os.environ.get("SIGNAI_DATA_ROOT", Path.home() / "signai-data"))


def uri_seguimiento(raiz: Path | None = None) -> str:
    """URI del almacén SQLite de MLflow bajo la raíz de datos."""
    return f"sqlite:///{_raiz_datos(raiz) / 'mlflow' / 'mlflow.db'}"


def versiones_bibliotecas() -> dict[str, str]:
    """Versiones de Python y de las bibliotecas de las que depende el resultado de una corrida."""
    return {"python": platform.python_version()} | {b: version(b) for b in _BIBLIOTECAS}


def _validar(experimento: str, semilla: int, metricas: Mapping[str, float]) -> None:
    if not experimento.strip():
        raise ValueError("El experimento es obligatorio y no puede estar vacío")
    if isinstance(semilla, bool) or not isinstance(semilla, int) or semilla < 0:
        raise ValueError(f"La semilla debe ser un entero ≥ 0; se recibió {semilla!r}")
    if not metricas:
        raise ValueError("Una corrida sin métricas no se registra")
    no_finitas = [nombre for nombre, valor in metricas.items() if not math.isfinite(valor)]
    if no_finitas:
        raise ValueError(f"Métricas no finitas: {', '.join(no_finitas)}")


def registrar_corrida(
    experimento: str,
    configuracion: Mapping[str, Escalar],
    semilla: int,
    metricas: Mapping[str, float],
    raiz: Path | None = None,
) -> str:
    """Registra una corrida completa y devuelve su id.

    Valida antes de escribir: si los datos no son válidos, no se crea nada en el almacén.
    """
    _validar(experimento, semilla, metricas)

    cliente, id_experimento = cliente_y_experimento(experimento, raiz)
    corrida = cliente.create_run(id_experimento)
    id_corrida = corrida.info.run_id
    parametros = {**configuracion, "semilla": semilla}
    cliente.log_batch(
        id_corrida,
        metrics=[Metric(k, float(v), 0, 0) for k, v in metricas.items()],
        params=[Param(k, str(v)) for k, v in parametros.items()],
        tags=[RunTag(f"version.{b}", v) for b, v in versiones_bibliotecas().items()],
    )
    cliente.set_terminated(id_corrida)
    return id_corrida


# --- API incremental (TT-14 · SP-215; specs/006-entrenamiento-reconocedor, research R-10) ---------

ESTADOS_FINALES = ("FINISHED", "FAILED", "KILLED")


def cliente_y_experimento(experimento: str, raiz: Path | None = None) -> tuple[MlflowClient, str]:
    """El cliente del almacén local y el id del experimento, que se crea si no existe."""
    artefactos = _raiz_datos(raiz) / "mlflow" / "artifacts"
    artefactos.mkdir(parents=True, exist_ok=True)
    cliente = MlflowClient(tracking_uri=uri_seguimiento(raiz))
    existente = cliente.get_experiment_by_name(experimento)
    if existente is not None:
        return cliente, existente.experiment_id
    return cliente, cliente.create_experiment(experimento, artifact_location=artefactos.as_uri())


def _comprobar_finitas(metricas: Mapping[str, float]) -> None:
    no_finitas = [nombre for nombre, valor in metricas.items() if not math.isfinite(valor)]
    if no_finitas:
        raise ValueError(f"Métricas no finitas: {', '.join(no_finitas)}")


def abrir_corrida(
    experimento: str,
    nombre: str,
    parametros: Mapping[str, Escalar],
    etiquetas: Mapping[str, str],
    raiz: Path | None = None,
) -> str:
    """Crea una corrida en curso con sus parámetros, sus etiquetas y las versiones de las bibliotecas."""
    if not experimento.strip():
        raise ValueError("El experimento es obligatorio y no puede estar vacío")
    cliente, id_experimento = cliente_y_experimento(experimento, raiz)
    id_corrida = cliente.create_run(id_experimento, run_name=nombre).info.run_id
    versiones = {f"version.{b}": v for b, v in versiones_bibliotecas().items()}
    cliente.log_batch(
        id_corrida,
        params=[Param(k, str(v)) for k, v in parametros.items()],
        tags=[RunTag(k, str(v)) for k, v in {**versiones, **etiquetas}.items()],
    )
    return id_corrida


def registrar_epoca(id_corrida: str, epoca: int, metricas: Mapping[str, float], raiz: Path | None = None) -> None:
    """Guarda las métricas de una época con `step = epoca`."""
    _comprobar_finitas(metricas)
    MlflowClient(tracking_uri=uri_seguimiento(raiz)).log_batch(
        id_corrida, metrics=[Metric(k, float(v), 0, epoca) for k, v in metricas.items()]
    )


def cerrar_corrida(
    id_corrida: str,
    estado: str,
    metricas: Mapping[str, float] | None = None,
    artefactos: Sequence[Path] = (),
    etiquetas: Mapping[str, str] | None = None,
    raiz: Path | None = None,
) -> None:
    """Cierra la corrida como `FINISHED`, `FAILED` o `KILLED`, con sus métricas finales y artefactos."""
    if estado not in ESTADOS_FINALES:
        raise ValueError(f"El estado debe ser uno de {', '.join(ESTADOS_FINALES)}; se recibió {estado!r}")
    metricas = dict(metricas or {})
    _comprobar_finitas(metricas)
    cliente = MlflowClient(tracking_uri=uri_seguimiento(raiz))
    cliente.log_batch(
        id_corrida,
        metrics=[Metric(k, float(v), 0, 0) for k, v in metricas.items()],
        tags=[RunTag(k, str(v)) for k, v in (etiquetas or {}).items()],
    )
    for artefacto in artefactos:
        cliente.log_artifact(id_corrida, str(artefacto))
    cliente.set_terminated(id_corrida, status=estado)
