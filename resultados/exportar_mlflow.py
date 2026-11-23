"""Exporta a CSV las corridas de los barridos b01 y b02 desde la base de MLflow.

Abre la base en modo de solo lectura, así que puede ejecutarse mientras un barrido sigue en curso.
Escribe dos archivos junto a este guion:

- `corridas.csv`: una fila por corrida, con su configuración, su semilla, su rotación, su estado y
  las métricas de la época elegida.
- `epocas.csv`: una fila por corrida y época, con las métricas de entrenamiento y de validación.

Uso:
    uv run python resultados/exportar_mlflow.py ~/signai-data/mlflow/mlflow.db
"""

import csv
import sqlite3
import sys
from pathlib import Path

BARRIDOS = ("reconocedor/b01", "reconocedor/b02")
ETIQUETAS = ("barrido", "ronda", "configuracion", "rotacion", "fase", "estado", "modelo.parametros", "particion.huella")
PARAMETROS = (
    "semilla",
    "forma.capas",
    "forma.dimension",
    "forma.cabezas",
    "forma.intermedia",
    "entrenamiento.epocas",
    "entrenamiento.dropout",
    "entrenamiento.tasa",
    "entrenamiento.regularizacion",
    "entrenamiento.suavizado_etiquetas",
    "entrenamiento.lote",
)
FINALES = (
    "elegida.epoca",
    "elegida.top1",
    "elegida.f1_macro",
    "elegida.acierto_95",
    "elegida.falsas_95",
    "duracion_s",
)
POR_EPOCA = (
    "entrenamiento.perdida",
    "entrenamiento.top1",
    "validacion.perdida",
    "validacion.top1",
    "validacion.f1_macro",
    "validacion.acierto_95",
    "validacion.falsas_95",
)


def exportar(base: Path, destino: Path) -> tuple[int, int]:
    conexion = sqlite3.connect(f"file:{base}?mode=ro", uri=True)
    consulta = lambda sql, *args: conexion.execute(sql, args).fetchall()  # noqa: E731
    experimentos = dict(consulta("select name, experiment_id from experiments"))
    corridas, epocas = [], []
    for nombre in BARRIDOS:
        if nombre not in experimentos:
            continue
        for (run_id,) in consulta("select run_uuid from runs where experiment_id = ?", experimentos[nombre]):
            etiquetas = dict(consulta("select key, value from tags where run_uuid = ?", run_id))
            if "configuracion" not in etiquetas:
                continue
            parametros = dict(consulta("select key, value from params where run_uuid = ?", run_id))
            finales = dict(consulta("select key, value from latest_metrics where run_uuid = ?", run_id))
            fila = {"run_id": run_id}
            fila |= {clave: etiquetas.get(clave, "") for clave in ETIQUETAS}
            fila |= {clave: parametros.get(clave, "") for clave in PARAMETROS}
            fila |= {clave: finales.get(clave, "") for clave in FINALES}
            corridas.append(fila)
            serie: dict[int, dict[str, float]] = {}
            for clave, paso, valor in consulta(
                "select key, step, value from metrics where run_uuid = ? order by timestamp", run_id
            ):
                if clave in POR_EPOCA:
                    serie.setdefault(paso, {})[clave] = valor
            for paso in sorted(serie):
                epocas.append({"run_id": run_id, "epoca": paso} | {clave: serie[paso].get(clave, "") for clave in POR_EPOCA})
    _escribir(destino / "corridas.csv", corridas)
    _escribir(destino / "epocas.csv", epocas)
    return len(corridas), len(epocas)


def _escribir(ruta: Path, filas: list[dict]) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    total, filas = exportar(Path(sys.argv[1]).expanduser(), Path(__file__).parent)
    print(f"{total} corridas y {filas} filas por época exportadas")
