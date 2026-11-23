"""Corpus procesado sintético para la suite `particion` (TT-13 y TT-22; constitución, Seguridad).

Sin datos de señantes: clases, glosas, grabaciones e identificadores son inventados. Las ventanas son
mínimas, de forma `(N, 1)`: la partición solo usa N y la huella de `ventanas.npy`, nunca la forma
`(75, 96)`, y el ejecutor del CI comparte temporales con el equipo (tasks.md T005).
"""

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from signai_ml.preproceso.almacen import COLUMNAS_INDICE

GRABACIONES_POR_CLASE = 3

# Como el corpus aprobado: 100 clases, 96 con 90 clips y 4 con 89 (nombres inventados).
CLASES_DE_89 = ("8-Sint", "24-Sint", "36-Sint", "95-Sint")
RECUENTOS_CORPUS: dict[str, int] = {
    f"{numero}-Sint": 89 if f"{numero}-Sint" in CLASES_DE_89 else 90 for numero in range(1, 101)
}


def indice_sintetico(
    recuentos: Mapping[str, int], grabaciones: Mapping[str, int] | None = None
) -> list[dict[str, str]]:
    """Filas de `indice.csv` con las columnas de TT-12, en orden de `clip_id` y numeradas desde 0.

    Cada clase tiene `GRABACIONES_POR_CLASE` grabaciones (o las que diga `grabaciones`), y sus clips se
    reparten entre ellas por turno: con 90 clips y 3 grabaciones, 30 cada una (TT-22, research R-01).
    """
    grabaciones = grabaciones or {}
    filas: list[dict[str, str]] = []
    for clase, n in recuentos.items():
        numero = clase.split("-", 1)[0]
        por_clase = grabaciones.get(clase, GRABACIONES_POR_CLASE)
        for i in range(n):
            grabacion = f"SINT_{numero}_{i % por_clase}"
            repeticion = i // por_clase + 1
            filas.append({
                "clip_id": f"{clase}__{grabacion}__rep{repeticion:02d}",
                "clase": clase,
                "numero": numero,
                "glosa": clase.lower(),
                "grabacion": grabacion,
                "repeticion": str(repeticion),
                "n_fotogramas_60": "80",
                "n_fotogramas_30": "40",
                "md5_clip": hashlib.md5(f"{clase}{i}".encode()).hexdigest(),
            })
    filas.sort(key=lambda f: f["clip_id"])
    for numero_fila, fila in enumerate(filas):
        fila["fila"] = str(numero_fila)
    return filas


def escribir_indice(dir_ventanas: Path, filas: Sequence[Mapping[str, str]]) -> None:
    dir_ventanas.mkdir(parents=True, exist_ok=True)
    with (dir_ventanas / "indice.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_INDICE, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows({c: fila[c] for c in COLUMNAS_INDICE} for fila in filas)


def escribir_corpus(raiz: Path, filas: Sequence[Mapping[str, str]]) -> Path:
    """`datos/ventanas/` sintético: índice, ventanas mínimas y manifiesto con su huella real."""
    dir_ventanas = raiz / "datos" / "ventanas"
    escribir_indice(dir_ventanas, filas)
    ruta_npy = dir_ventanas / "ventanas.npy"
    with ruta_npy.open("wb") as f:
        np.save(f, np.zeros((len(filas), 1), dtype=np.float32), allow_pickle=False)
    manifiesto = {
        "ventanas": len(filas),
        "forma": [len(filas), 1],
        "sha256_ventanas": hashlib.sha256(ruta_npy.read_bytes()).hexdigest(),
        "normalizacion": "sintetica",
    }
    (dir_ventanas / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
    return raiz


def leer_csv_particion(raiz: Path) -> list[dict[str, str]]:
    with (raiz / "datos" / "particion" / "particion.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def reescribir_particion(raiz: Path, filas: Sequence[Mapping[str, str]]) -> None:
    """Reescribe `particion.csv` en formato canónico y pone su huella en el manifiesto.

    Simula una partición alterada «con cuidado»: la huella cuadra, así que solo la detectan las
    comprobaciones de recuentos y de re-sorteo.
    """
    base = raiz / "datos" / "particion"
    ruta_csv = base / "particion.csv"
    with ruta_csv.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(
            f, fieldnames=("fila", "clip_id", "clase", "grabacion", "lado"), lineterminator="\n"
        )
        escritor.writeheader()
        escritor.writerows(filas)
    ruta_manifiesto = base / "manifiesto.json"
    manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
    manifiesto["huella"] = hashlib.sha256(ruta_csv.read_bytes()).hexdigest()
    ruta_manifiesto.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")


def intercambiar_lados(filas: list[dict[str, str]], clase: str, lado_a: str, lado_b: str) -> list[dict[str, str]]:
    """Intercambia el lado del primer clip de `lado_a` y el primero de `lado_b` en `clase`."""
    a = next(f for f in filas if f["clase"] == clase and f["lado"] == lado_a)
    b = next(f for f in filas if f["clase"] == clase and f["lado"] == lado_b)
    a["lado"], b["lado"] = lado_b, lado_a
    return filas


def grabaciones_por_lado(filas: Sequence[Mapping[str, str]], clase: str) -> dict[str, set[str]]:
    """Las grabaciones de `clase` en cada lado de `particion.csv`."""
    lados: dict[str, set[str]] = {}
    for fila in filas:
        if fila["clase"] == clase:
            lados.setdefault(fila["lado"], set()).add(fila["grabacion"])
    return lados


def intercambiar_grabaciones(filas: list[dict[str, str]], clase: str, lado_a: str, lado_b: str) -> list[dict[str, str]]:
    """Intercambia el lado de las grabaciones de `lado_a` y `lado_b` en `clase`: cada grabación sigue entera."""
    for fila in filas:
        if fila["clase"] == clase and fila["lado"] in (lado_a, lado_b):
            fila["lado"] = lado_b if fila["lado"] == lado_a else lado_a
    return filas


def escribir_particion_version_1(raiz: Path) -> Path:
    """Una partición con el formato de TT-13 (sin `grabacion`, versión 1): la que TT-22 sustituye."""
    base = raiz / "datos" / "particion"
    base.mkdir(parents=True, exist_ok=True)
    with (raiz / "datos" / "ventanas" / "indice.csv").open(encoding="utf-8", newline="") as f:
        indice = list(csv.DictReader(f))
    ruta_csv = base / "particion.csv"
    with ruta_csv.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.writer(f, lineterminator="\n")
        escritor.writerow(("fila", "clip_id", "clase", "lado"))
        escritor.writerows((fila["fila"], fila["clip_id"], fila["clase"], "entrenamiento") for fila in indice)
    manifiesto = {
        "version_particion": "signai-ml/particion/1",
        "semilla": 214,
        "huella": hashlib.sha256(ruta_csv.read_bytes()).hexdigest(),
    }
    (base / "manifiesto.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    (base / "informe.md").write_text("# Partición de evaluación (TT-13)\n", encoding="utf-8")
    return base
