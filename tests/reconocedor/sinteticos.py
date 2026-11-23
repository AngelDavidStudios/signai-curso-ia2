"""Datos sintéticos de la suite `reconocedor`: ventanas inventadas, sin datos de señantes.

Cada clase tiene un patrón propio y fácil de distinguir, con algunos puntos ausentes (`-1`), para que
un modelo diminuto aprenda algo en una o dos épocas sobre CPU.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

COLUMNAS_INDICE = (
    "fila", "clip_id", "clase", "numero", "glosa", "grabacion", "repeticion",
    "n_fotogramas_60", "n_fotogramas_30", "md5_clip",
)


def nombre_clase(numero: int) -> str:
    return f"{numero}-Sint{numero}"


def ventanas_sinteticas(n_clases: int, por_clase: int, semilla: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`(X float32 (N, 75, 96), y int64 (N,))`, ordenadas por clase."""
    rng = np.random.default_rng(semilla)
    t = np.linspace(0.0, 1.0, 75, dtype=np.float32)[:, None]
    rasgos = np.arange(96, dtype=np.float32)[None, :]
    ventanas, etiquetas = [], []
    for clase in range(n_clases):
        patron = np.sin(2 * np.pi * (clase + 1) * t + rasgos * 0.1 * (clase + 1)).astype(np.float32)
        for _ in range(por_clase):
            ventana = patron + rng.normal(0, 0.05, size=patron.shape).astype(np.float32)
            ausentes = rng.random((75, 48)) < 0.03
            ventana.reshape(75, 48, 2)[ausentes] = -1.0
            ventanas.append(ventana)
            etiquetas.append(clase)
    return np.stack(ventanas).astype(np.float32), np.array(etiquetas, dtype=np.int64)


def vocabulario_sintetico(n_clases: int) -> list[dict[str, str]]:
    return [
        {"numero": str(n), "clase": nombre_clase(n), "glosa": f"glosa{n}", "nombre": f"Sint{n}"}
        for n in range(1, n_clases + 1)
    ]


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def escribir_corpus(raiz: Path, n_clases: int, por_clase: int) -> Path:
    """Escribe `datos/ventanas/` y `config/corpus/vocabulario.csv` pequeños bajo `raiz`."""
    x, y = ventanas_sinteticas(n_clases, por_clase)
    ventanas = raiz / "datos" / "ventanas"
    ventanas.mkdir(parents=True)
    np.save(ventanas / "ventanas.npy", x)
    with open(ventanas / "indice.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo, lineterminator="\n")
        escritor.writerow(COLUMNAS_INDICE)
        for fila, clase in enumerate(y):
            numero = int(clase) + 1
            rep = fila % por_clase + 1
            escritor.writerow([
                fila, f"{nombre_clase(numero)}__SINT_1__rep{rep:02d}", nombre_clase(numero), numero,
                f"glosa{numero}", "SINT_1", rep, 80, 40, "0" * 32,
            ])
    manifiesto = {"ventanas": len(y), "forma": list(x.shape), "sha256_ventanas": sha256(ventanas / "ventanas.npy")}
    (ventanas / "manifiesto.json").write_text(json.dumps(manifiesto), encoding="utf-8")
    corpus = raiz / "config" / "corpus"
    corpus.mkdir(parents=True)
    with open(corpus / "vocabulario.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=("numero", "clase", "glosa", "nombre"), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(vocabulario_sintetico(n_clases))
    return raiz


def reparto_sintetico(n_clases: int, por_clase: int, validacion: int) -> tuple[np.ndarray, np.ndarray]:
    """Filas de entrenamiento y validación: las `validacion` últimas de cada clase validan."""
    entrenamiento, valida = [], []
    for clase in range(n_clases):
        filas = list(range(clase * por_clase, (clase + 1) * por_clase))
        entrenamiento += filas[: por_clase - validacion]
        valida += filas[por_clase - validacion :]
    return np.array(entrenamiento, dtype=np.int64), np.array(valida, dtype=np.int64)
