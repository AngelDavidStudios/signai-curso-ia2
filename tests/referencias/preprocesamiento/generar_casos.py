"""Genera `normalizacion-casos.json`, los casos de referencia del contrato de normalización (N-8).

Entradas sintéticas, sin datos de señantes, con semilla fija; las salidas las calcula la
implementación de referencia. Uso:

    uv run python tests/referencias/preprocesamiento/generar_casos.py

Volver a ejecutarlo con la misma implementación da el mismo fichero, byte a byte.
"""

import json
from pathlib import Path

import numpy as np

from signai_ml.preproceso import NORMALIZACION
from signai_ml.preproceso.normalizacion import Normalizador

SEMILLA = 20261104
AUS = -1.0
DESTINO = Path(__file__).with_name("normalizacion-casos.json")


def _vacio(t: int) -> np.ndarray:
    return np.full((t, 96), AUS)


def _poner(kp: np.ndarray, punto: int, x, y) -> None:
    kp[:, 2 * punto] = x
    kp[:, 2 * punto + 1] = y


def _aleatoria(rng: np.random.Generator, t: int, ausencia: float) -> np.ndarray:
    """Una persona que se mueve con suavidad, con un `ausencia` de puntos perdidos al azar."""
    base = rng.uniform(0.25, 0.75, size=96)
    deriva = rng.normal(0, 0.004, size=(t, 96)).cumsum(axis=0)
    kp = base + deriva
    perdidos = rng.random((t, 48)) < ausencia
    kp.reshape(t, 48, 2)[perdidos] = AUS
    return kp


def casos(rng: np.random.Generator) -> list[tuple[str, str, np.ndarray]]:
    salida: list[tuple[str, str, np.ndarray]] = []

    kp = _aleatoria(rng, 75, 0.0)
    salida.append(("identidad_75", "T = 75 sin ausencias: el remuestreo es la identidad", kp))

    for t in (1, 18, 102, 203):
        salida.append((f"longitud_{t}", f"T = {t}, pseudoaleatoria con un 20 % de puntos perdidos", _aleatoria(rng, t, 0.20)))

    kp = _aleatoria(rng, 20, 0.0)
    kp[5:12, 2 * 35:2 * 35 + 2] = AUS  # punto 35 = 27 + 8: punta del índice derecho (E-2)
    salida.append(("hueco_interior", "la punta del índice derecho se pierde 7 fotogramas", kp))

    kp = _aleatoria(rng, 20, 0.0)
    kp[:4, 2 * 12:2 * 12 + 2] = AUS
    kp[-4:, 2 * 12:2 * 12 + 2] = AUS
    salida.append(("huecos_extremos", "un punto falta al principio y al final", kp))

    kp = _aleatoria(rng, 20, 0.0)
    kp[3:15, 2 * 4:2 * 4 + 2] = AUS  # muñeca izquierda de pose
    kp[:, 2 * 5:2 * 5 + 2] = AUS  # muñeca derecha de pose, en todo el clip
    kp[:, 2 * 6:2 * 6 + 2] += np.linspace(0, 0.05, 20)[:, None]  # la mano se mueve
    salida.append(("respaldo_muneca", "las muñecas de pose faltan y las de mano están", kp))

    kp = _aleatoria(rng, 20, 0.05)
    kp[:, 2 * 6:2 * 27] = AUS
    salida.append(("mano_nunca_presente", "la mano izquierda no aparece: se ancla a su muñeca", kp))

    kp = _aleatoria(rng, 20, 0.05)
    kp[:, 0:4] = AUS
    salida.append(("hombros_nunca_presentes", "sin hombros: se anclan y el tramo no se normaliza", kp))

    kp = _aleatoria(rng, 20, 0.0)
    kp[7, 0] += 0.15  # un fotograma con el hombro desplazado: la mediana lo ignora, la media no
    salida.append(("ancho_variable", "ancho de hombros con un fotograma atípico", kp))

    salida.append(("todo_ausente", "ningún punto en todo el clip", _vacio(10)))
    return salida


def _redondear(m: np.ndarray, decimales: int) -> list[list[float]]:
    """Redondea y escribe enteros sin decimales (−1, 0): el fichero debe pesar < 1 MB (canal)."""
    filas = []
    for fila in m:
        valores = [round(float(v), decimales) for v in fila]
        filas.append([int(v) if v == int(v) else v for v in valores])
    return filas


def generar() -> dict:
    rng = np.random.default_rng(SEMILLA)
    normalizador = Normalizador()
    lista = []
    for nombre, descripcion, entrada in casos(rng):
        entrada32 = np.asarray(_redondear(entrada, 4), dtype=np.float32)
        ventana = normalizador.normalizar(entrada32)
        lista.append({
            "nombre": nombre,
            "descripcion": descripcion,
            "entrada": _redondear(entrada32, 4),
            "salida": _redondear(ventana.valores, 5),
            "sin_normalizar": ventana.sin_normalizar,
        })
    return {
        "contrato": "normalizacion",
        "version": NORMALIZACION,
        "tolerancia": 1e-4,
        "semilla": SEMILLA,
        "casos": lista,
    }


if __name__ == "__main__":
    DESTINO.write_text(json.dumps(generar(), ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(DESTINO)
