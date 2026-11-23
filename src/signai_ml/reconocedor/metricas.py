"""Métricas de validación (spec FR-007; research R-09).

Todas reciben las probabilidades `(N, C)` y las etiquetas `(N,)`. El umbral del 95 % es el de la
aplicación (ADR-0020): «acierto al 95 %» son los clips bien clasificados con confianza ≥ 0,95 y
«glosas falsas al 95 %», los mal clasificados con esa confianza; ambos sobre el total de clips.
"""

import numpy as np

from . import UMBRAL_CONFIANZA


def _prediccion(probabilidades: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return probabilidades.argmax(axis=1), probabilidades.max(axis=1)


def top1(probabilidades: np.ndarray, etiquetas: np.ndarray) -> float:
    return float((_prediccion(probabilidades)[0] == etiquetas).mean())


def f1_por_clase(probabilidades: np.ndarray, etiquetas: np.ndarray, clases: int) -> np.ndarray:
    prediccion = _prediccion(probabilidades)[0]
    f1 = np.zeros(clases)
    for clase in range(clases):
        tp = np.sum((prediccion == clase) & (etiquetas == clase))
        fp = np.sum((prediccion == clase) & (etiquetas != clase))
        fn = np.sum((prediccion != clase) & (etiquetas == clase))
        denominador = tp + 0.5 * (fp + fn)
        f1[clase] = tp / denominador if denominador else 0.0
    return f1


def f1_macro(probabilidades: np.ndarray, etiquetas: np.ndarray, clases: int) -> float:
    return float(f1_por_clase(probabilidades, etiquetas, clases).mean())


def top1_por_clase(probabilidades: np.ndarray, etiquetas: np.ndarray, clases: int) -> list[float]:
    prediccion = _prediccion(probabilidades)[0]
    return [
        float((prediccion[etiquetas == clase] == clase).mean()) if np.any(etiquetas == clase) else 0.0
        for clase in range(clases)
    ]


def acierto_95(probabilidades: np.ndarray, etiquetas: np.ndarray, umbral: float = UMBRAL_CONFIANZA) -> float:
    prediccion, confianza = _prediccion(probabilidades)
    return float(np.mean((prediccion == etiquetas) & (confianza >= umbral)))


def falsas_95(probabilidades: np.ndarray, etiquetas: np.ndarray, umbral: float = UMBRAL_CONFIANZA) -> float:
    prediccion, confianza = _prediccion(probabilidades)
    return float(np.mean((prediccion != etiquetas) & (confianza >= umbral)))


def calcular(probabilidades: np.ndarray, etiquetas: np.ndarray, clases: int) -> dict[str, float]:
    return {
        "top1": top1(probabilidades, etiquetas),
        "f1_macro": f1_macro(probabilidades, etiquetas, clases),
        "acierto_95": acierto_95(probabilidades, etiquetas),
        "falsas_95": falsas_95(probabilidades, etiquetas),
    }
