"""Medición del candidato sobre la prueba (spec FR-003; research R-03).

`probabilidades` ejecuta el modelo en CPU, en 32 bits y sin gradiente, que es determinista bit a
bit. `evaluar` es pura: con las mismas definiciones que TT-14 (`reconocedor.metricas`) añade el
recuento de aciertos, que es lo que decide (R-04), y la matriz de confusión.
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..reconocedor import metricas


@dataclass(frozen=True)
class ResultadoEvaluacion:
    aciertos: int
    total: int
    top1: float
    f1_macro: float
    acierto_95: float
    falsas_95: float
    top1_por_clase: tuple[float, ...]
    f1_por_clase: tuple[float, ...]
    confusion: np.ndarray  # (clases, clases); filas = clase real

    def metricas(self) -> dict[str, float]:
        """Las métricas escalares que se registran en la consulta."""
        return {
            "aciertos": float(self.aciertos),
            "total": float(self.total),
            "top1": self.top1,
            "f1_macro": self.f1_macro,
            "acierto_95": self.acierto_95,
            "falsas_95": self.falsas_95,
        }


def probabilidades(modelo: torch.nn.Module, x: np.ndarray, lote: int = 200) -> np.ndarray:
    """`softmax(logits)` del modelo en CPU, por lotes, como `float64 (N, clases)`."""
    modelo = modelo.to("cpu").eval()
    partes = []
    with torch.no_grad():
        for inicio in range(0, len(x), lote):
            entrada = torch.from_numpy(np.ascontiguousarray(x[inicio : inicio + lote], dtype=np.float32))
            partes.append(torch.softmax(modelo(entrada), dim=1).double().numpy())
    return np.concatenate(partes)


def confusion(prediccion: np.ndarray, etiquetas: np.ndarray, clases: int) -> np.ndarray:
    matriz = np.zeros((clases, clases), dtype=np.int64)
    np.add.at(matriz, (etiquetas, prediccion), 1)
    return matriz


def evaluar(probabilidades: np.ndarray, etiquetas: np.ndarray, clases: int) -> ResultadoEvaluacion:
    if probabilidades.ndim != 2 or len(probabilidades) != len(etiquetas):
        raise ValueError("Las probabilidades deben ser (N, clases) con una etiqueta por fila.")
    if probabilidades.shape[1] != clases:
        raise ValueError(f"Las probabilidades tienen {probabilidades.shape[1]} clases y se esperaban {clases}.")
    prediccion = probabilidades.argmax(axis=1)
    aciertos = int((prediccion == etiquetas).sum())
    return ResultadoEvaluacion(
        aciertos=aciertos,
        total=len(etiquetas),
        top1=aciertos / len(etiquetas),
        f1_macro=metricas.f1_macro(probabilidades, etiquetas, clases),
        acierto_95=metricas.acierto_95(probabilidades, etiquetas),
        falsas_95=metricas.falsas_95(probabilidades, etiquetas),
        top1_por_clase=tuple(metricas.top1_por_clase(probabilidades, etiquetas, clases)),
        f1_por_clase=tuple(float(v) for v in metricas.f1_por_clase(probabilidades, etiquetas, clases)),
        confusion=confusion(prediccion, etiquetas, clases),
    )
