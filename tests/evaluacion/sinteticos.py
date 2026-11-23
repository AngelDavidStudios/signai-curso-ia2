"""Candidato y prueba sintéticos de la suite `evaluacion` (TT-15): nada de datos de señantes.

Un corpus de 4 clases × 6 clips (el de `reconocedor.sinteticos`), con los 3 últimos clips de cada
clase en la prueba, y un `modelos/reconocedor/` con el formato de TT-14 (contrato `modelo.md`, M-2).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from reconocedor.sinteticos import escribir_corpus, ventanas_sinteticas, vocabulario_sintetico
from signai_ml.reconocedor.configuracion import Forma
from signai_ml.reconocedor.modelo import KPoseFormer, contar_parametros, guardar
from signai_ml.ventanas import huellas_corpus

N_CLASES = 4
CLIPS_POR_CLASE = 6
PRUEBA_POR_CLASE = 3
TOTAL_PRUEBA = N_CLASES * PRUEBA_POR_CLASE
HUELLA_PARTICION = "e" * 64
FORMA = Forma(capas=1, dimension=16, cabezas=2, intermedia=16)


@dataclass(frozen=True)
class LadoPruebaFalso:
    prueba: np.ndarray
    huella: str


def filas_prueba() -> np.ndarray:
    filas = [
        fila
        for clase in range(N_CLASES)
        for fila in range((clase + 1) * CLIPS_POR_CLASE - PRUEBA_POR_CLASE, (clase + 1) * CLIPS_POR_CLASE)
    ]
    return np.array(filas, dtype=np.int64)


def lector_prueba(_raiz: Path) -> LadoPruebaFalso:
    return LadoPruebaFalso(filas_prueba(), HUELLA_PARTICION)


def _entrenar(modelo: KPoseFormer, pasos: int = 40) -> KPoseFormer:
    """Unos pasos sobre el corpus sintético para que las predicciones sean claras (paridad sin empates)."""
    x, y = ventanas_sinteticas(N_CLASES, CLIPS_POR_CLASE)
    entrada, etiquetas = torch.from_numpy(x), torch.from_numpy(y)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
    modelo.train()
    for _ in range(pasos):
        optimizador.zero_grad()
        torch.nn.functional.cross_entropy(modelo(entrada), etiquetas).backward()
        optimizador.step()
    return modelo


def escribir_candidato(raiz: Path, semilla: int = 0, entrenado: bool = False, **cambios_ficha) -> Path:
    """`modelos/reconocedor/{modelo.pt, candidato.json}` como los deja TT-14, ligados al corpus de `raiz`."""
    destino = Path(raiz) / "modelos" / "reconocedor"
    destino.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(semilla)
    modelo = KPoseFormer(FORMA, clases=N_CLASES)
    if entrenado:
        _entrenar(modelo)
    modelo.eval()
    vocabulario = vocabulario_sintetico(N_CLASES)
    huellas = {**huellas_corpus(raiz), "particion.huella": HUELLA_PARTICION}
    corrida = {"barrido": "b01", "run_id": "c" * 32, "configuracion": "r01-base", "semilla": 216, "epoca": 3}
    guardar(
        destino / "modelo.pt", modelo, FORMA,
        [f["clase"] for f in vocabulario], [f["glosa"] for f in vocabulario], corrida, huellas,
    )
    ficha = {
        "barrido": "b01",
        "run_id": corrida["run_id"],
        "configuracion": corrida["configuracion"],
        "semilla": corrida["semilla"],
        "ronda": 1,
        "metricas": {"top1": 1.0, "f1_macro": 1.0, "acierto_95": 0.9, "falsas_95": 0.0, "perdida": 0.1, "epoca": 3},
        "parametros": contar_parametros(modelo),
        "huellas": huellas,
        "commit": "d" * 40,
        "fecha": "2026-11-15T09:59:31-05:00",
        "estado": "pendiente de aceptación (TT-15)",
    } | cambios_ficha
    (destino / "candidato.json").write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destino


def escribir_raiz_prueba(raiz: Path, entrenado: bool = False) -> Path:
    """Corpus sintético y candidato ligados a él, bajo `raiz`."""
    escribir_corpus(raiz, N_CLASES, CLIPS_POR_CLASE)
    escribir_candidato(raiz, entrenado=entrenado)
    return raiz
