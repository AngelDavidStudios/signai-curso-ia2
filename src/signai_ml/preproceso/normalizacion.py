"""Normalización de la ventana (75, 96): implementación de referencia del contrato 2.0.0.

Especificación única, compartida con la app: `signai-architecture/contracts/normalizacion.md`
(N-1…N-8). Es la preparación con la que se midió y se validó KPoseFormer en el laboratorio
(`Corpus-LSEC/Scripts/09_preparar.py`, réplica en Swift `Tuberia.swift`). Cuatro pasos, cada uno en
su clase, que el `Normalizador` aplica en un orden fijo:

    RespaldoMuneca → Imputacion → NormalizacionHombros → Remuestreo(75)

El normalizador solo conoce la abstracción `PasoNormalizacion` (principio de inversión de
dependencias) y no deja configurar sus pasos desde fuera: la cadena del corpus no puede tener una
variante propia (constitución, principio II). Se calcula en `float64` y se redondea a `float32` al
final (N-7).
"""

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

PUNTOS = 48
FOTOGRAMAS = 75
ANCHO_MINIMO = 1e-6
AUSENTE = -1.0
HOMBRO_IZQ, HOMBRO_DER = 0, 1
MUNECA_POSE_IZQ, MUNECA_POSE_DER = 4, 5
MUNECA_MANO_IZQ, MUNECA_MANO_DER = 6, 27


@dataclass(frozen=True)
class Secuencia:
    """Estado entre pasos: coordenadas por punto y si cada una es un valor real."""

    valores: np.ndarray  # (T, 48, 2) float64
    presente: np.ndarray  # (T, 48) bool
    sin_normalizar: bool = False


@dataclass(frozen=True)
class VentanaNormalizada:
    valores: np.ndarray  # (75, 96) float32
    sin_normalizar: bool  # N-5.3: el tramo no tenía un ancho de hombros válido


class PasoNormalizacion(Protocol):
    def aplicar(self, secuencia: Secuencia) -> Secuencia: ...


def secuencia_desde_keypoints(keypoints: np.ndarray) -> Secuencia:
    """N-1: (T, 96) intercalado → (T, 48, 2). Presente = su `x` distinta de −1."""
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[1] != 2 * PUNTOS or kp.shape[0] < 1:
        raise ValueError(f"se esperaba (T ≥ 1, 96) y llegó {kp.shape}")
    valores = kp.reshape(kp.shape[0], PUNTOS, 2).copy()
    return Secuencia(valores, valores[:, :, 0] != AUSENTE)


class RespaldoMuneca:
    """N-3: la muñeca de pose ausente toma la de la mano del mismo lado (4 ← 6, 5 ← 27)."""

    def aplicar(self, secuencia: Secuencia) -> Secuencia:
        valores, presente = secuencia.valores.copy(), secuencia.presente.copy()
        for pose, mano in ((MUNECA_POSE_IZQ, MUNECA_MANO_IZQ), (MUNECA_POSE_DER, MUNECA_MANO_DER)):
            usa = ~presente[:, pose] & presente[:, mano]
            valores[usa, pose] = valores[usa, mano]
            presente[usa, pose] = True
        return replace(secuencia, valores=valores, presente=presente)


class Imputacion:
    """N-4: interpolación lineal punto a punto; un punto que no aparece nunca se ancla a la muñeca
    de pose de su lado, después al centro de hombros y después a (0,5; 0,5). En el orden k = 0…47."""

    def aplicar(self, secuencia: Secuencia) -> Secuencia:
        valores, presente = secuencia.valores.copy(), secuencia.presente
        t = np.arange(valores.shape[0])
        hombros = presente[:, HOMBRO_IZQ] & presente[:, HOMBRO_DER]
        centro = np.full((len(t), 2), np.nan)
        centro[hombros] = (valores[hombros, HOMBRO_IZQ] + valores[hombros, HOMBRO_DER]) / 2

        for k in range(PUNTOS):
            vistos = presente[:, k]
            if vistos.all():
                continue
            if not vistos.any():
                ancla = MUNECA_POSE_IZQ if k < MUNECA_MANO_DER else MUNECA_POSE_DER
                base = valores[:, ancla].copy()
                sin_ancla = base[:, 0] == AUSENTE
                base[sin_ancla] = centro[sin_ancla]
                base[np.isnan(base[:, 0])] = 0.5
                valores[:, k] = base
                continue
            for eje in range(2):
                valores[~vistos, k, eje] = np.interp(t[~vistos], t[vistos], valores[vistos, k, eje])
        return replace(secuencia, valores=valores, presente=np.ones_like(presente))


class NormalizacionHombros:
    """N-5: centro de hombros en cada fotograma y ancho de hombros MEDIANO del tramo."""

    def aplicar(self, secuencia: Secuencia) -> Secuencia:
        izq, der = secuencia.valores[:, HOMBRO_IZQ], secuencia.valores[:, HOMBRO_DER]
        ancho = float(np.median(np.hypot(izq[:, 0] - der[:, 0], izq[:, 1] - der[:, 1])))
        if not np.isfinite(ancho) or ancho < ANCHO_MINIMO:
            return replace(secuencia, sin_normalizar=True)
        centro = (izq + der) / 2
        return replace(secuencia, valores=(secuencia.valores - centro[:, None, :]) / ancho)


class Remuestreo:
    """N-6: `t_k = k·(T−1)/(n−1)` con interpolación lineal; con T = n es la identidad exacta."""

    def __init__(self, fotogramas: int = FOTOGRAMAS) -> None:
        self.fotogramas = fotogramas

    def aplicar(self, secuencia: Secuencia) -> Secuencia:
        origen = secuencia.valores.shape[0]
        if origen == self.fotogramas:
            return secuencia
        nuevo = np.linspace(0, 1, self.fotogramas)
        viejo = np.linspace(0, 1, origen) if origen > 1 else np.zeros(1)
        plano = secuencia.valores.reshape(origen, -1)
        valores = np.stack([np.interp(nuevo, viejo, plano[:, c]) for c in range(plano.shape[1])], axis=1)
        presente = np.repeat(secuencia.presente.all(axis=0)[None, :], self.fotogramas, axis=0)
        return replace(secuencia, valores=valores.reshape(self.fotogramas, PUNTOS, 2), presente=presente)


def aplicar_pasos(pasos: tuple[PasoNormalizacion, ...], keypoints: np.ndarray) -> VentanaNormalizada:
    """Aplica `pasos` en orden y compone la salida N-7. El normalizador fija los pasos."""
    secuencia = secuencia_desde_keypoints(keypoints)
    for paso in pasos:
        secuencia = paso.aplicar(secuencia)
    salida = secuencia.valores.reshape(secuencia.valores.shape[0], 2 * PUNTOS).astype(np.float32)
    return VentanaNormalizada(salida, secuencia.sin_normalizar)


class Normalizador:
    """La normalización del contrato: una sola operación que oculta sus pasos (Figura 49)."""

    PASOS: tuple[PasoNormalizacion, ...] = (
        RespaldoMuneca(), Imputacion(), NormalizacionHombros(), Remuestreo(FOTOGRAMAS)
    )

    def normalizar(self, keypoints: np.ndarray) -> VentanaNormalizada:
        ventana = aplicar_pasos(self.PASOS, keypoints)
        if ventana.valores.shape != (FOTOGRAMAS, 2 * PUNTOS) or not np.isfinite(ventana.valores).all():
            raise ValueError(f"ventana inválida: {ventana.valores.shape}")
        return ventana
