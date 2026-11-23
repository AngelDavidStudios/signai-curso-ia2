"""Aumento de datos sobre ventanas ya normalizadas `(B, 75, 96)` (spec FR-006; research R-08).

Funciones puras con un `torch.Generator` sembrado por la semilla de la corrida. Ninguna convierte un
punto ausente (`-1`) en presente: las espaciales solo tocan los presentes y la ocultación solo pasa
presentes a ausentes; las temporales toman el fotograma más cercano, sin interpolar, para no mezclar
`-1` con coordenadas (el mismo motivo por el que la normalización imputa antes de remuestrear, N-2).
Solo se aplica a los lotes de entrenamiento.
"""

import torch

from . import FOTOGRAMAS
from .configuracion import Aumento

AUSENTE = -1.0
_CENTRO = (FOTOGRAMAS - 1) / 2


def _aleatorio(generador: torch.Generator, *forma: int) -> torch.Tensor:
    return torch.rand(*forma, generator=generador)


def _puntos(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], x.shape[1], -1, 2)


def _desplazamiento(x: torch.Tensor, intensidad: float, generador: torch.Generator) -> torch.Tensor:
    puntos = _puntos(x)
    delta = (_aleatorio(generador, x.shape[0], 1, 1, 2) * 2 - 1) * intensidad
    return torch.where(puntos != AUSENTE, puntos + delta, puntos).reshape(x.shape)


def _escala(x: torch.Tensor, intensidad: float, generador: torch.Generator) -> torch.Tensor:
    puntos = _puntos(x)
    factor = 1 + (_aleatorio(generador, x.shape[0], 1, 1, 1) * 2 - 1) * intensidad
    return torch.where(puntos != AUSENTE, puntos * factor, puntos).reshape(x.shape)


def _rotacion(x: torch.Tensor, intensidad: float, generador: torch.Generator) -> torch.Tensor:
    """Gira cada clip un ángulo uniforme en `[−intensidad, +intensidad]` radianes alrededor del origen, que
    en la ventana normalizada es el centro de los hombros (N-5). Solo los puntos presentes (TT-22, R-07)."""
    puntos = _puntos(x)
    angulo = (_aleatorio(generador, x.shape[0], 1, 1) * 2 - 1) * intensidad
    coseno, seno = torch.cos(angulo), torch.sin(angulo)
    px, py = puntos[..., 0], puntos[..., 1]
    girados = torch.stack((px * coseno - py * seno, px * seno + py * coseno), dim=-1)
    presentes = (puntos != AUSENTE).all(dim=-1, keepdim=True)
    return torch.where(presentes, girados, puntos).reshape(x.shape)


def _ocultacion(x: torch.Tensor, intensidad: float, generador: torch.Generator) -> torch.Tensor:
    puntos = _puntos(x)
    ocultos = (_aleatorio(generador, *puntos.shape[:3]) < intensidad).unsqueeze(-1)
    return torch.where(ocultos, torch.full_like(puntos, AUSENTE), puntos).reshape(x.shape)


def _reindexar(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x[torch.arange(x.shape[0]).unsqueeze(1), indices.long()]


def _velocidad(x: torch.Tensor, intensidad: float, generador: torch.Generator) -> torch.Tensor:
    factor = 1 + (_aleatorio(generador, x.shape[0], 1) * 2 - 1) * intensidad
    t = torch.arange(FOTOGRAMAS, dtype=torch.float32).unsqueeze(0)
    return _reindexar(x, torch.round(_CENTRO + (t - _CENTRO) * factor).clamp(0, FOTOGRAMAS - 1))


def _recorte_temporal(x: torch.Tensor, intensidad: float, generador: torch.Generator) -> torch.Tensor:
    fraccion = intensidad + _aleatorio(generador, x.shape[0], 1) * (1 - intensidad)
    largo = torch.round(fraccion * FOTOGRAMAS).clamp(2, FOTOGRAMAS)
    inicio = torch.floor(_aleatorio(generador, x.shape[0], 1) * (FOTOGRAMAS - largo + 1))
    t = torch.arange(FOTOGRAMAS, dtype=torch.float32).unsqueeze(0)
    return _reindexar(x, inicio + torch.round(t * (largo - 1) / (FOTOGRAMAS - 1)))


_TRANSFORMACIONES = {
    "desplazamiento": _desplazamiento,
    "escala": _escala,
    "rotacion": _rotacion,
    "ocultacion": _ocultacion,
    "velocidad": _velocidad,
    "recorte_temporal": _recorte_temporal,
}


def aplicar(x: torch.Tensor, aumento: tuple[tuple[str, Aumento], ...], generador: torch.Generator) -> torch.Tensor:
    """Aplica cada tipo declarado, por clip, con su probabilidad e intensidad."""
    if not aumento:
        return x
    for tipo, parametros in aumento:
        elegidos = (_aleatorio(generador, x.shape[0]) < parametros.probabilidad).view(-1, 1, 1)
        x = torch.where(elegidos, _TRANSFORMACIONES[tipo](x, parametros.intensidad, generador), x)
    return x
