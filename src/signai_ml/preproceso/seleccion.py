"""De lo que ve el detector a los 48 puntos del esquema (E-2…E-6; FR-006…FR-008; research R-05).

Réplica de `KeypointSelector` de la app (`signai-ios/…/Captura/Dominio/KeypointSelector.swift`),
probada con sus mismos casos: elige a la persona, reparte las manos, invierte el eje vertical y deja a
−1 lo que no llega. No recibe ninguna confianza, así que no puede filtrar por ella. No recorta las
coordenadas a [0, 1], igual que la app. Las coordenadas se tratan en `float32`, como en la app.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .deteccion import ORDEN_MANO, ORDEN_POSE, CuerpoDetectado, Deteccion, FotogramaDetectado, ManoDetectada, PuntoDetector

INICIO_BLOQUE = {"left": 6, "right": 27}
LADOS = ("left", "right")
MUNECA_POSE = {"left": "leftWrist", "right": "rightWrist"}


def _area(puntos: Sequence[PuntoDetector]) -> float:
    if not puntos:
        return 0.0
    xs = [p.x for p in puntos]
    ys = [p.y for p in puntos]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _persona(cuerpos: Sequence[CuerpoDetectado]) -> CuerpoDetectado | None:
    """E-5: el del rectángulo de puntos más grande; a igualdad, el primero. Sin puntos no cuenta."""
    mejor: tuple[CuerpoDetectado, float] | None = None
    for cuerpo in cuerpos:
        if not cuerpo.puntos:
            continue
        area = _area(list(cuerpo.puntos.values()))
        if mejor is None or area > mejor[1]:
            mejor = (cuerpo, area)
    return None if mejor is None else mejor[0]


def _posicion(mano: ManoDetectada) -> PuntoDetector:
    """E-6.1: su muñeca o, si falta, el centro de sus puntos."""
    if "wrist" in mano.puntos:
        return mano.puntos["wrist"]
    puntos = list(mano.puntos.values())
    return PuntoDetector(sum(p.x for p in puntos) / len(puntos), sum(p.y for p in puntos) / len(puntos))


def _por_cercania(manos: Sequence[ManoDetectada], munecas: Mapping[str, PuntoDetector]) -> dict[str, ManoDetectada]:
    """E-6.3: la pareja mano–muñeca más cercana primero (ausente = ∞); desempate: orden de llegada,
    después izquierda antes que derecha."""
    candidatos = []
    for i, mano in enumerate(manos):
        pos = _posicion(mano)
        for orden_lado, lado in enumerate(LADOS):
            m = munecas.get(lado)
            distancia = math.inf if m is None else math.hypot(pos.x - m.x, pos.y - m.y)
            candidatos.append((distancia, i, orden_lado, lado))
    asignadas: dict[str, ManoDetectada] = {}
    colocadas: set[int] = set()
    for _, i, _, lado in sorted(candidatos):
        if lado not in asignadas and i not in colocadas:
            asignadas[lado] = manos[i]
            colocadas.add(i)
    return asignadas


def _por_lado_conservando_la_mayor(manos: Sequence[ManoDetectada]) -> dict[str, ManoDetectada]:
    """E-6.4: sin muñecas de pose, por lado; entre dos iguales, la que ocupa más; sin lado, fuera."""
    asignadas: dict[str, ManoDetectada] = {}
    for mano in manos:
        if mano.lado is None:
            continue
        actual = asignadas.get(mano.lado)
        if actual is None or _area(list(mano.puntos.values())) > _area(list(actual.puntos.values())):
            asignadas[mano.lado] = mano
    return asignadas


def _repartir(manos: Sequence[ManoDetectada], persona: CuerpoDetectado | None) -> dict[str, ManoDetectada]:
    manos = [m for m in manos if m.puntos]
    lados = [m.lado for m in manos if m.lado is not None]
    if len(lados) == len(manos) and len(set(lados)) == len(lados):  # E-6.2
        return {m.lado: m for m in manos}  # type: ignore[misc]
    munecas = {
        lado: persona.puntos[nombre]
        for lado, nombre in MUNECA_POSE.items()
        if persona is not None and nombre in persona.puntos
    }
    return _por_cercania(manos, munecas) if munecas else _por_lado_conservando_la_mayor(manos)


def _escribir(fila: np.ndarray, indice: int, p: PuntoDetector) -> None:
    """E-3: x tal cual; y = 1 − y, en float32 como la app."""
    fila[2 * indice] = np.float32(p.x)
    fila[2 * indice + 1] = np.float32(1) - np.float32(p.y)


class Selector:
    def seleccionar(self, fotograma: FotogramaDetectado) -> np.ndarray:
        fila = np.full(96, -1, dtype=np.float32)
        if fotograma.fallido:  # E-4: se registra con los 48 a −1, no se descarta
            return fila
        persona = _persona(fotograma.cuerpos)
        if persona is not None:
            for indice, nombre in enumerate(ORDEN_POSE):
                if nombre in persona.puntos:
                    _escribir(fila, indice, persona.puntos[nombre])
        for lado, mano in _repartir(fotograma.manos, persona).items():
            for k, nombre in enumerate(ORDEN_MANO):
                if nombre in mano.puntos:
                    _escribir(fila, INICIO_BLOQUE[lado] + k, mano.puntos[nombre])
        return fila


def seleccionar_clip(deteccion: Deteccion, selector: Selector = Selector()) -> np.ndarray:
    """(T, 96) float32: una fila por fotograma decimado."""
    if not deteccion.fotogramas:
        return np.empty((0, 96), dtype=np.float32)
    return np.stack([selector.seleccionar(f) for f in deteccion.fotogramas])
