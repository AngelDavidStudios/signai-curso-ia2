"""Sorteo de la partición por grabación entera (TT-22, research R-01).

Dentro de cada clase, sus grabaciones se ordenan por `sha256("<semilla>|<clase>|<grabacion>")` y cada
una recibe su lado con `reglas.lados_por_posicion`; todos sus clips van a ese lado. La grabación se
identifica por el par `(clase, grabacion)`: el mismo nombre en dos clases son dos grabaciones. Solo
intervienen `clase`, `grabacion` y `clip_id` (spec FR-003). Determinista en cualquier versión de Python
o NumPy e independiente del orden de las filas del índice.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping

from . import Lado
from .reglas import lados_por_posicion


def clave_sorteo(semilla: int, clase: str, grabacion: str) -> str:
    return hashlib.sha256(f"{semilla}|{clase}|{grabacion}".encode()).hexdigest()


def lados_de_grabaciones(filas: Iterable[Mapping[str, str]], semilla: int) -> dict[tuple[str, str], Lado]:
    """El lado de cada grabación `(clase, grabacion)`."""
    por_clase: dict[str, set[str]] = defaultdict(set)
    for fila in filas:
        por_clase[fila["clase"]].add(fila["grabacion"])
    lados: dict[tuple[str, str], Lado] = {}
    for clase in sorted(por_clase):
        ordenadas = sorted(por_clase[clase], key=lambda g: (clave_sorteo(semilla, clase, g), g))
        for grabacion, lado in zip(ordenadas, lados_por_posicion(len(ordenadas), clase), strict=True):
            lados[(clase, grabacion)] = lado
    return lados


def sortear(filas: Iterable[Mapping[str, str]], semilla: int) -> dict[str, Lado]:
    """El lado de cada `clip_id`: el de su grabación."""
    filas = list(filas)
    lados = lados_de_grabaciones(filas, semilla)
    return {fila["clip_id"]: lados[(fila["clase"], fila["grabacion"])] for fila in filas}
