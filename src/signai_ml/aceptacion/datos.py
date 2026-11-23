"""La prueba de la aceptación (spec FR-001, FR-002; research R-04).

La única vía hacia los 1.800 clips de prueba es el contrato de lectura de TT-13
(`para_evaluacion`). Antes de leer una sola ventana se comprueba que el corpus y la partición
actuales son aquellos con los que se entrenó el candidato, y después que la prueba tiene exactamente
18 clips por clase (ADR-0011).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..particion.lectura import para_evaluacion
from ..ventanas import VentanasNoValidas, abrir_corpus
from . import CLIPS_PRUEBA, POR_CLASE
from .candidato import HUELLAS, Candidato
from .errores import ErrorAceptacion, HuellasDistintas, PruebaInesperada


@dataclass(frozen=True)
class DatosPrueba:
    x: np.ndarray
    y: np.ndarray
    clases: tuple[str, ...]
    glosas: tuple[str, ...]
    huellas: dict[str, str]


def _comprobar_huellas(actuales: dict[str, str], candidato: Candidato) -> None:
    distintas = [clave for clave in HUELLAS if actuales[clave] != candidato.huellas[clave]]
    if distintas:
        detalle = "; ".join(
            f"{clave}: candidato {candidato.huellas[clave][:12]}…, actual {actuales[clave][:12]}…" for clave in distintas
        )
        raise HuellasDistintas(
            f"El candidato no se entrenó con los datos actuales ({detalle}). Medir sobre otros datos rompería "
            "la comparabilidad (ADR-0011): restaura los datos versionados con `uv run dvc checkout`."
        )


def _comprobar_reparto(y: np.ndarray, clases: int, total: int, por_clase: int) -> None:
    if len(y) != total:
        raise PruebaInesperada(f"La prueba tiene {len(y)} clips y se esperaban {total} (ADR-0011).")
    recuento = np.bincount(y, minlength=clases)
    if len(recuento) != clases or np.any(recuento != por_clase):
        raise PruebaInesperada(
            f"La prueba no tiene {por_clase} clips por clase: los recuentos van de {recuento.min()} a "
            f"{recuento.max()} (ADR-0011)."
        )


def cargar_prueba(
    raiz: Path,
    candidato: Candidato,
    lector: Callable = para_evaluacion,
    total: int = CLIPS_PRUEBA,
    por_clase: int = POR_CLASE,
) -> DatosPrueba:
    raiz = Path(raiz)
    lado = lector(raiz)
    try:
        corpus = abrir_corpus(raiz)
    except VentanasNoValidas as error:
        raise ErrorAceptacion(str(error)) from error
    actuales = {**corpus.huellas, "particion.huella": lado.huella}
    _comprobar_huellas(actuales, candidato)
    _comprobar_reparto(corpus.etiquetas[lado.prueba], len(corpus.clases), total, por_clase)
    x, y = corpus.tomar(lado.prueba)
    return DatosPrueba(x=x, y=np.array(y), clases=corpus.clases, glosas=corpus.glosas, huellas=actuales)
