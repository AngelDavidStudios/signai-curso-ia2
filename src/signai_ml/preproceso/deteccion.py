"""Lo que ve el detector, sin interpretar (contracts/extractor.md X-4, X-5; research R-01, R-04).

`keypoints-vision` (Swift + Apple Vision) escribe, por fotograma decimado, los cuerpos y las manos
con sus articulaciones en coordenadas de Vision (origen abajo), el lado de cada mano y **ninguna
confianza**. Este módulo lo lee, comprueba que corresponde al clip y a la versión vigente, y envuelve
el binario. Elegir persona y manos es de `seleccion`.
"""

import gzip
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from . import VERSION_EXTRACTOR
from .clips import ClipEtiquetado
from .decimacion import Decimacion
from .errores import ErrorExtraccion, FaltaHerramienta

# Orden de E-2: las 6 articulaciones de pose y las 21 de cada mano, con sus nombres de Vision.
ORDEN_POSE = ("leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist")
ORDEN_MANO = (
    "wrist",
    "thumbCMC", "thumbMP", "thumbIP", "thumbTip",
    "indexMCP", "indexPIP", "indexDIP", "indexTip",
    "middleMCP", "middlePIP", "middleDIP", "middleTip",
    "ringMCP", "ringPIP", "ringDIP", "ringTip",
    "littleMCP", "littlePIP", "littleDIP", "littleTip",
)
DETECTOR = {"pose_revision": 2, "manos_revision": 1, "max_manos": 2}  # E-1
LADOS = ("left", "right")

_RAIZ_REPO = Path(__file__).resolve().parents[3]
BINARIO_POR_DEFECTO = _RAIZ_REPO / "herramientas" / "keypoints-vision" / ".build" / "release" / "keypoints-vision"


class PuntoDetector(NamedTuple):
    """Coordenada de Vision: normalizada a la imagen, **origen abajo a la izquierda**."""

    x: float
    y: float


@dataclass(frozen=True)
class CuerpoDetectado:
    puntos: Mapping[str, PuntoDetector]


@dataclass(frozen=True)
class ManoDetectada:
    lado: str | None
    puntos: Mapping[str, PuntoDetector]


@dataclass(frozen=True)
class FotogramaDetectado:
    origen: int  # índice en el clip a 60 fps
    fallido: bool
    cuerpos: tuple[CuerpoDetectado, ...]
    manos: tuple[ManoDetectada, ...]


@dataclass(frozen=True)
class Deteccion:
    version_extractor: str
    macos: str
    md5: str
    fotogramas_decodificados: int
    cada: int
    fotogramas: tuple[FotogramaDetectado, ...]


def _puntos(clip_id: str, crudo: Mapping, validos: tuple[str, ...]) -> dict[str, PuntoDetector]:
    puntos = {}
    for nombre, (x, y) in crudo.items():
        if nombre not in validos:
            raise ErrorExtraccion(clip_id, f"articulación desconocida en la salida del extractor: {nombre}")
        puntos[nombre] = PuntoDetector(float(x), float(y))
    return puntos


def _fotograma(clip_id: str, crudo: Mapping) -> FotogramaDetectado:
    cuerpos = tuple(CuerpoDetectado(_puntos(clip_id, c["puntos"], ORDEN_POSE)) for c in crudo["cuerpos"])
    manos = []
    for m in crudo["manos"]:
        lado = m["lado"]
        if lado is not None and lado not in LADOS:
            raise ErrorExtraccion(clip_id, f"lado de mano desconocido: {lado}")
        manos.append(ManoDetectada(lado, _puntos(clip_id, m["puntos"], ORDEN_MANO)))
    return FotogramaDetectado(int(crudo["origen"]), bool(crudo["fallido"]), cuerpos, tuple(manos))


def leer_deteccion(ruta: Path, clip: ClipEtiquetado, decimacion: Decimacion = Decimacion()) -> Deteccion:
    """Lee la salida X-4 y comprueba X-5: versión, revisiones, clip, fotogramas y decimación."""
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        datos = json.load(f)
    cid = clip.clip_id
    if datos.get("version_extractor") != VERSION_EXTRACTOR:
        raise ErrorExtraccion(cid, f"versión del extractor {datos.get('version_extractor')}; la vigente es {VERSION_EXTRACTOR}")
    if datos.get("detector") != DETECTOR:
        raise ErrorExtraccion(cid, f"revisiones del detector {datos.get('detector')}; E-1 fija {DETECTOR}")
    if datos.get("md5") != clip.md5:
        raise ErrorExtraccion(cid, f"md5 {datos.get('md5')}; etiquetas.csv dice {clip.md5}")
    if datos.get("fotogramas_decodificados") != clip.n_fotogramas:
        raise ErrorExtraccion(
            cid, f"fotogramas decodificados {datos.get('fotogramas_decodificados')}; etiquetas.csv dice {clip.n_fotogramas}"
        )
    if datos.get("cada") != decimacion.cada:
        raise ErrorExtraccion(cid, f"cada = {datos.get('cada')}; la decimación usa {decimacion.cada}")
    fotogramas = tuple(_fotograma(cid, f) for f in datos["fotogramas"])
    if [f.origen for f in fotogramas] != list(decimacion.indices(clip.n_fotogramas)):
        raise ErrorExtraccion(cid, "los índices de origen no son los de la decimación (pares desde 0)")
    return Deteccion(
        datos["version_extractor"], datos["macos"], datos["md5"], datos["fotogramas_decodificados"],
        datos["cada"], fotogramas,
    )


class ExtractorKeypoints:
    """Envoltorio del binario Swift (contrato X-1)."""

    def __init__(self, binario: Path = BINARIO_POR_DEFECTO) -> None:
        self.binario = Path(binario)

    def extraer(self, video: Path, destino: Path, cada: int, volcar: Path | None = None) -> None:
        if not self.binario.is_file():
            raise FaltaHerramienta(
                "keypoints-vision",
                "Compílalo con `swift build -c release --package-path herramientas/keypoints-vision`.",
            )
        orden = [str(self.binario), str(video), str(destino), "--cada", str(cada)]
        if volcar is not None:
            orden += ["--volcar", str(volcar)]
        resultado = subprocess.run(orden, capture_output=True, text=True)
        if resultado.returncode != 0:
            raise ErrorExtraccion(Path(video).stem, resultado.stderr.strip() or f"código {resultado.returncode}")
