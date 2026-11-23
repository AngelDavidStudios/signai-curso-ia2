"""Verificación del resultado, sin escribir nada (SC-001…SC-003, SC-006; contracts/cli.md C-4).

Devuelve la lista de problemas; vacía significa que el resultado es correcto.
"""

import csv
import json
from collections.abc import Sequence

import numpy as np

from .almacen import Almacen, sha256_fichero
from .clips import ClipEtiquetado

CLAVES_VERSION = ("version_cadena", "version_extractor", "esquema", "normalizacion", "macos")


def _keypoints(clip: ClipEtiquetado, almacen: Almacen, versiones: dict[str, set]) -> list[str]:
    if not almacen.ruta_keypoints(clip).is_file():
        return [f"{clip.clip_id}: faltan sus keypoints"]
    problemas = []
    kp = almacen.leer_keypoints(clip)
    esperado = -(-clip.n_fotogramas // 2)
    if kp.shape != (esperado, 96):
        problemas.append(f"{clip.clip_id}: keypoints {kp.shape}; se esperaba ({esperado}, 96) por ⌈n/2⌉")
    if not np.isfinite(kp).all():
        problemas.append(f"{clip.clip_id}: keypoints con valores no finitos")
    pares = kp.reshape(kp.shape[0], -1, 2)
    if ((pares[..., 0] == -1) != (pares[..., 1] == -1)).any():
        problemas.append(f"{clip.clip_id}: un punto ausente a medias (−1 en una sola coordenada)")
    meta = almacen.leer_meta(clip)
    for clave in CLAVES_VERSION:
        versiones[clave].add(meta.get(clave))
    return problemas


def verificar(clips: Sequence[ClipEtiquetado], almacen: Almacen) -> list[str]:
    problemas: list[str] = []
    ruta = almacen.dir_ventanas / "ventanas.npy"
    if not ruta.is_file():
        return [f"no existe {ruta}: lanza `signai-preproceso procesar`"]

    versiones: dict[str, set] = {c: set() for c in CLAVES_VERSION}
    for clip in clips:
        problemas += _keypoints(clip, almacen, versiones)
    for clave, valores in versiones.items():
        if len(valores) > 1:
            problemas.append(f"versiones mezcladas en el corpus: {clave} = {sorted(map(str, valores))}")

    with (almacen.dir_ventanas / "indice.csv").open(encoding="utf-8") as f:
        indice = [fila["clip_id"] for fila in csv.DictReader(f)]
    esperados = [c.clip_id for c in clips]
    if indice != esperados:
        faltan = sorted(set(esperados) - set(indice))
        sobran = sorted(set(indice) - set(esperados))
        problemas.append(
            f"el índice no coincide con etiquetas.csv: faltan {faltan[:5]}, sobran {sobran[:5]}"
            + ("" if faltan or sobran else " (orden distinto)")
        )

    ventanas = np.load(ruta, allow_pickle=False)
    if ventanas.shape != (len(clips), 75, 96) or ventanas.dtype != np.float32:
        problemas.append(f"ventanas {ventanas.shape} {ventanas.dtype}; se esperaba ({len(clips)}, 75, 96) float32")
    if not np.isfinite(ventanas).all():
        problemas.append("ventanas con valores no finitos")
    manifiesto = json.loads((almacen.dir_ventanas / "manifiesto.json").read_text(encoding="utf-8"))
    if manifiesto.get("sha256_ventanas") != sha256_fichero(ruta):
        problemas.append("la huella de ventanas.npy no es la del manifiesto")
    return problemas
