"""Lote de validación y vídeos de revisión (FR-018, FR-020; research R-12).

Un clip por grabación de cada clase (par clase–grabación: 300), elegido al azar con una semilla
registrada sobre sus clips ordenados por `clip_id`: cubre todas las señas y las tres grabaciones de
cada una. Es por par y no por grabación porque dos grabaciones (IMG_1239, IMG_1387) están en dos
clases a la vez (28-Feliz y 81-Alegre).
"""

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from .almacen import escribir_texto
from .clips import ClipEtiquetado
from .informe import InformeClip

SEMILLA_LOTE = 213
SEMILLA_REVISION = 2130
REVISION_AL_AZAR = 20
SOLO_INFORMATIVAS = {"mano_nunca_presente"}


def elegir_lote(clips: Sequence[ClipEtiquetado], semilla: int = SEMILLA_LOTE) -> tuple[ClipEtiquetado, ...]:
    por_grabacion: dict[tuple[str, str], list[ClipEtiquetado]] = defaultdict(list)
    for clip in clips:
        por_grabacion[(clip.clase, clip.grabacion)].append(clip)
    rng = random.Random(semilla)
    elegidos = [rng.choice(sorted(g, key=lambda c: c.clip_id)) for _, g in sorted(por_grabacion.items())]
    return tuple(sorted(elegidos, key=lambda c: c.clip_id))


def _motivo(anomalias: Sequence[str]) -> str:
    informativa = " (informativa)" if set(anomalias) <= SOLO_INFORMATIVAS else ""
    return f"anomalía{informativa}: " + ", ".join(anomalias)


def elegir_revision(
    informes: Sequence[InformeClip], semilla: int = SEMILLA_REVISION, al_azar: int = REVISION_AL_AZAR
) -> list[dict[str, str]]:
    """Todos los clips con alguna anomalía, también las informativas (FR-020), y `al_azar` del resto."""
    anomalos = [i for i in informes if i.anomalias]
    resto = sorted(i.clip_id for i in informes if not i.anomalias)
    azar = random.Random(semilla).sample(resto, min(al_azar, len(resto)))
    revision = [{"clip_id": i.clip_id, "motivo": _motivo(i.anomalias)} for i in anomalos]
    revision += [{"clip_id": c, "motivo": "azar"} for c in sorted(azar)]
    return sorted(revision, key=lambda r: r["clip_id"])


def guardar_lote(ruta: Path, lote: Sequence[ClipEtiquetado], revision: Sequence[Mapping]) -> None:
    datos = {
        "semilla": SEMILLA_LOTE, "clips": [c.clip_id for c in lote],
        "semilla_revision": SEMILLA_REVISION, "revision": list(revision),
    }
    escribir_texto(ruta, json.dumps(datos, ensure_ascii=False, indent=2))


def leer_lote(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))
