"""Informe por clip y del lote (FR-019; research R-12).

Las anomalías señalan, no descartan: dirigen la mirada del responsable en la revisión del lote.
Un bloque (pose, mano izquierda, mano derecha) cuenta como ausente en un fotograma si faltan todos
sus puntos.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .almacen import escribir_texto, sha256_fichero
from .clips import ClipEtiquetado

BLOQUES = {"pose": slice(0, 12), "mano_izq": slice(12, 54), "mano_der": slice(54, 96)}
UMBRAL_POSE = 0.20
UMBRAL_MANOS = 0.50


@dataclass(frozen=True)
class InformeClip:
    clip_id: str
    clase: str
    grabacion: str
    n_60: int
    n_30: int
    ausente_pose: float
    ausente_mano_izq: float
    ausente_mano_der: float
    fallidos: int
    sin_normalizar: bool
    anomalias: list[str] = field(default_factory=list)


def _ausente(keypoints: np.ndarray, bloque: str) -> np.ndarray:
    return (keypoints[:, BLOQUES[bloque]] == -1).all(axis=1)


def informe_clip(clip: ClipEtiquetado, keypoints: np.ndarray, fallidos: int, sin_normalizar: bool) -> InformeClip:
    pose, izq, der = (_ausente(keypoints, b) for b in ("pose", "mano_izq", "mano_der"))
    anomalias = []
    if pose.mean() > UMBRAL_POSE:
        anomalias.append("pose_ausente>20%")
    if (izq & der).mean() > UMBRAL_MANOS:
        anomalias.append("manos_ausentes>50%")
    if izq.all() or der.all():
        anomalias.append("mano_nunca_presente")
    if fallidos:
        anomalias.append("fallos_detector")
    if sin_normalizar:
        anomalias.append("sin_normalizar")
    return InformeClip(
        clip.clip_id, clip.clase, clip.grabacion, clip.n_fotogramas, int(keypoints.shape[0]),
        round(float(pose.mean()), 4), round(float(izq.mean()), 4), round(float(der.mean()), 4),
        int(fallidos), bool(sin_normalizar), anomalias,
    )


def informe_lote(
    informes: Sequence[InformeClip], segundos: float, fotogramas: int, fotogramas_corpus: int
) -> dict:
    fps = fotogramas / segundos if segundos > 0 else 0.0
    conteo: dict[str, int] = {}
    for i in informes:
        for a in i.anomalias:
            conteo[a] = conteo.get(a, 0) + 1
    return {
        "clips": len(informes),
        "fotogramas_extraidos": fotogramas,
        "segundos": round(segundos, 1),
        "fotogramas_por_segundo": round(fps, 2),
        "fotogramas_corpus": fotogramas_corpus,
        "proyeccion_horas": round(fotogramas_corpus / fps / 3600, 2) if fps else None,
        "anomalias": dict(sorted(conteo.items())),
        "por_clip": [asdict(i) for i in sorted(informes, key=lambda i: i.clip_id)],
    }


def _markdown(lote: dict) -> str:
    lineas = [
        "# Informe del lote de validación (TT-12)", "",
        f"- Clips: {lote['clips']}",
        f"- Fotogramas extraídos: {lote['fotogramas_extraidos']} en {lote['segundos']} s "
        f"({lote['fotogramas_por_segundo']} fotogramas/s)",
        f"- Proyección del corpus completo ({lote['fotogramas_corpus']} fotogramas): "
        f"{lote['proyeccion_horas']} h",
        "", "## Anomalías (señalan, no descartan)", "",
    ]
    lineas += [f"- `{a}`: {n} clip(s)" for a, n in lote["anomalias"].items()] or ["- ninguna"]
    lineas += [
        "", "## Por clip", "",
        "| Clip | 60 fps | 30 fps | Pose ausente | Mano izq. ausente | Mano der. ausente | Fallidos | Anomalías |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in lote["por_clip"]:
        lineas.append(
            f"| `{c['clip_id']}` | {c['n_60']} | {c['n_30']} | {c['ausente_pose']:.0%} | "
            f"{c['ausente_mano_izq']:.0%} | {c['ausente_mano_der']:.0%} | {c['fallidos']} | "
            f"{', '.join(c['anomalias']) or '—'} |"
        )
    return "\n".join(lineas) + "\n"


def escribir_informe(directorio: Path, lote: dict) -> str:
    """Escribe `informe.json` e `informe.md`; devuelve el SHA-256 de `informe.json`."""
    ruta = directorio / "informe.json"
    escribir_texto(ruta, json.dumps(lote, ensure_ascii=False, indent=2, sort_keys=True))
    escribir_texto(directorio / "informe.md", _markdown(lote))
    return sha256_fichero(ruta)
