"""Informe por clip y del lote (FR-019; research R-12)."""

import json
from pathlib import Path

import numpy as np

from signai_ml.preproceso.clips import ClipEtiquetado
from signai_ml.preproceso.informe import escribir_informe, informe_clip, informe_lote


def clip(n: int = 20) -> ClipEtiquetado:
    return ClipEtiquetado("1-Yo__IMG_1__rep01", "1-Yo", 1, "yo", "IMG_1", 1, n, "a" * 32, Path("x"))


def kp(t: int = 10) -> np.ndarray:
    m = np.full((t, 96), 0.5, dtype=np.float32)
    return m


def test_clip_completo_sin_anomalias() -> None:
    i = informe_clip(clip(), kp(), fallidos=0, sin_normalizar=False)
    assert (i.n_60, i.n_30) == (20, 10)
    assert (i.ausente_pose, i.ausente_mano_izq, i.ausente_mano_der) == (0.0, 0.0, 0.0)
    assert i.anomalias == []


def test_cada_anomalia_con_su_umbral() -> None:
    m = kp()
    m[:3, :12] = -1  # pose ausente en 30 %
    m[:6, 12:] = -1  # dos manos ausentes en 60 %
    m[:, 12:54] = -1  # mano izquierda nunca presente
    i = informe_clip(clip(), m, fallidos=2, sin_normalizar=True)
    assert i.ausente_pose == 0.3 and i.ausente_mano_izq == 1.0 and i.ausente_mano_der == 0.6
    assert i.anomalias == [
        "pose_ausente>20%", "manos_ausentes>50%", "mano_nunca_presente", "fallos_detector", "sin_normalizar",
    ]


def test_un_bloque_cuenta_como_ausente_solo_si_faltan_todos_sus_puntos() -> None:
    m = kp()
    m[:, 0:2] = -1  # falta un hombro: la pose sigue presente
    assert informe_clip(clip(), m, 0, False).ausente_pose == 0.0


def test_informe_del_lote_con_proyeccion_y_ficheros(tmp_path: Path) -> None:
    informes = [informe_clip(clip(), kp(), 0, False)]
    lote = informe_lote(informes, segundos=2.0, fotogramas=100, fotogramas_corpus=540_000)
    assert lote["clips"] == 1 and lote["fotogramas_por_segundo"] == 50.0
    assert lote["proyeccion_horas"] == 3.0
    sha = escribir_informe(tmp_path, lote)
    datos = json.loads((tmp_path / "informe.json").read_text())
    assert datos["por_clip"][0]["clip_id"] == "1-Yo__IMG_1__rep01" and len(sha) == 64
    assert "Proyección" in (tmp_path / "informe.md").read_text()
