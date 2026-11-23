"""Lote de validación y selección de los vídeos de revisión (FR-018, FR-020; research R-12)."""

from pathlib import Path

from signai_ml.preproceso.clips import ClipEtiquetado
from signai_ml.preproceso.informe import InformeClip
from signai_ml.preproceso.lote import SEMILLA_LOTE, elegir_lote, elegir_revision, guardar_lote, leer_lote


def clip(cid: str) -> ClipEtiquetado:
    clase, grab, rep = cid.split("__")
    return ClipEtiquetado(cid, clase, int(clase.split("-")[0]), "x", grab, int(rep[3:]), 10, "a" * 32, Path("x"))


CLIPS = [clip(f"{c}__{g}__rep{r:02d}") for c, g in [("1-Yo", "IMG_1"), ("1-Yo", "IMG_2"), ("2-Tu", "IMG_3")] for r in range(1, 31)]


def informe(cid: str, anomalias: list[str]) -> InformeClip:
    return InformeClip(cid, cid.split("__")[0], cid.split("__")[1], 10, 5, 0.0, 0.0, 0.0, 0, False, anomalias)


def test_exactamente_un_clip_por_grabacion_con_la_semilla_213() -> None:
    lote = elegir_lote(CLIPS)
    assert SEMILLA_LOTE == 213
    assert sorted(c.grabacion for c in lote) == ["IMG_1", "IMG_2", "IMG_3"]
    assert elegir_lote(CLIPS) == lote
    assert elegir_lote(list(reversed(CLIPS))) == lote  # no depende del orden de entrada


def test_una_grabacion_compartida_por_dos_clases_cuenta_una_vez_por_clase() -> None:
    # En el corpus, IMG_1239 e IMG_1387 están en 28-Feliz y en 81-Alegre: el lote es por clase–grabación.
    compartida = CLIPS + [clip(f"3-Ustedes__IMG_1__rep{r:02d}") for r in range(1, 31)]
    lote = elegir_lote(compartida)
    assert sorted((c.clase, c.grabacion) for c in lote) == [
        ("1-Yo", "IMG_1"), ("1-Yo", "IMG_2"), ("2-Tu", "IMG_3"), ("3-Ustedes", "IMG_1"),
    ]


def test_otra_semilla_puede_dar_otro_lote() -> None:
    assert any(elegir_lote(CLIPS, semilla=s) != elegir_lote(CLIPS) for s in range(1, 20))


def test_revision_incluye_todos_los_anomalos_y_veinte_al_azar() -> None:
    informes = [informe(c.clip_id, []) for c in CLIPS]
    informes[3] = informe(CLIPS[3].clip_id, ["pose_ausente>20%"])
    informes[5] = informe(CLIPS[5].clip_id, ["mano_nunca_presente"])  # informativa: también tiene vídeo

    revision = elegir_revision(informes)

    motivos = {r["clip_id"]: r["motivo"] for r in revision}
    assert motivos[CLIPS[3].clip_id] == "anomalía: pose_ausente>20%"
    assert motivos[CLIPS[5].clip_id] == "anomalía (informativa): mano_nunca_presente"
    assert sum(1 for m in motivos.values() if m == "azar") == 20
    assert len(revision) == 22
    assert elegir_revision(informes) == revision


def test_una_anomalia_real_junto_a_una_informativa_no_es_informativa() -> None:
    informes = [informe(c.clip_id, []) for c in CLIPS]
    informes[2] = informe(CLIPS[2].clip_id, ["manos_ausentes>50%", "mano_nunca_presente"])

    motivos = {r["clip_id"]: r["motivo"] for r in elegir_revision(informes)}

    assert motivos[CLIPS[2].clip_id] == "anomalía: manos_ausentes>50%, mano_nunca_presente"


def test_guarda_y_lee_el_lote(tmp_path: Path) -> None:
    lote = elegir_lote(CLIPS)
    ruta = tmp_path / "lote.json"
    guardar_lote(ruta, lote, [{"clip_id": lote[0].clip_id, "motivo": "azar", "video": "v.mp4"}])
    datos = leer_lote(ruta)
    assert datos["semilla"] == 213 and datos["clips"] == [c.clip_id for c in lote]
    assert datos["revision"][0]["video"] == "v.mp4" and "semilla_revision" in datos
