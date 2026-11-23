"""Almacén de resultados: rutas, escritura atómica, vigencia, índice y manifiesto (research R-09, R-10)."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from signai_ml.preproceso.almacen import Almacen, Versiones, escribir_atomico, sha256_fichero
from signai_ml.preproceso.clips import ClipEtiquetado

V = Versiones("signai-ml/preproceso/1", "signai-ml/keypoints/1", "1.0.0", "1.0.0", "27.2.0")


def clip(cid: str = "1-Yo__IMG_1008__rep01", md5: str = "a" * 32, n: int = 10) -> ClipEtiquetado:
    clase = cid.split("__")[0]
    return ClipEtiquetado(cid, clase, int(clase.split("-")[0]), "yo", cid.split("__")[1], 1, n, md5, Path("x"))


def test_rutas_de_keypoints_y_ventanas(tmp_path: Path) -> None:
    a = Almacen(tmp_path / "datos")
    assert a.ruta_keypoints(clip()) == tmp_path / "datos" / "keypoints" / "1-Yo" / "1-Yo__IMG_1008__rep01.npy"
    assert a.dir_ventanas == tmp_path / "datos" / "ventanas"


def test_guarda_y_lee_keypoints_con_su_meta(tmp_path: Path) -> None:
    a = Almacen(tmp_path)
    kp = np.full((5, 96), -1, dtype=np.float32)
    a.guardar_keypoints(clip(), kp, V, fallidos=2)
    assert np.array_equal(a.leer_keypoints(clip()), kp)
    meta = a.leer_meta(clip())
    assert meta["md5_clip"] == "a" * 32 and meta["fallidos"] == 2 and meta["n_fotogramas_30"] == 5
    assert meta["macos"] == "27.2.0" and meta["version_cadena"] == V.version_cadena


@pytest.mark.parametrize(
    "cambio",
    [
        {"version_cadena": "signai-ml/preproceso/2"},
        {"version_extractor": "signai-ml/keypoints/2"},
        {"esquema": "1.1.0"},
        {"normalizacion": "1.1.0"},
        {"macos": "27.3.0"},
    ],
)
def test_deja_de_ser_vigente_si_cambia_una_version(tmp_path: Path, cambio: dict) -> None:
    a = Almacen(tmp_path)
    a.guardar_keypoints(clip(), np.zeros((5, 96), dtype=np.float32), V)
    assert a.vigente(clip(), V)
    assert not a.vigente(clip(), replace(V, **cambio))


def test_deja_de_ser_vigente_si_cambia_el_clip_o_no_hay_resultado(tmp_path: Path) -> None:
    a = Almacen(tmp_path)
    assert not a.vigente(clip(), V)
    a.guardar_keypoints(clip(), np.zeros((5, 96), dtype=np.float32), V)
    assert not a.vigente(clip(md5="b" * 32), V)


def test_la_escritura_atomica_no_deja_nada_si_falla(tmp_path: Path) -> None:
    destino = tmp_path / "x.npy"

    def falla(temporal: Path) -> None:
        temporal.write_bytes(b"a medias")
        raise RuntimeError("corte")

    with pytest.raises(RuntimeError):
        escribir_atomico(destino, falla)
    assert list(tmp_path.iterdir()) == []


def test_escribe_ventanas_indice_y_manifiesto(tmp_path: Path) -> None:
    a = Almacen(tmp_path)
    clips = [clip("1-Yo__IMG_1008__rep01", n=10), clip("2-Tu__IMG_1010__rep03", n=7)]
    ventanas = np.zeros((2, 75, 96), dtype=np.float32)

    a.escribir_ventanas(clips, ventanas, V, {"aprobacion_lote": {"fecha": "f", "sha256_informe": "s"}})

    assert np.array_equal(np.load(a.dir_ventanas / "ventanas.npy"), ventanas)
    with (a.dir_ventanas / "indice.csv").open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    assert list(filas[0]) == [
        "fila", "clip_id", "clase", "numero", "glosa", "grabacion", "repeticion",
        "n_fotogramas_60", "n_fotogramas_30", "md5_clip",
    ]
    assert [r["clip_id"] for r in filas] == ["1-Yo__IMG_1008__rep01", "2-Tu__IMG_1010__rep03"]
    assert filas[1]["fila"] == "1" and filas[1]["n_fotogramas_30"] == "4"
    m = json.loads((a.dir_ventanas / "manifiesto.json").read_text())
    assert m["ventanas"] == 2 and m["forma"] == [2, 75, 96] and m["dtype"] == "float32"
    assert m["sha256_ventanas"] == sha256_fichero(a.dir_ventanas / "ventanas.npy")
    assert m["version_cadena"] == V.version_cadena and m["macos"] == "27.2.0"
    assert m["aprobacion_lote"]["sha256_informe"] == "s"
