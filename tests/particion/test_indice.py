"""Lectura y validación del corpus procesado (data-model «Reglas de validación»; spec FR-011)."""

import hashlib
import json
from pathlib import Path

import pytest

from signai_ml.particion.errores import IndiceNoValido
from signai_ml.particion.indice import leer_corpus

from .sinteticos import RECUENTOS_CORPUS, escribir_corpus, escribir_indice, indice_sintetico


def _ventanas(raiz: Path) -> Path:
    return raiz / "datos" / "ventanas"


def test_lee_filas_huellas_y_n(raiz_sintetica: Path) -> None:
    corpus = leer_corpus(raiz_sintetica)
    dir_ventanas = _ventanas(raiz_sintetica)
    manifiesto = json.loads((dir_ventanas / "manifiesto.json").read_text(encoding="utf-8"))
    assert corpus.n == 8996
    assert len(corpus.filas) == 8996
    assert corpus.filas[0]["fila"] == "0" and corpus.filas[0]["clip_id"] == "1-Sint__SINT_1_0__rep01"
    assert corpus.sha256_ventanas == manifiesto["sha256_ventanas"]
    assert corpus.sha256_indice == hashlib.sha256((dir_ventanas / "indice.csv").read_bytes()).hexdigest()


def test_rechaza_un_numero_de_clases_distinto_de_100(tmp_path: Path) -> None:
    recuentos = dict(list(RECUENTOS_CORPUS.items())[:99])
    raiz = escribir_corpus(tmp_path, indice_sintetico(recuentos))
    with pytest.raises(IndiceNoValido, match="99 clases"):
        leer_corpus(raiz)


def test_rechaza_un_clip_repetido(raiz_sintetica: Path) -> None:
    filas = indice_sintetico(RECUENTOS_CORPUS)
    filas[1]["clip_id"] = filas[0]["clip_id"]
    escribir_indice(_ventanas(raiz_sintetica), filas)
    with pytest.raises(IndiceNoValido, match=filas[0]["clip_id"]):
        leer_corpus(raiz_sintetica)


def test_rechaza_filas_que_no_son_0_a_n_menos_1(raiz_sintetica: Path) -> None:
    filas = indice_sintetico(RECUENTOS_CORPUS)
    filas[5]["fila"] = "9999"
    escribir_indice(_ventanas(raiz_sintetica), filas)
    with pytest.raises(IndiceNoValido, match="fila"):
        leer_corpus(raiz_sintetica)


def test_rechaza_un_n_distinto_del_manifiesto(raiz_sintetica: Path) -> None:
    ruta = _ventanas(raiz_sintetica) / "manifiesto.json"
    manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
    manifiesto["ventanas"] = 9000
    ruta.write_text(json.dumps(manifiesto), encoding="utf-8")
    with pytest.raises(IndiceNoValido, match="9000"):
        leer_corpus(raiz_sintetica)


@pytest.mark.parametrize("fichero", ["indice.csv", "manifiesto.json"])
def test_sin_corpus_procesado_dice_como_restaurarlo(raiz_sintetica: Path, fichero: str) -> None:
    (_ventanas(raiz_sintetica) / fichero).unlink()
    with pytest.raises(IndiceNoValido, match=r"uv run dvc checkout datos/ventanas\.dvc"):
        leer_corpus(raiz_sintetica)


# ── TT-22 · la grabación es la unidad de partición (research R-01) ──


def test_rechaza_un_indice_sin_la_columna_grabacion(raiz_sintetica: Path) -> None:
    import csv

    filas = indice_sintetico(RECUENTOS_CORPUS)
    columnas = [c for c in filas[0] if c != "grabacion"]
    with (_ventanas(raiz_sintetica) / "indice.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore", lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    with pytest.raises(IndiceNoValido, match="grabacion"):
        leer_corpus(raiz_sintetica)


def test_rechaza_un_clip_sin_grabacion(raiz_sintetica: Path) -> None:
    filas = indice_sintetico(RECUENTOS_CORPUS)
    filas[3]["grabacion"] = ""
    escribir_indice(_ventanas(raiz_sintetica), filas)
    with pytest.raises(IndiceNoValido, match=filas[3]["clip_id"]):
        leer_corpus(raiz_sintetica)
