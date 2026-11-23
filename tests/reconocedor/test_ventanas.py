"""Lectura común de las ventanas del corpus procesado (`signai_ml.ventanas`; research R-02 de TT-15)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from signai_ml.ventanas import Corpus, VentanasNoValidas, abrir_corpus, huellas_corpus, vocabulario

from .sinteticos import nombre_clase


def test_vocabulario_ordenado_por_numero(raiz_sintetica: Path) -> None:
    filas = vocabulario(raiz_sintetica)

    assert [f["numero"] for f in filas] == ["1", "2", "3", "4"]
    assert filas[0]["clase"] == nombre_clase(1)


def test_vocabulario_con_huecos_es_un_error(raiz_sintetica: Path) -> None:
    ruta = raiz_sintetica / "config" / "corpus" / "vocabulario.csv"
    ruta.write_text("\n".join(l for l in ruta.read_text().splitlines() if not l.startswith("3,")) + "\n")

    with pytest.raises(VentanasNoValidas, match="1…N"):
        vocabulario(raiz_sintetica)


def test_abrir_corpus(raiz_sintetica: Path) -> None:
    corpus = abrir_corpus(raiz_sintetica)

    assert isinstance(corpus, Corpus)
    assert corpus.clases == tuple(nombre_clase(n) for n in range(1, 5))
    assert corpus.glosas == ("glosa1", "glosa2", "glosa3", "glosa4")
    assert corpus.etiquetas.dtype == np.int64
    assert corpus.etiquetas.tolist() == [c for c in range(4) for _ in range(8)]


def test_indice_que_no_corresponde_al_vocabulario(raiz_sintetica: Path) -> None:
    ruta = raiz_sintetica / "config" / "corpus" / "vocabulario.csv"
    ruta.write_text(ruta.read_text().replace("2-Sint2", "2-Otra"), encoding="utf-8")

    with pytest.raises(VentanasNoValidas, match="vocabulario"):
        abrir_corpus(raiz_sintetica)


def test_tomar_filas(raiz_sintetica: Path) -> None:
    corpus = abrir_corpus(raiz_sintetica)

    x, y = corpus.tomar(np.array([0, 9, 31]))

    assert x.shape == (3, 75, 96)
    assert x.dtype == np.float32
    assert y.tolist() == [0, 1, 3]
    assert np.array_equal(x[1], np.load(raiz_sintetica / "datos" / "ventanas" / "ventanas.npy")[9])


def test_huellas_del_corpus(raiz_sintetica: Path) -> None:
    ventanas = raiz_sintetica / "datos" / "ventanas"
    manifiesto = json.loads((ventanas / "manifiesto.json").read_text())

    assert huellas_corpus(raiz_sintetica) == {
        "corpus.sha256_ventanas": manifiesto["sha256_ventanas"],
        "corpus.sha256_indice": hashlib.sha256((ventanas / "indice.csv").read_bytes()).hexdigest(),
    }
    assert abrir_corpus(raiz_sintetica).huellas == huellas_corpus(raiz_sintetica)
