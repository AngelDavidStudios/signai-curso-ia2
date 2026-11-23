"""Lectura de la entrada de la cadena: los clips de TT-11 y su `etiquetas.csv` (FR-001, FR-002)."""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from signai_ml.preproceso.clips import ClipEtiquetado, comprobar_md5, leer_etiquetas
from signai_ml.preproceso.errores import ClipAlterado, ErrorPreproceso

from .conftest import fila_etiqueta

Escribir = Callable[[Path, Sequence[dict[str, object]]], Path]


def test_lee_cada_fila_como_clip_etiquetado(tmp_path: Path, escribir_etiquetas: Escribir) -> None:
    raiz = tmp_path / "clips"
    escribir_etiquetas(raiz, [
        fila_etiqueta("2-Tu__IMG_1010__rep01", 40, "b" * 32),
        fila_etiqueta("1-Yo__IMG_1008__rep02", 82, "a" * 32),
    ])

    clips = leer_etiquetas(raiz)

    assert [c.clip_id for c in clips] == ["1-Yo__IMG_1008__rep02", "2-Tu__IMG_1010__rep01"]
    yo = clips[0]
    assert yo == ClipEtiquetado(
        clip_id="1-Yo__IMG_1008__rep02", clase="1-Yo", numero=1, glosa="yo", grabacion="IMG_1008",
        repeticion=2, n_fotogramas=82, md5="a" * 32,
        ruta=raiz / "1-Yo" / "1-Yo__IMG_1008__rep02.mp4",
    )


def test_rechaza_clip_id_duplicado(tmp_path: Path, escribir_etiquetas: Escribir) -> None:
    raiz = tmp_path / "clips"
    fila = fila_etiqueta("1-Yo__IMG_1008__rep01", 82, "a" * 32)
    escribir_etiquetas(raiz, [fila, fila])

    with pytest.raises(ErrorPreproceso, match="duplicado"):
        leer_etiquetas(raiz)


def test_rechaza_n_fotogramas_menor_que_uno(tmp_path: Path, escribir_etiquetas: Escribir) -> None:
    raiz = tmp_path / "clips"
    escribir_etiquetas(raiz, [fila_etiqueta("1-Yo__IMG_1008__rep01", 0, "a" * 32)])

    with pytest.raises(ErrorPreproceso, match="n_fotogramas"):
        leer_etiquetas(raiz)


def test_falta_etiquetas_csv(tmp_path: Path) -> None:
    with pytest.raises(ErrorPreproceso, match="etiquetas.csv"):
        leer_etiquetas(tmp_path / "no-existe")


def test_comprobar_md5_acepta_el_clip_intacto_y_rechaza_el_alterado(tmp_path: Path) -> None:
    ruta = tmp_path / "1-Yo" / "1-Yo__IMG_1008__rep01.mp4"
    ruta.parent.mkdir(parents=True)
    ruta.write_bytes(b"contenido")
    import hashlib
    bueno = hashlib.md5(b"contenido").hexdigest()
    clip = ClipEtiquetado("1-Yo__IMG_1008__rep01", "1-Yo", 1, "yo", "IMG_1008", 1, 82, bueno, ruta)

    comprobar_md5(clip)

    ruta.write_bytes(b"otro")
    with pytest.raises(ClipAlterado, match="1-Yo__IMG_1008__rep01"):
        comprobar_md5(clip)
