"""Decimación de 60 a 30 fps (FR-003, FR-004; research R-02)."""

import math

import pytest

from signai_ml.preproceso.decimacion import Decimacion


@pytest.mark.parametrize("n", [1, 2, 35, 82, 203])
def test_conserva_los_indices_pares_desde_el_primero(n: int) -> None:
    indices = Decimacion().indices(n)
    assert list(indices) == list(range(0, n, 2))
    assert len(indices) == math.ceil(n / 2)


def test_el_paso_es_dos() -> None:
    assert Decimacion().cada == 2


def test_rechaza_una_tasa_de_origen_distinta_de_60() -> None:
    with pytest.raises(ValueError, match="60"):
        Decimacion(fps_origen=59)


def test_rechaza_una_razon_no_entera() -> None:
    with pytest.raises(ValueError, match="entera"):
        Decimacion(fps_origen=60, fps_destino=25)


def test_rechaza_n_menor_que_uno() -> None:
    with pytest.raises(ValueError):
        Decimacion().indices(0)
