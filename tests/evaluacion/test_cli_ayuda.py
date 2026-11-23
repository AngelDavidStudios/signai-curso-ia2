"""Ayuda y códigos de uso de `signai-aceptacion` (contracts/cli.md)."""

import pytest

from signai_ml.aceptacion.cli import ORDENES, main


def test_sin_orden_muestra_la_ayuda_y_sale_con_2(capsys: pytest.CaptureFixture) -> None:
    assert main([]) == 2
    salida = capsys.readouterr().out
    for orden in ("evaluar", "registrar", "exportar", "estado"):
        assert orden in salida


def test_ayuda_explicita_sale_con_0(capsys: pytest.CaptureFixture) -> None:
    assert main(["--help"]) == 0
    assert "TT-15" in capsys.readouterr().out


def test_orden_desconocida_sale_con_2() -> None:
    assert main(["entrenar"]) == 2


def test_las_cuatro_ordenes_del_contrato() -> None:
    assert list(ORDENES) == ["evaluar", "registrar", "exportar", "estado"]
