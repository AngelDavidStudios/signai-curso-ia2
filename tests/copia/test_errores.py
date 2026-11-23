"""Errores de la copia y su código de salida en la CLI (contracts/cli.md)."""

import pytest

from signai_ml.copia.errores import ErrorCopia, ErrorUso, FaltaHerramienta, RechazoDominio


@pytest.mark.parametrize(
    ("error", "codigo"),
    [
        (RechazoDominio("hay corridas en curso"), 1),
        (ErrorUso("falta --bucket"), 2),
        (FaltaHerramienta("restic"), 3),
    ],
)
def test_codigos_de_salida(error: ErrorCopia, codigo: int) -> None:
    assert isinstance(error, ErrorCopia)
    assert error.codigo_salida == codigo


def test_falta_herramienta_dice_cual() -> None:
    assert "restic" in str(FaltaHerramienta("restic"))
    assert not isinstance(FaltaHerramienta("restic"), RechazoDominio)
