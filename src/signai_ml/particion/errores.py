"""Errores de la partición y su código de salida (contracts/cli.md).

`1` error de los datos o del proceso (incluido «ya está fijada») · `2` uso incorrecto. Cada mensaje
dice la causa y qué hacer.
"""

RESTAURAR = "uv run dvc checkout datos/ventanas.dvc datos/particion.dvc"


class ErrorParticion(Exception):
    codigo_salida = 1


class ErrorUso(ErrorParticion):
    codigo_salida = 2


class IndiceNoValido(ErrorParticion):
    """El índice del corpus procesado no permite generar la partición (FR-011)."""


class ParticionFijada(ErrorParticion):
    """Ya existe una partición: no se rehace entre experimentos (FR-006, research R-06)."""


class ParticionAusente(ErrorParticion):
    """No hay partición que leer o verificar."""


class ParticionNoValida(ErrorParticion):
    """La partición no corresponde al corpus procesado o no cumple sus reglas (FR-005, R-05, R-08)."""
