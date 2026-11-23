"""Errores de la copia y su código de salida en la CLI (contracts/cli.md).

`1` rechazo de dominio · `2` error de uso · `3` falta una herramienta.
"""


class ErrorCopia(Exception):
    codigo_salida = 1


class RechazoDominio(ErrorCopia):
    """La operación no se puede hacer en este estado: precondición, verificación o clave."""

    codigo_salida = 1


class ErrorUso(ErrorCopia):
    codigo_salida = 2


class FaltaHerramienta(ErrorCopia):
    codigo_salida = 3
