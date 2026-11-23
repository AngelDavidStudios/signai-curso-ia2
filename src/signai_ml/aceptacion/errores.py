"""Errores de la aceptación y su código de salida (contracts/cli.md).

`1` error de los datos o del estado · `2` uso incorrecto · `3` árbol con cambios sin confirmar. Cada
mensaje dice la causa y qué hacer.
"""


class ErrorAceptacion(Exception):
    codigo_salida = 1


class ErrorUso(ErrorAceptacion):
    codigo_salida = 2


class ArbolSucio(ErrorAceptacion):
    """Hay cambios sin confirmar: la versión del código registrada no describiría lo ejecutado."""

    codigo_salida = 3


class CandidatoNoValido(ErrorAceptacion):
    """`modelos/reconocedor/` no cumple el contrato `modelo.md` de TT-14 o el límite de parámetros."""


class HuellasDistintas(ErrorAceptacion):
    """El corpus o la partición actuales no son aquellos con los que se entrenó el candidato (FR-001)."""


class PruebaInesperada(ErrorAceptacion):
    """La prueba no tiene los 1.800 clips, 18 por clase, de ADR-0011."""


class NoAceptado(ErrorAceptacion):
    """El candidato no tiene una evaluación terminada con `aceptado`: no se registra ni se exporta."""


class ArtefactoNoValido(ErrorAceptacion):
    """La conversión falló o el artefacto no pasa las comprobaciones que hace la app."""


class ParidadFallida(ErrorAceptacion):
    """El exportado no da los mismos resultados que el original (FR-012)."""
