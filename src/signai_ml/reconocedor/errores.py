"""Errores del reconocedor y su código de salida (contracts/cli.md).

`1` error de los datos o del estado · `2` uso incorrecto. Cada mensaje dice la causa y qué hacer.
"""


class ErrorReconocedor(Exception):
    codigo_salida = 1


class ErrorUso(ErrorReconocedor):
    codigo_salida = 2


class SemillaNoPermitida(ErrorUso):
    """Solo se entrena con las semillas fijas del paquete (research R-06)."""


class RondaNoValida(ErrorReconocedor):
    """La declaración de una ronda no cumple contracts/ronda.md."""


class RondaModificada(ErrorReconocedor):
    """Una ronda ya lanzada cambió después de lanzarse (contracts/ronda.md, regla 7)."""


class BarridoCerrado(ErrorReconocedor):
    """Sobre un barrido cerrado no se lanzan corridas ni se declaran rondas (FR-010)."""


class DatosDistintos(ErrorReconocedor):
    """La versión del corpus o de la partición no es la del resto del barrido."""


class ArbolSucio(ErrorReconocedor):
    """Hay cambios sin confirmar: la versión del código registrada no describiría lo entrenado."""


class SinCandidato(ErrorReconocedor):
    """El barrido no tiene una configuración elegible con la que elegir candidato."""


class CorridaDivergente(ErrorReconocedor):
    """La pérdida dejó de ser finita durante el entrenamiento."""
