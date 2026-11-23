"""Errores de la cadena de preprocesamiento y su código de salida (contracts/cli.md).

`1` error de los datos o del proceso · `2` uso incorrecto. Cada mensaje dice la causa y qué hacer.
"""


class ErrorPreproceso(Exception):
    codigo_salida = 1


class ErrorUso(ErrorPreproceso):
    codigo_salida = 2


class ErrorExtraccion(ErrorPreproceso):
    """La salida del extractor no corresponde al clip o no es de la versión vigente."""

    def __init__(self, clip_id: str, mensaje: str) -> None:
        self.clip_id = clip_id
        super().__init__(f"{clip_id}: {mensaje}")


class ClipAlterado(ErrorPreproceso):
    """El contenido del clip no coincide con la huella registrada en TT-11 (FR-002)."""

    def __init__(self, clip_id: str, esperado: str, obtenido: str) -> None:
        self.clip_id = clip_id
        super().__init__(
            f"{clip_id}: el md5 del clip es {obtenido} y etiquetas.csv dice {esperado}. "
            "Restaura el clip con `uv run dvc checkout datos/clips.dvc` y vuelve a lanzar la orden."
        )


class FaltaAprobacion(ErrorPreproceso):
    """El corpus completo no se procesa sin una aprobación vigente del lote (FR-021)."""


class FaltaHerramienta(ErrorPreproceso):
    def __init__(self, herramienta: str, como: str) -> None:
        self.herramienta = herramienta
        super().__init__(f"Falta la herramienta '{herramienta}'. {como}")
