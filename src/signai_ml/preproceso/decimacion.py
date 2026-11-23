"""Decimación de 60 a 30 fps (FR-003; research R-02).

Se conservan los fotogramas de índice par, empezando por el primero: un clip de N fotogramas da
⌈N/2⌉. Los índices los decide esta etapa y el extractor solo entrega esos a Vision.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Decimacion:
    fps_origen: int = 60
    fps_destino: int = 30

    def __post_init__(self) -> None:
        if self.fps_origen != 60:
            raise ValueError(f"los clips del corpus son de 60 fps y este es de {self.fps_origen}")
        if self.fps_origen % self.fps_destino:
            raise ValueError(
                f"la razón {self.fps_origen}/{self.fps_destino} no es entera: no se puede decimar"
            )

    @property
    def cada(self) -> int:
        return self.fps_origen // self.fps_destino

    def indices(self, n: int) -> range:
        if n < 1:
            raise ValueError(f"un clip tiene al menos un fotograma, no {n}")
        return range(0, n, self.cada)
