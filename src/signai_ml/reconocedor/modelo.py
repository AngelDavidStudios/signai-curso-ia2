"""KPoseFormer (contracts/modelo.md; spec FR-001, FR-002; research R-02; ADR-0013).

La forma de la referencia publicada (Vilcamiza, ICPRS 2025, §IV-D), adaptada a 75 fotogramas:

1. *pooling* espacial atendido: `Linear(96 → 48)` + softmax sobre los 48 puntos → centroide `(x, y)`
   concatenado a los 96 → 98;
2. proyección `Linear(98 → d)` solo si la forma pide `d ≠ 98`;
3. *dropout* y codificación posicional aprendible `(75, d)`;
4. capas *pre-norm* de atención multicabeza y red intermedia GELU;
5. *LayerNorm*, *pooling* temporal atendido, GELU y `Linear(d → clases)`.

La atención se escribe con `Linear`, `matmul` y `softmax`: `nn.TransformerEncoder` no convierte a
Core ML (lo comprobó el laboratorio), y así TT-15 exporta este mismo modelo sin reescribirlo.
"""

import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from . import CLASES, FOTOGRAMAS, RASGOS
from .configuracion import Forma

FORMATO = "signai-ml/reconocedor/1"
PUNTOS = RASGOS // 2


class AtencionMulticabeza(nn.Module):
    """Atención multicabeza con `Linear`, `matmul` y `softmax`.

    La dimensión por cabeza y la escala se fijan aquí, como atributos de Python y no como *buffers*,
    para que el `state_dict` no cambie. Si se calcularan a partir de `x.shape`, el trazado dejaría
    una división entera en el grafo, que coremltools 9.0 no convierte (TT-15, research R-06). El
    resultado es idéntico bit a bit al de la versión con la que se entrenó TT-14.
    """

    def __init__(self, dimension: int, cabezas: int) -> None:
        super().__init__()
        self.cabezas = cabezas
        self.dimension = dimension
        self.por_cabeza = dimension // cabezas
        self.escala = math.sqrt(self.por_cabeza)
        self.consulta = nn.Linear(dimension, dimension)
        self.clave = nn.Linear(dimension, dimension)
        self.valor = nn.Linear(dimension, dimension)
        self.salida = nn.Linear(dimension, dimension)

    def _separar(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        return x.reshape(b, t, self.cabezas, self.por_cabeza).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape[0], x.shape[1]
        q, k, v = (self._separar(capa(x), b, t) for capa in (self.consulta, self.clave, self.valor))
        pesos = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.escala, dim=-1)
        return self.salida(torch.matmul(pesos, v).transpose(1, 2).reshape(b, t, self.dimension))


class CapaCodificador(nn.Module):
    """Capa *pre-norm*: atención y red intermedia, cada una con su residuo."""

    def __init__(self, dimension: int, cabezas: int, intermedia: int, dropout: float) -> None:
        super().__init__()
        self.norma_atencion = nn.LayerNorm(dimension)
        self.atencion = AtencionMulticabeza(dimension, cabezas)
        self.norma_intermedia = nn.LayerNorm(dimension)
        self.intermedia = nn.Sequential(
            nn.Linear(dimension, intermedia), nn.GELU(), nn.Dropout(dropout), nn.Linear(intermedia, dimension)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.atencion(self.norma_atencion(x)))
        return x + self.dropout(self.intermedia(self.norma_intermedia(x)))


class PoolingEspacial(nn.Module):
    """Añade a cada fotograma el centroide atendido de sus 48 puntos: 96 → 98."""

    def __init__(self) -> None:
        super().__init__()
        self.puntua = nn.Linear(RASGOS, PUNTOS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pesos = torch.softmax(self.puntua(x), dim=-1).unsqueeze(-1)
        xy = x.reshape(x.shape[0], x.shape[1], PUNTOS, 2)
        return torch.cat([x, (pesos * xy).sum(dim=2)], dim=-1)


class PoolingTemporal(nn.Module):
    """Media de los fotogramas ponderada por una atención aprendida."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.puntua = nn.Linear(dimension, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.softmax(self.puntua(x), dim=1) * x).sum(dim=1)


class KPoseFormer(nn.Module):
    def __init__(self, forma: Forma, clases: int = CLASES, dropout: float = 0.0) -> None:
        super().__init__()
        if forma.dimension % forma.cabezas:
            raise ValueError(f"{forma.cabezas} cabezas no dividen la dimensión {forma.dimension}")
        self.espacial = PoolingEspacial()
        entrada = RASGOS + 2
        self.proyeccion = nn.Identity() if forma.dimension == entrada else nn.Linear(entrada, forma.dimension)
        self.dropout = nn.Dropout(dropout)
        self.posicional = nn.Parameter(torch.zeros(1, FOTOGRAMAS, forma.dimension))
        self.capas = nn.ModuleList(
            CapaCodificador(forma.dimension, forma.cabezas, forma.intermedia, dropout) for _ in range(forma.capas)
        )
        self.norma = nn.LayerNorm(forma.dimension)
        self.temporal = PoolingTemporal(forma.dimension)
        self.activacion = nn.GELU()
        self.clasificador = nn.Linear(forma.dimension, clases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.proyeccion(self.espacial(x))) + self.posicional
        for capa in self.capas:
            h = capa(h)
        return self.clasificador(self.activacion(self.temporal(self.norma(h))))


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters())


def guardar(
    ruta: Path, modelo: KPoseFormer, forma: Forma, clases: list[str], glosas: list[str], corrida: dict, huellas: dict
) -> Path:
    """`modelo.pt` con el formato de contracts/modelo.md M-2."""
    torch.save(
        {
            "formato": FORMATO,
            "forma": asdict(forma),
            "clases": list(clases),
            "glosas": list(glosas),
            "estado": {k: v.detach().cpu() for k, v in modelo.state_dict().items()},
            "corrida": dict(corrida),
            "huellas": dict(huellas),
        },
        ruta,
    )
    return Path(ruta)


def cargar(ruta: Path) -> tuple[KPoseFormer, dict]:
    """Reconstruye el modelo sin conocer su configuración y devuelve sus metadatos."""
    datos = torch.load(ruta, map_location="cpu", weights_only=True)
    if datos.get("formato") != FORMATO:
        raise ValueError(f"{ruta}: formato {datos.get('formato')!r}; se esperaba {FORMATO}")
    modelo = KPoseFormer(Forma(**datos["forma"]), clases=len(datos["clases"]))
    modelo.load_state_dict(datos["estado"])
    return modelo.eval(), {k: v for k, v in datos.items() if k != "estado"}
