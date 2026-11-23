"""Qué se copia (data-model, «Conjunto protegido»; research R-02; spec FR-001).

Dos conjuntos protegidos y nada más: el almacén de DVC y el seguimiento de MLflow. El registro de
copias viaja en cada instantánea como metadato, no como conjunto. Las rutas son relativas a la raíz
de datos, porque la copia se lanza desde ella.
"""

from dataclasses import dataclass
from pathlib import Path

REGISTRO = "copia/registro.jsonl"

_EXCLUIDAS = ("revision", "cierre", "studio", "copia/repositorio", "copia/para-ipad", "copia/restauracion-*")


@dataclass(frozen=True)
class ConjuntoProtegido:
    nombre: str
    ruta: str


def conjuntos() -> tuple[ConjuntoProtegido, ...]:
    return (ConjuntoProtegido("datos-versionados", "dvc-store"), ConjuntoProtegido("seguimiento", "mlflow"))


def rutas_de_copia(raiz: Path) -> tuple[str, ...]:
    """Los conjuntos, y el registro si ya existe: restic falla con una ruta inexistente."""
    rutas = tuple(conjunto.ruta for conjunto in conjuntos())
    return (*rutas, REGISTRO) if (Path(raiz) / REGISTRO).is_file() else rutas


def exclusiones(raiz: Path) -> tuple[str, ...]:
    """Defensa adicional: nada de esto cuelga de las rutas copiadas, pero nunca debe entrar."""
    return tuple(str(Path(raiz) / relativa) for relativa in _EXCLUIDAS)
