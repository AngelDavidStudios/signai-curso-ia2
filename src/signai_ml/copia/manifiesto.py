"""Inventarios del origen (data-model; research R-09, R-10).

El **inventario ligero** (ruta, tamaño y fecha de modificación) se guarda tras cada actualización en
`copia/ultimo-manifiesto.json`. `estado` lo compara con el origen actual para saber qué conjunto ha
cambiado sin abrir el repositorio, y por tanto sin pedir la clave. El **manifiesto con huellas**
(SHA-256) solo se usa en la prueba de restauración, para demostrar que lo restaurado es idéntico.
"""

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .conjuntos import conjuntos

_ULTIMO = Path("copia") / "ultimo-manifiesto.json"


@dataclass(frozen=True)
class EntradaLigera:
    ruta: str
    bytes: int
    mtime: int


def _archivos(raiz: Path, ruta: str) -> Iterable[Path]:
    origen = raiz / ruta
    if origen.is_file():
        yield origen
    elif origen.is_dir():
        yield from (archivo for archivo in origen.rglob("*") if archivo.is_file())


def inventario_ligero(raiz: Path, rutas: Sequence[str]) -> list[EntradaLigera]:
    raiz = Path(raiz)
    entradas = []
    for ruta in rutas:
        for archivo in _archivos(raiz, ruta):
            estado = archivo.stat()
            entradas.append(EntradaLigera(archivo.relative_to(raiz).as_posix(), estado.st_size, estado.st_mtime_ns))
    return sorted(entradas, key=lambda entrada: entrada.ruta)


def guardar(raiz: Path, fecha: str, entradas: Sequence[EntradaLigera]) -> Path:
    ruta = Path(raiz) / _ULTIMO
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"fecha": fecha, "entradas": [asdict(e) for e in entradas]}), encoding="utf-8")
    return ruta


def leer(raiz: Path) -> tuple[str, list[EntradaLigera]] | None:
    ruta = Path(raiz) / _ULTIMO
    if not ruta.exists():
        return None
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return datos["fecha"], [EntradaLigera(**entrada) for entrada in datos["entradas"]]


def cambiados(anterior: Sequence[EntradaLigera], actual: Sequence[EntradaLigera]) -> list[str]:
    """Nombres de los conjuntos con alguna alta, baja o cambio de tamaño o de fecha."""
    diferentes = set(anterior) ^ set(actual)
    return [
        conjunto.nombre
        for conjunto in conjuntos()
        if any(entrada.ruta.startswith(f"{conjunto.ruta}/") for entrada in diferentes)
    ]


# ── Manifiesto con huellas, para la prueba de restauración (research R-09) ─────────────────────

_BLOQUE = 1024 * 1024


@dataclass(frozen=True)
class EntradaHuella:
    ruta: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Comparacion:
    faltan: tuple[str, ...]
    sobran: tuple[str, ...]
    distintos: tuple[str, ...]

    @property
    def identico(self) -> bool:
        """La restauración es correcta solo si los tres están vacíos (SC-003)."""
        return not (self.faltan or self.sobran or self.distintos)


def _sha256(archivo: Path) -> str:
    huella = hashlib.sha256()
    with archivo.open("rb") as entrada:
        while bloque := entrada.read(_BLOQUE):
            huella.update(bloque)
    return huella.hexdigest()


def manifiesto(raiz: Path, rutas: Sequence[str]) -> list[EntradaHuella]:
    raiz = Path(raiz)
    entradas = [
        EntradaHuella(archivo.relative_to(raiz).as_posix(), archivo.stat().st_size, _sha256(archivo))
        for ruta in rutas
        for archivo in _archivos(raiz, ruta)
    ]
    return sorted(entradas, key=lambda entrada: entrada.ruta)


def comparar(origen: Sequence[EntradaHuella], restaurado: Sequence[EntradaHuella]) -> Comparacion:
    antes = {entrada.ruta: entrada for entrada in origen}
    despues = {entrada.ruta: entrada for entrada in restaurado}
    return Comparacion(
        faltan=tuple(sorted(antes.keys() - despues.keys())),
        sobran=tuple(sorted(despues.keys() - antes.keys())),
        distintos=tuple(sorted(r for r in antes.keys() & despues.keys() if antes[r] != despues[r])),
    )
