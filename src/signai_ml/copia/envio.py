"""El envío al iPad: una carpeta fechada que se arrastra con Finder (data-model; research R-03, R-05).

Finder no fusiona carpetas: al soltar una con el mismo nombre, la sustituye entera. Por eso cada
actualización prepara una carpeta con nombre propio, clonada del repositorio en APFS (no ocupa
espacio extra), y la anterior no se borra hasta que el responsable confirma que la nueva llegó
entera, con el recuento que muestra el iPad.
"""

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .errores import FaltaHerramienta, RechazoDominio

PREFIJO = "signai-copia-"


@dataclass(frozen=True)
class Envio:
    carpeta: str
    ruta: Path
    instantanea: str
    archivos: int
    elementos: int
    bytes: int

    @property
    def espacio_necesario(self) -> int:
        """Durante el cambio conviven en el iPad la copia anterior y la nueva."""
        return 2 * self.bytes


class Clonador(Protocol):
    def clonar(self, origen: Path, destino: Path) -> None: ...


class ClonadorApfs:
    """`cp -cR` clona en APFS: la carpeta de envío no duplica el repositorio en el disco."""

    def clonar(self, origen: Path, destino: Path) -> None:
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        try:
            resultado = subprocess.run(
                ["cp", "-cR", str(origen), str(destino)], capture_output=True, text=True, check=False
            )
        except FileNotFoundError as error:
            raise FaltaHerramienta("falta 'cp' en el PATH") from error
        if resultado.returncode != 0:
            raise RechazoDominio(f"no se pudo clonar el repositorio: {resultado.stderr.strip()}")


def nombre_carpeta(momento: datetime) -> str:
    return f"{PREFIJO}{momento:%Y%m%d-%H%M}"


def contar(ruta: Path) -> tuple[int, int, int]:
    """Archivos, elementos (archivos y subcarpetas, sin la propia carpeta) y bytes."""
    archivos = carpetas = total = 0
    for elemento in Path(ruta).rglob("*"):
        if elemento.is_dir():
            carpetas += 1
        elif elemento.is_file():
            archivos += 1
            total += elemento.stat().st_size
    return archivos, archivos + carpetas, total


class Envios:
    def __init__(self, raiz: Path, clonador: Clonador) -> None:
        self._repositorio = Path(raiz) / "copia" / "repositorio"
        self._para_ipad = Path(raiz) / "copia" / "para-ipad"
        self._clonador = clonador

    def preparar(self, instantanea: str, momento: datetime) -> Envio:
        carpeta = nombre_carpeta(momento)
        destino = self._para_ipad / carpeta
        if destino.exists():
            raise RechazoDominio(f"ya existe un envío {carpeta}: espera un minuto o confírmalo antes")
        self._para_ipad.mkdir(parents=True, exist_ok=True)
        self._clonador.clonar(self._repositorio, destino)
        archivos, elementos, total = contar(destino)
        return Envio(carpeta, destino, instantanea, archivos, elementos, total)

    def confirmar(self, carpeta: str, recuento: int) -> tuple[int, int]:
        """Comprueba el recuento del iPad y borra del equipo los envíos anteriores."""
        ruta = self._para_ipad / carpeta
        if not carpeta.startswith(PREFIJO) or not ruta.is_dir():
            raise RechazoDominio(f"no hay ningún envío preparado con el nombre {carpeta}")
        archivos, elementos, total = contar(ruta)
        if recuento not in (archivos, elementos):
            raise RechazoDominio(
                f"el iPad muestra {recuento} y el envío tiene {archivos} archivos ({elementos} elementos): "
                "la copia no llegó entera, vuelve a arrastrarla"
            )
        # El nombre lleva la fecha: solo se borran las anteriores, nunca un envío más nuevo sin confirmar.
        for anterior in self._para_ipad.glob(f"{PREFIJO}*"):
            if anterior.is_dir() and anterior.name < carpeta:
                shutil.rmtree(anterior)
        return archivos, total
