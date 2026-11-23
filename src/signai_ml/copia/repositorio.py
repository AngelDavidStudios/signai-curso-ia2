"""Puerto `Repositorio` y su adaptador sobre restic (research R-01, R-07, R-08).

La copia y la restauración dependen del puerto, no de restic (principio II, DIP). El adaptador
pasa la clave **solo** en el entorno del proceso hijo, nunca en `argv`, y copia desde la raíz de
datos con rutas relativas: así el árbol de cada instantánea es relativo y se restaura en cualquier
destino. Comportamientos de restic 0.19.1 comprobados: código 12 = clave incorrecta, 10 = no hay
repositorio, y `forget` necesita `--group-by host`.
"""

import json
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .errores import FaltaHerramienta, RechazoDominio

Ejecutar = Callable[..., subprocess.CompletedProcess[str]]

_CLAVE_INCORRECTA = 12
_SIN_REPOSITORIO = 10
_OTRAS_FUENTES_DE_CLAVE = ("RESTIC_PASSWORD_FILE", "RESTIC_PASSWORD_COMMAND")


@dataclass(frozen=True)
class Instantanea:
    id: str
    fecha: str
    etiquetas: tuple[str, ...]
    bytes: int = 0


@dataclass(frozen=True)
class ResultadoCopia:
    instantanea: str
    bytes_anadidos: int


class Repositorio(Protocol):
    def existe(self) -> bool: ...

    def iniciar(self) -> None: ...

    def copiar(
        self,
        rutas: Sequence[str],
        *,
        base: Path,
        etiquetas: Sequence[str],
        excluir: Sequence[str] = (),
        cuando: str | None = None,
    ) -> ResultadoCopia: ...

    def instantaneas(self) -> list[Instantanea]: ...

    def olvidar(self, *, dentro_de: str, ultimas: int) -> None: ...

    def verificar(self, *, subconjunto: str | None = None, completo: bool = False) -> None: ...

    def restaurar(self, instantanea: str, destino: Path, *, ruta: str | None = None) -> None: ...


class Restic:
    def __init__(self, ruta: Path, clave: str, ejecutar: Ejecutar = subprocess.run) -> None:
        self._ruta = Path(ruta)
        self._clave = clave
        self._ejecutar = ejecutar

    def existe(self) -> bool:
        return (self._ruta / "config").is_file()

    def iniciar(self) -> None:
        if self.existe():
            raise RechazoDominio(f"el repositorio ya existe en {self._ruta}")
        self._restic("init")

    def copiar(
        self,
        rutas: Sequence[str],
        *,
        base: Path,
        etiquetas: Sequence[str],
        excluir: Sequence[str] = (),
        cuando: str | None = None,
    ) -> ResultadoCopia:
        opciones = [opcion for etiqueta in etiquetas for opcion in ("--tag", etiqueta)]
        opciones += [opcion for patron in excluir for opcion in ("--exclude", patron)]
        if cuando is not None:
            opciones += ["--time", cuando]
        salida = self._restic("backup", "--json", *opciones, *rutas, cwd=Path(base))
        resumen = next(
            mensaje
            for mensaje in map(json.loads, reversed(salida.splitlines()))
            if mensaje.get("message_type") == "summary"
        )
        return ResultadoCopia(resumen["snapshot_id"], int(resumen.get("data_added", 0)))

    def instantaneas(self) -> list[Instantanea]:
        datos = json.loads(self._restic("snapshots", "--json") or "[]")
        return sorted(
            (
                Instantanea(
                    d["id"], d["time"], tuple(d.get("tags") or ()), int((d.get("summary") or {}).get("total_bytes_processed", 0))
                )
                for d in datos
            ),
            key=lambda instantanea: instantanea.fecha,
        )

    def olvidar(self, *, dentro_de: str, ultimas: int) -> None:
        # Sin --group-by host, una instantánea con otras rutas forma su propio grupo y nunca se olvida.
        self._restic(
            "forget", "--group-by", "host", "--keep-within", dentro_de, "--keep-last", str(ultimas), "--prune"
        )

    def verificar(self, *, subconjunto: str | None = None, completo: bool = False) -> None:
        if completo:
            self._restic("check", "--read-data")
        elif subconjunto is not None:
            self._restic("check", f"--read-data-subset={subconjunto}")
        else:
            self._restic("check")

    def restaurar(self, instantanea: str, destino: Path, *, ruta: str | None = None) -> None:
        incluir = ["--include", f"/{ruta.lstrip('/')}"] if ruta else []
        self._restic("restore", instantanea, "--target", str(destino), *incluir)

    def _restic(self, *argumentos: str, cwd: Path | None = None) -> str:
        entorno = {k: v for k, v in os.environ.items() if k not in _OTRAS_FUENTES_DE_CLAVE}
        entorno |= {"RESTIC_REPOSITORY": str(self._ruta), "RESTIC_PASSWORD": self._clave}
        try:
            resultado = self._ejecutar(
                ["restic", *argumentos], capture_output=True, text=True, env=entorno, cwd=cwd, check=False
            )
        except FileNotFoundError as error:
            raise FaltaHerramienta("falta 'restic' en el PATH: brew install restic") from error
        if resultado.returncode == _CLAVE_INCORRECTA:
            raise RechazoDominio("la clave no abre el repositorio: revisa lo tecleado contra el papel")
        if resultado.returncode == _SIN_REPOSITORIO:
            raise RechazoDominio(f"no hay ningún repositorio en {self._ruta}: ejecuta `signai-copia iniciar`")
        if resultado.returncode != 0:
            ultima = (resultado.stderr.strip().splitlines() or ["sin detalle"])[-1]
            raise RechazoDominio(f"restic {argumentos[0]} falló: {ultima}")
        return resultado.stdout
