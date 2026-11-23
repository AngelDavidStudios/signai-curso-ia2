"""Puerto `EstadoDvc` y su adaptador (research R-02).

La copia protege el **remoto** de DVC (`dvc-store/`), no la caché del repositorio. Si quedan objetos
sin `dvc push`, la copia estaría incompleta sin dar error: por eso es una precondición. Tras una
restauración, el almacén restaurado se compara con los punteros del repositorio mediante un remoto
temporal que siempre se retira.
"""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .errores import FaltaHerramienta, RechazoDominio

Ejecutar = Callable[..., subprocess.CompletedProcess[str]]


_REMOTO_TEMPORAL = "prueba-restauracion"


class EstadoDvc(Protocol):
    def sin_subir(self) -> bool:
        """`True` si hay datos versionados que todavía no están en el remoto."""
        ...

    def ausentes_en(self, almacen: Path) -> int:
        """Cuántos datos de los punteros del repositorio faltan en `almacen` (research R-09)."""
        ...


class EstadoDvcLocal:
    def __init__(self, raiz_repo: Path, ejecutar: Ejecutar = subprocess.run) -> None:
        self._raiz_repo = Path(raiz_repo)
        self._ejecutar = ejecutar

    def sin_subir(self) -> bool:
        return bool(self._dvc("status", "--cloud", "--json"))

    def ausentes_en(self, almacen: Path) -> int:
        # Solo el número: las rutas que devuelve DVC llevan nombres de clips y de grabaciones.
        self._dvc("remote", "add", "--local", "-f", _REMOTO_TEMPORAL, str(almacen), json_=False)
        try:
            return len(self._dvc("status", "--cloud", "--remote", _REMOTO_TEMPORAL, "--json"))
        finally:
            self._dvc("remote", "remove", "--local", _REMOTO_TEMPORAL, json_=False)

    def _dvc(self, *argumentos: str, json_: bool = True) -> dict:
        resultado = self._ejecutar(
            [sys.executable, "-m", "dvc", *argumentos],
            capture_output=True,
            text=True,
            cwd=self._raiz_repo,
            check=False,
        )
        if resultado.returncode != 0:
            if "No module named dvc" in resultado.stderr:
                raise FaltaHerramienta("falta dvc en el entorno: uv sync")
            ultima = (resultado.stderr.strip().splitlines() or ["sin detalle"])[-1]
            raise RechazoDominio(f"dvc {argumentos[0]} falló: {ultima}")
        return json.loads(resultado.stdout or "{}") if json_ else {}
