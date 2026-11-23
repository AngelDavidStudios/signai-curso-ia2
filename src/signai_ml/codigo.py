"""Versión del código que se registra con cada corrida, y la guarda de árbol limpio.

Común al entrenamiento (TT-14) y a la aceptación (TT-15): si hay cambios sin confirmar, el commit
registrado no describiría lo que se ejecutó (constitución VI). Cada paquete pasa su propio error para
conservar su código de salida.
"""

import subprocess
from pathlib import Path


def estado_codigo(raiz: Path) -> tuple[str, bool]:
    """`(commit, sucio)`. Fuera de un repositorio Git: `("desconocido", True)`."""
    def git(*argumentos: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(raiz), *argumentos], capture_output=True, text=True)

    commit = git("rev-parse", "HEAD")
    if commit.returncode != 0:
        return "desconocido", True
    return commit.stdout.strip(), bool(git("status", "--porcelain").stdout.strip())


def codigo_permitido(
    raiz: Path, permitir_sucio: bool, error: type[Exception], que_hacer: str
) -> tuple[str, bool]:
    """El commit que se registra; sin cambios sin confirmar, salvo `permitir_sucio` (solo pruebas)."""
    commit, sucio = estado_codigo(raiz)
    if sucio and not permitir_sucio:
        raise error(
            f"Hay cambios sin confirmar en {raiz} (o no es un repositorio Git): la versión del código registrada no "
            f"describiría lo que se ejecuta. {que_hacer}"
        )
    return commit, sucio
