"""Huella SHA-256 de un fichero, por bloques (las ventanas ocupan ≈260 MB)."""

import hashlib
from pathlib import Path


def sha256_fichero(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with Path(ruta).open("rb") as f:
        while bloque := f.read(1 << 20):
            resumen.update(bloque)
    return resumen.hexdigest()
