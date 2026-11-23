"""Aprobación del lote de validación (FR-021, SC-005; research R-13).

Solo la registra el responsable con `signai-preproceso aprobar-lote`. El corpus completo no se
procesa sin una aprobación «aprobado» con las versiones vigentes y el informe que se revisó.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .almacen import Versiones, escribir_texto
from .errores import ErrorUso, FaltaAprobacion

FICHERO = "aprobacion.json"
ANTERIORES = "aprobaciones-anteriores"
DECISIONES = ("aprobado", "rechazado")


def leer(directorio: Path) -> dict | None:
    ruta = directorio / FICHERO
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else None


def invalidar(directorio: Path) -> bool:
    """Mueve la decisión vigente a `aprobaciones-anteriores/`. Devuelve si había alguna."""
    ruta = directorio / FICHERO
    if not ruta.is_file():
        return False
    datos = leer(directorio) or {}
    destino = directorio / ANTERIORES / f"{datos.get('fecha', 'sin-fecha').replace(':', '-')}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    ruta.replace(destino)
    return True


def registrar(
    directorio: Path, decision: str, nota: str, versiones: Versiones, semilla_lote: int,
    sha256_informe: str, ahora: datetime | None = None,
) -> dict:
    if decision not in DECISIONES:
        raise ErrorUso(f"decisión desconocida: {decision}")
    if decision == "rechazado" and not nota.strip():
        raise ErrorUso("Para rechazar el lote hace falta una nota: `aprobar-lote --rechazar --nota \"…\"`.")
    invalidar(directorio)
    datos = {
        "decision": decision,
        "fecha": (ahora or datetime.now()).astimezone().isoformat(timespec="seconds"),
        "nota": nota, **asdict(versiones), "semilla_lote": semilla_lote, "sha256_informe": sha256_informe,
    }
    escribir_texto(directorio / FICHERO, json.dumps(datos, ensure_ascii=False, indent=2))
    return datos


def _motivo(directorio: Path, versiones: Versiones, sha256_informe: str) -> str | None:
    datos = leer(directorio)
    if datos is None:
        return "no hay aprobación del lote: procesa el lote, revisa sus vídeos y regístrala con `aprobar-lote`"
    if datos["decision"] != "aprobado":
        return f"el lote está rechazado ({datos.get('nota', '')}): corrige, vuelve a procesarlo y apruébalo"
    for clave, valor in asdict(versiones).items():
        if datos.get(clave) != valor:
            return f"la aprobación es de otra versión ({clave}: {datos.get(clave)} ≠ {valor}): vuelve a validar el lote"
    if datos.get("sha256_informe") != sha256_informe:
        return "la aprobación es de otro informe del lote: vuelve a revisarlo y aprobarlo"
    return None


def vigente(directorio: Path, versiones: Versiones, sha256_informe: str) -> bool:
    return _motivo(directorio, versiones, sha256_informe) is None


def exigir(directorio: Path, versiones: Versiones, sha256_informe: str) -> dict:
    motivo = _motivo(directorio, versiones, sha256_informe)
    if motivo is not None:
        raise FaltaAprobacion(f"No se procesa el corpus completo: {motivo}.")
    return leer(directorio) or {}
