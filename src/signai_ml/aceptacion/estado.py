"""Orden `estado` (contracts/cli.md; SC-008): de dónde sale el modelo, sin tocar nada.

Es solo lectura y nunca lee la prueba. Muestra:
- el candidato y su estado;
- las consultas a la prueba, con el total que hay que declarar (ADR-0011);
- la versión con alias `aceptado` y su paridad;
- la última exportación;
- si `modelos/reconocedor` y `modelos/coreml` coinciden con sus punteros de DVC.
"""

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from . import ALIAS, CANDIDATO, COREML, MODELO_REGISTRADO
from .candidato import leer
from .consultas import total
from .errores import ErrorAceptacion
from .formato import miles

if TYPE_CHECKING:  # pragma: no cover
    from .cli import Contexto

OBJETIVOS_DVC = (CANDIDATO.as_posix(), COREML.as_posix())


def estado_dvc(raiz: Path, objetivos: Sequence[str]) -> dict[str, str]:
    """Por objetivo: `al día`, `con cambios`, `sin versionar` o `desconocido` (si `dvc status` falla)."""
    raiz = Path(raiz)
    resultado = {o: "sin versionar" for o in objetivos if not (raiz / f"{o}.dvc").is_file()}
    versionados = [o for o in objetivos if o not in resultado]
    if versionados:
        salida = subprocess.run(
            [sys.executable, "-m", "dvc", "status", "--json", *(f"{o}.dvc" for o in versionados)],
            cwd=raiz, capture_output=True, text=True,
        )
        try:
            cambios = json.loads(salida.stdout) if salida.returncode == 0 else None
        except json.JSONDecodeError:
            cambios = None
        for objetivo in versionados:
            if cambios is None:
                resultado[objetivo] = "desconocido"
            else:
                resultado[objetivo] = "con cambios" if f"{objetivo}.dvc" in cambios else "al día"
    return {o: resultado[o] for o in objetivos}


def _linea_candidato(ctx: "Contexto") -> str:
    try:
        candidato = leer(ctx.raiz)
    except ErrorAceptacion as error:
        return f"Candidato: no válido ({error})"
    ficha = candidato.ficha
    return (
        f"Candidato: {ficha.get('barrido')} / {ficha.get('configuracion')} / semilla {ficha.get('semilla')} · "
        f"modelo {candidato.modelo_sha256[:8]}… · estado {ficha.get('estado')}"
    )


def cmd_estado(ctx: "Contexto") -> int:
    print(_linea_candidato(ctx))
    evaluaciones = sorted(ctx.registro.evaluaciones(), key=lambda e: (e.consulta, e.run_id))
    print(f"Consultas a la prueba: {total(evaluaciones)} en total" if evaluaciones else "Consultas a la prueba: 0")
    for e in evaluaciones:
        recuento = (
            f" · {miles(int(e.metricas['aciertos']))} / {miles(int(e.metricas['total']))}"
            if "aciertos" in e.metricas else ""
        )
        decision = f" · {e.decision}" if e.decision else ""
        print(f"  consulta-{e.consulta:02d} · {e.estado}{decision}{recuento}")
    version = ctx.registro.version_con_alias()
    if version is None:
        print(f"Versión con alias {ALIAS}: ninguna")
    else:
        paridad = version.etiquetas.get("paridad", "sin demostrar")
        print(
            f"Versión con alias {ALIAS}: {version.version} de {MODELO_REGISTRADO} · "
            f"consulta {version.etiquetas.get('consulta')} · paridad {paridad}"
        )
    exportaciones = sorted(ctx.registro.exportaciones(), key=lambda e: e.nombre)
    if exportaciones:
        ultima = exportaciones[-1]
        print(f"Última exportación: {ultima.nombre} · paridad {ultima.paridad or ultima.estado}")
    else:
        print("Última exportación: ninguna")
    dvc = ctx.estado_dvc(ctx.raiz, list(OBJETIVOS_DVC))
    print("DVC: " + " · ".join(f"{objetivo} {situacion}" for objetivo, situacion in dvc.items()))
    return 0
