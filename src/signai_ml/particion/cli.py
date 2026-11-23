"""CLI `signai-particion` (TT-22; contracts/cli.md P-1', P-2'): generar una sola vez y verificar.

    signai-particion generar
    signai-particion verificar

`generar` escribe la partición versión 2 (por grabación entera). Si encuentra la versión 1 de TT-13 la
sustituye, una sola vez: la anterior sigue en el historial de DVC (research R-03). Con la versión 2 ya
fijada, se niega. Los datos se resuelven desde la raíz del repositorio (`datos/`). Código de salida: 0
bien, 1 error de los datos o del proceso (incluido «ya está fijada»), 2 uso incorrecto. No hay opciones
de semilla ni de forzado: la CLI tampoco toca Git ni DVC, los lanza la persona.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import SEMILLA_PARTICION, VERSION_ANTERIOR, VERSION_PARTICION
from .almacen import MANIFIESTO, PUNTERO, escribir_particion, leer_manifiesto, linea_totales, miles, ruta_particion
from .errores import RESTAURAR, ErrorParticion, ParticionFijada
from .indice import leer_corpus
from .sorteo import sortear
from .verificacion import verificar

VERSIONAR = (
    "Versiónala ahora:\n"
    "  uv run dvc add datos/particion && uv run dvc push\n"
    "  git add datos/particion.dvc datos/.gitignore\n"
    "  uv run signai-copia actualizar"
)


def _raiz_repo() -> Path:
    return Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signai-particion",
        description="Partición de evaluación por grabación entera (TT-22): se genera una vez y se verifica.",
    )
    ordenes = parser.add_subparsers(dest="orden", required=True)
    ordenes.add_parser("generar", help="Reparte el corpus procesado; se niega si ya está fijada.")
    ordenes.add_parser("verificar", help="Comprueba huellas, recuentos y re-sorteo sin escribir nada.")
    return parser


def _version_existente(raiz: Path) -> str | None:
    """La versión de la partición en disco; `None` si no hay. Con el puntero sin datos, se niega."""
    base = ruta_particion(raiz)
    if (base / MANIFIESTO).exists():
        return leer_manifiesto(base).get("version_particion")
    if (raiz / PUNTERO).exists():
        raise ParticionFijada(
            f"Hay una partición versionada ({PUNTERO}) pero no está descargada, así que no se sabe qué versión "
            f"es. Descárgala primero: uv run dvc checkout {PUNTERO}"
        )
    return None


def cmd_generar(raiz: Path) -> int:
    base = ruta_particion(raiz)
    version = _version_existente(raiz)
    if version is not None and version != VERSION_ANTERIOR:
        raise ParticionFijada(
            f"La partición ya está fijada ({base.relative_to(raiz) / MANIFIESTO}, {version}). No se rehace entre "
            f"experimentos (ADR-0011).\n{VERSIONAR}"
        )
    corpus = leer_corpus(raiz)
    manifiesto = escribir_particion(
        base, sortear(corpus.filas, SEMILLA_PARTICION), corpus, SEMILLA_PARTICION, sustituir=version is not None
    )
    por_clase = manifiesto["recuentos"]["por_clase"]
    grabaciones = sum(len(lado["grabaciones"]) for lados in por_clase.values() for lado in lados.values())
    if version == VERSION_ANTERIOR:
        print(f"La partición {VERSION_PARTICION} sustituye a la versión 1 (por clip), que sigue en el historial de DVC.")
    print(
        f"Partición fijada por grabación: {linea_totales(manifiesto['recuentos']['total'])} "
        f"({miles(corpus.n)} ventanas, {len(por_clase)} clases, {grabaciones} grabaciones)."
    )
    print(f"Huella: {manifiesto['huella']}")
    print(VERSIONAR)
    return 0


def cmd_verificar(raiz: Path) -> int:
    informe = verificar(raiz)
    for comprobacion in informe.comprobaciones:
        print(f"{'✓' if comprobacion.ok else '✗'} {comprobacion.nombre}: {comprobacion.detalle}")
    if not informe.ok:
        print(f"La partición no supera la verificación. Restaura los datos versionados con `{RESTAURAR}`.")
        return 1
    print(f"Totales: {linea_totales({lado.value: n for lado, n in informe.totales.items()})}.")
    print(f"Huella: {informe.huella}")
    return 0


def main(argv: Sequence[str] | None = None, raiz: Path | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as salida:
        return int(salida.code or 0)
    raiz = Path(raiz) if raiz is not None else _raiz_repo()
    try:
        if args.orden == "generar":
            return cmd_generar(raiz)
        return cmd_verificar(raiz)
    except ErrorParticion as fallo:
        print(f"Error: {fallo}")
        return fallo.codigo_salida
