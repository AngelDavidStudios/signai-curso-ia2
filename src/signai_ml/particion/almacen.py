"""Dónde y cómo se guarda la partición versión 2 (TT-22; data-model «Partición versión 2»).

    datos/particion/particion.csv     fila,clip_id,clase,grabacion,lado — en el orden de ventanas.npy
    datos/particion/manifiesto.json   versión, semilla, reglas, origen, recuentos, fecha y huella
    datos/particion/informe.md        qué grabación de cada clase va a cada lado, legible

Los tres ficheros se escriben en un directorio temporal que después se renombra: un corte no deja
una partición a medias. Al sustituir la versión 1, la anterior se aparta, se pone la nueva y solo
entonces se borra la anterior (sigue en el historial de DVC). La huella es el SHA-256 de
`particion.csv`, que por eso tiene un formato canónico.
"""

import csv
import io
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import VERSION_PARTICION, Lado
from .errores import ErrorParticion
from .huella import sha256_fichero
from .indice import CorpusProcesado

CSV = "particion.csv"
MANIFIESTO = "manifiesto.json"
INFORME = "informe.md"
COLUMNAS = ("fila", "clip_id", "clase", "grabacion", "lado")
REGLAS = {
    "unidad": "(clase, grabacion)",
    "orden_sorteo": "sha256(semilla|clase|grabacion)",
    "reparto": "1.ª prueba, 2.ª pliegue_a, 3.ª pliegue_b; más de 3: alternando pliegue_a y pliegue_b",
}
PUNTERO = Path("datos") / "particion.dvc"
NOMBRE_LADO = {Lado.PRUEBA: "prueba", Lado.PLIEGUE_A: "pliegue A", Lado.PLIEGUE_B: "pliegue B"}


@dataclass(frozen=True)
class AsignacionClip:
    fila: int
    clip_id: str
    clase: str
    grabacion: str
    lado: Lado


@dataclass(frozen=True)
class ParticionLeida:
    asignaciones: tuple[AsignacionClip, ...]
    manifiesto: dict


def ruta_particion(raiz: Path) -> Path:
    return Path(raiz) / "datos" / "particion"


def miles(n: int) -> str:
    """6396 → «6.396», como se escriben las cifras en la documentación del proyecto."""
    return f"{n:,}".replace(",", ".")


def orden_clase(clase: str) -> tuple[int, str]:
    numero, _, _ = clase.partition("-")
    return (int(numero) if numero.isdigit() else 0, clase)


def linea_totales(total: Mapping[str, int]) -> str:
    """«2.998 prueba · 2.999 pliegue A · 2.999 pliegue B»."""
    return " · ".join(f"{miles(total[lado.value])} {NOMBRE_LADO[lado]}" for lado in Lado)


def _texto_csv(corpus: CorpusProcesado, asignaciones: Mapping[str, Lado]) -> str:
    salida = io.StringIO()
    escritor = csv.writer(salida, lineterminator="\n")
    escritor.writerow(COLUMNAS)
    for fila in corpus.filas:
        escritor.writerow(
            (fila["fila"], fila["clip_id"], fila["clase"], fila["grabacion"], asignaciones[fila["clip_id"]].value)
        )
    return salida.getvalue()


def recuentos(clips: Iterable[tuple[str, str, Lado]]) -> dict:
    """Los recuentos del manifiesto a partir de `(clase, grabacion, lado)` de cada clip.

    `total`: clips por lado. `por_clase`: por clase y lado, sus grabaciones (ordenadas) y sus clips.
    """
    grabaciones: dict[str, dict[Lado, set[str]]] = defaultdict(lambda: defaultdict(set))
    clips_por: dict[str, dict[Lado, int]] = defaultdict(lambda: defaultdict(int))
    total: dict[Lado, int] = defaultdict(int)
    for clase, grabacion, lado in clips:
        grabaciones[clase][lado].add(grabacion)
        clips_por[clase][lado] += 1
        total[lado] += 1
    return {
        "total": {lado.value: total[lado] for lado in Lado},
        "por_clase": {
            clase: {
                lado.value: {"grabaciones": sorted(grabaciones[clase][lado]), "clips": clips_por[clase][lado]}
                for lado in Lado
            }
            for clase in sorted(grabaciones, key=orden_clase)
        },
    }


def informe_markdown(manifiesto: dict) -> str:
    total = manifiesto["recuentos"]["total"]
    por_clase = manifiesto["recuentos"]["por_clase"]
    lineas = [
        "# Partición de evaluación por grabación (TT-22)",
        "",
        f"- Versión: `{manifiesto['version_particion']}` · semilla {manifiesto['semilla']} · {manifiesto['fecha']}",
        f"- Huella de `particion.csv`: `{manifiesto['huella']}`",
        f"- Corpus de origen: {miles(manifiesto['origen']['ventanas'])} ventanas, "
        f"`sha256_ventanas` {manifiesto['origen']['sha256_ventanas']}",
        f"- **Total**: {linea_totales(total)}",
        "- Cada grabación va entera a un lado; la prueba es una grabación de cada clase y los pliegues A y B "
        "son las dos rotaciones de la selección.",
        "",
        "| Clase | Prueba | Pliegue A | Pliegue B |",
        "|---|---|---|---|",
    ]

    def celda(lado: dict) -> str:
        return f"{', '.join(lado['grabaciones'])} ({lado['clips']})"

    for clase, lados in por_clase.items():
        lineas.append(f"| {clase} | " + " | ".join(celda(lados[lado.value]) for lado in Lado) + " |")
    return "\n".join(lineas) + "\n"


def escribir_particion(
    base: Path,
    asignaciones: Mapping[str, Lado],
    corpus: CorpusProcesado,
    semilla: int,
    sustituir: bool = False,
) -> dict:
    """Escribe los tres ficheros de golpe y devuelve el manifiesto.

    Sin `sustituir`, se niega si `base` ya existe; con él, sustituye la partición anterior sin dejar un
    momento en el que no haya ninguna completa.
    """
    base = Path(base)
    if base.exists() and not sustituir:
        raise ErrorParticion(f"{base} ya existe: no se sobrescribe una partición (research R-03).")
    texto_csv = _texto_csv(corpus, asignaciones)
    cuentas = recuentos((f["clase"], f["grabacion"], asignaciones[f["clip_id"]]) for f in corpus.filas)
    manifiesto = {
        "version_particion": VERSION_PARTICION,
        "semilla": semilla,
        "reglas": REGLAS,
        "origen": {
            "sha256_ventanas": corpus.sha256_ventanas,
            "sha256_indice": corpus.sha256_indice,
            "ventanas": corpus.n,
        },
        "recuentos": cuentas,
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "huella": None,
    }

    temporal = base.with_name(f".{base.name}.tmp-{os.getpid()}")
    anterior = base.with_name(f".{base.name}.anterior-{os.getpid()}")
    try:
        temporal.mkdir(parents=True)
        (temporal / CSV).write_text(texto_csv, encoding="utf-8", newline="")
        manifiesto["huella"] = sha256_fichero(temporal / CSV)
        (temporal / INFORME).write_text(informe_markdown(manifiesto), encoding="utf-8")
        (temporal / MANIFIESTO).write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if base.exists():
            os.rename(base, anterior)
        os.rename(temporal, base)
    except BaseException:
        if anterior.exists() and not base.exists():
            os.rename(anterior, base)
        raise
    finally:
        shutil.rmtree(temporal, ignore_errors=True)
        shutil.rmtree(anterior, ignore_errors=True)
    return manifiesto


def leer_manifiesto(base: Path) -> dict:
    return json.loads((Path(base) / MANIFIESTO).read_text(encoding="utf-8"))


def leer_particion(base: Path) -> ParticionLeida:
    base = Path(base)
    manifiesto = leer_manifiesto(base)
    with (base / CSV).open(encoding="utf-8", newline="") as f:
        asignaciones = tuple(
            AsignacionClip(int(fila["fila"]), fila["clip_id"], fila["clase"], fila["grabacion"], Lado(fila["lado"]))
            for fila in csv.DictReader(f)
        )
    return ParticionLeida(asignaciones, manifiesto)
