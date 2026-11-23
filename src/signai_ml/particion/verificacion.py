"""Comprobaciones de la partición versión 2 (TT-22; contracts/lectura.md, precondiciones 1–6).

`comprobar_ligadura` es la precondición de toda lectura: la partición es la versión 2, es la registrada
y corresponde exactamente al corpus procesado del que salió, clip a clip y grabación a grabación. No
recalcula la huella de `ventanas.npy` (≈260 MB): la garantiza el manifiesto de TT-12.

`verificar` es la auditoría completa (`signai-particion verificar`, también en un clon limpio): las
mismas comprobaciones más el SHA-256 real de las ventanas, las garantías de grabación (ninguna en dos
lados, una por clase en prueba y al menos una en cada pliegue) y el re-sorteo clip a clip. No se
detiene en el primer fallo, sino que informa de todas, y no escribe nada.
"""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import CLASES_ESPERADAS, VERSION_ANTERIOR, VERSION_PARTICION, Lado
from .almacen import (
    CSV,
    MANIFIESTO,
    PUNTERO,
    AsignacionClip,
    ParticionLeida,
    leer_manifiesto,
    leer_particion,
    miles,
    orden_clase,
    recuentos,
    ruta_particion,
)
from .errores import RESTAURAR, ErrorParticion, ParticionAusente, ParticionNoValida
from .huella import sha256_fichero
from .indice import dir_ventanas, leer_corpus
from .sorteo import sortear

MIGRAR = "la partición es la versión 1 (por clip); genera la versión 2: `signai-particion generar`."

MAX_EJEMPLOS = 5


def _ausente(raiz: Path) -> ParticionAusente:
    if (raiz / PUNTERO).exists():
        return ParticionAusente(
            f"La partición está versionada pero no descargada: `uv run dvc checkout {PUNTERO}`."
        )
    return ParticionAusente("No hay partición: genérala una sola vez con `uv run signai-particion generar`.")


def _no_valida(motivo: str) -> ParticionNoValida:
    return ParticionNoValida(f"{motivo} Restaura los datos versionados con `{RESTAURAR}`.")


def comprobar_version(manifiesto: dict) -> None:
    """Precondición 5: solo se lee la versión 2; la 1 (por clip) se rechaza con la orden de migrar."""
    version = manifiesto.get("version_particion")
    if version == VERSION_ANTERIOR:
        raise ParticionNoValida(f"No se puede usar: {MIGRAR}")
    if version != VERSION_PARTICION:
        raise ParticionNoValida(
            f"La partición dice ser la versión {version!r} y se esperaba {VERSION_PARTICION!r}. "
            f"Restaura los datos versionados con `{RESTAURAR}`."
        )


def leer_existente(raiz: Path) -> ParticionLeida:
    base = ruta_particion(raiz)
    if not (base / MANIFIESTO).is_file() or not (base / CSV).is_file():
        raise _ausente(Path(raiz))
    comprobar_version(leer_manifiesto(base))
    return leer_particion(base)


def _corto(huella: str | None) -> str:
    return f"{huella[:12]}…" if huella else "ninguna"


def _ejemplos(elementos: Sequence[str]) -> str:
    resto = len(elementos) - MAX_EJEMPLOS
    return ", ".join(elementos[:MAX_EJEMPLOS]) + (f" y {resto} más" if resto > 0 else "")


# ── Comprobaciones: cada una devuelve el motivo del fallo, o None si pasa ──


def _motivo_huella(raiz: Path, manifiesto: dict) -> str | None:
    obtenida = sha256_fichero(ruta_particion(raiz) / CSV)
    if obtenida != manifiesto["huella"]:
        return (
            f"La huella de datos/particion/particion.csv no es la de su manifiesto "
            f"(esperada {_corto(manifiesto['huella'])}, obtenida {_corto(obtenida)})."
        )
    return None


def _sha256_ventanas_del_manifiesto(raiz: Path) -> str | None:
    ruta = dir_ventanas(raiz) / "manifiesto.json"
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding="utf-8")).get("sha256_ventanas")


def _motivo_origen(raiz: Path, origen: dict) -> str | None:
    """El corpus procesado es el de origen según su manifiesto y la huella de `indice.csv`."""
    declarada = _sha256_ventanas_del_manifiesto(raiz)
    if declarada != origen["sha256_ventanas"]:
        return (
            "El corpus procesado no es el de origen: la huella de las ventanas de datos/ventanas/manifiesto.json "
            f"no es la registrada en la partición (esperada {_corto(origen['sha256_ventanas'])}, "
            f"obtenida {_corto(declarada)})."
        )
    ruta_indice = dir_ventanas(raiz) / "indice.csv"
    obtenida = sha256_fichero(ruta_indice) if ruta_indice.is_file() else None
    if obtenida != origen["sha256_indice"]:
        return (
            "La huella de datos/ventanas/indice.csv no es la registrada en la partición "
            f"(esperada {_corto(origen['sha256_indice'])}, obtenida {_corto(obtenida)})."
        )
    return None


def _motivo_ventanas_reales(raiz: Path, origen: dict) -> str | None:
    """Solo `verificar`: el SHA-256 real de `ventanas.npy`, no el que dice su manifiesto."""
    ruta = dir_ventanas(raiz) / "ventanas.npy"
    obtenida = sha256_fichero(ruta) if ruta.is_file() else None
    if obtenida != origen["sha256_ventanas"]:
        return (
            "La huella real de datos/ventanas/ventanas.npy no es la registrada en la partición "
            f"(esperada {_corto(origen['sha256_ventanas'])}, obtenida {_corto(obtenida)})."
        )
    return None


def _motivo_filas(asignaciones: Sequence[AsignacionClip], indice: Sequence[dict[str, str]]) -> str | None:
    if len(indice) != len(asignaciones):
        return f"La partición tiene {len(asignaciones)} filas y el índice {len(indice)}."
    for asignacion, fila in zip(asignaciones, indice, strict=True):
        esperada = (fila["fila"], fila["clip_id"], fila["clase"], fila["grabacion"])
        obtenida = (str(asignacion.fila), asignacion.clip_id, asignacion.clase, asignacion.grabacion)
        if obtenida != esperada:
            return (
                f"La fila {asignacion.fila} de particion.csv es {asignacion.clip_id} ({asignacion.clase}, "
                f"{asignacion.grabacion}) y en indice.csv la fila {fila['fila']} es {fila['clip_id']} "
                f"({fila['clase']}, {fila['grabacion']})."
            )
    return None


def comprobar_ligadura(raiz: Path) -> ParticionLeida:
    raiz = Path(raiz)
    leida = leer_existente(raiz)
    manifiesto = leida.manifiesto
    for motivo in (
        lambda: _motivo_huella(raiz, manifiesto),
        lambda: _motivo_origen(raiz, manifiesto["origen"]),
        lambda: _motivo_filas(leida.asignaciones, leer_corpus(raiz).filas),
    ):
        if fallo := motivo():
            raise _no_valida(fallo)
    return leida


# ── verificar (R-08) ──


@dataclass(frozen=True)
class Comprobacion:
    nombre: str
    ok: bool
    detalle: str


@dataclass(frozen=True)
class InformeVerificacion:
    comprobaciones: tuple[Comprobacion, ...]
    totales: dict[Lado, int]
    huella: str
    manifiesto: dict

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.comprobaciones)

    def comprobacion(self, nombre: str) -> Comprobacion:
        return next(c for c in self.comprobaciones if c.nombre == nombre)


def _comprobacion(nombre: str, motivo: str | None, si_pasa: str) -> Comprobacion:
    return Comprobacion(nombre, motivo is None, motivo or si_pasa)


def _motivo_grabaciones(
    asignaciones: Sequence[AsignacionClip], indice: Sequence[dict[str, str]] | None, manifiesto: dict
) -> str | None:
    repetidos = sorted(c for c, veces in Counter(a.clip_id for a in asignaciones).items() if veces > 1)
    if repetidos:
        return f"Hay clips repetidos: {_ejemplos(repetidos)}."

    lados_de: dict[tuple[str, str], set[Lado]] = {}
    for a in asignaciones:
        lados_de.setdefault((a.clase, a.grabacion), set()).add(a.lado)
    partidas = sorted(f"{clase}/{grabacion}" for (clase, grabacion), lados in lados_de.items() if len(lados) > 1)
    if partidas:
        return f"Hay grabaciones partidas entre dos lados: {_ejemplos(partidas)}."

    por_clase: dict[str, Counter] = {}
    for (clase, _), lados in lados_de.items():
        por_clase.setdefault(clase, Counter()).update(lados)
    if len(por_clase) != CLASES_ESPERADAS:
        return f"La partición tiene {len(por_clase)} clases y deben ser {CLASES_ESPERADAS}."
    fuera = [
        f"{clase} ({c[Lado.PRUEBA]} en prueba, {c[Lado.PLIEGUE_A]} en A, {c[Lado.PLIEGUE_B]} en B)"
        for clase, c in sorted(por_clase.items(), key=lambda kv: orden_clase(kv[0]))
        if c[Lado.PRUEBA] != 1 or c[Lado.PLIEGUE_A] < 1 or c[Lado.PLIEGUE_B] < 1
    ]
    if fuera:
        return f"Clases sin una grabación en prueba y al menos una en cada pliegue: {_ejemplos(fuera)}."

    if indice is not None:
        en_particion, en_indice = {a.clip_id for a in asignaciones}, {f["clip_id"] for f in indice}
        if en_particion != en_indice:
            return (
                f"La unión de los lados no es el índice: faltan {len(en_indice - en_particion)} clip(s) y "
                f"sobran {len(en_particion - en_indice)}."
            )
    if recuentos((a.clase, a.grabacion, a.lado) for a in asignaciones) != manifiesto["recuentos"]:
        return "Los recuentos de particion.csv no son los de su manifiesto."
    return None


def _re_sorteo(asignaciones: Sequence[AsignacionClip], semilla: int) -> Comprobacion:
    n = len(asignaciones)
    try:
        esperado = sortear(
            ({"clip_id": a.clip_id, "clase": a.clase, "grabacion": a.grabacion} for a in asignaciones), semilla
        )
    except ErrorParticion as fallo:
        return Comprobacion("re-sorteo", False, f"No se puede repetir el sorteo: {fallo}")
    distintos = [
        f"{a.clip_id} en {a.lado.value} (le toca {esperado[a.clip_id].value})"
        for a in asignaciones
        if esperado[a.clip_id] is not a.lado
    ]
    if distintos:
        return Comprobacion(
            "re-sorteo",
            False,
            f"Con la semilla {semilla}, {len(distintos)} de {n} clips no están en su lado: {_ejemplos(distintos)}.",
        )
    return Comprobacion("re-sorteo", True, f"re-sorteo idéntico ({n}/{n}) con la semilla {semilla}")


def verificar(raiz: Path) -> InformeVerificacion:
    """Las seis comprobaciones sobre la partición de `raiz`. Lanza `ParticionAusente` si no hay y
    `ParticionNoValida` si es la versión 1."""
    raiz = Path(raiz)
    leida = leer_existente(raiz)
    manifiesto, origen, asignaciones = leida.manifiesto, leida.manifiesto["origen"], leida.asignaciones

    try:
        indice: tuple[dict[str, str], ...] | None = leer_corpus(raiz).filas
        motivo_filas = _motivo_filas(asignaciones, indice)
    except ErrorParticion as fallo:
        indice, motivo_filas = None, str(fallo)

    comprobaciones = (
        Comprobacion("versión", True, f"{VERSION_PARTICION} (por grabación entera), semilla {manifiesto['semilla']}"),
        _comprobacion(
            "huella", _motivo_huella(raiz, manifiesto), "particion.csv tiene la huella de su manifiesto"
        ),
        _comprobacion(
            "corpus de origen",
            _motivo_origen(raiz, origen) or _motivo_ventanas_reales(raiz, origen),
            f"ventanas.npy e indice.csv son los registrados ({miles(origen['ventanas'])} ventanas)",
        ),
        _comprobacion("filas = índice", motivo_filas, f"las {miles(len(asignaciones))} filas son las del índice"),
        _comprobacion(
            "grabaciones y disjunción",
            _motivo_grabaciones(asignaciones, indice, manifiesto),
            f"{len({(a.clase, a.grabacion) for a in asignaciones})} grabaciones enteras, una por clase en prueba "
            "y al menos una en cada pliegue; la unión es el índice",
        ),
        _re_sorteo(asignaciones, manifiesto["semilla"]),
    )
    total = Counter(a.lado for a in asignaciones)
    return InformeVerificacion(
        comprobaciones=comprobaciones,
        totales={lado: total[lado] for lado in Lado},
        huella=manifiesto["huella"],
        manifiesto=manifiesto,
    )
