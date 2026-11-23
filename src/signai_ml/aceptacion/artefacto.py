"""Metadatos y validación de `KPoseFormer.mlpackage` (contracts/artefacto.md; research R-07).

Los metadatos se escriben editando solo la especificación del paquete
(`Data/com.apple.CoreML/model.mlmodel`), sin tocar los pesos, igual que se hizo con el artefacto
provisional de la app. `validar` replica lo que comprueba `ModelArtifactDescription` en `signai-ios`
al cargar el modelo: si la app lo rechazaría, aquí se rechaza antes de guardarlo.
"""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from coremltools.proto import Model_pb2

from ..seguimiento import versiones_bibliotecas
from . import ENTRADA, SALIDA
from .conversion import FORMA_ENTRADA, OBJETIVO
from .errores import ArtefactoNoValido

METADATOS = (
    "clases", "glosas", "cifra", "modelo.version", "modelo.sha256", "aceptacion.run_id", "particion.huella",
)
FLOAT16 = Model_pb2.ArrayFeatureType.FLOAT16
ESPECIFICACION = Path("Data") / "com.apple.CoreML" / "model.mlmodel"


def cifra(top1: float, consulta: int) -> str:
    """Qué mide la cifra que viaja en el artefacto: no es una cifra frente a un señante no visto (ADR-0011)."""
    return f"{top1:.3f} · prueba aleatoria 1.800 · consulta {consulta}"


def _leer_spec(paquete: Path) -> Model_pb2.Model:
    ruta = Path(paquete) / ESPECIFICACION
    if not ruta.is_file():
        raise ArtefactoNoValido(f"{paquete} no es un paquete de Core ML: falta {ESPECIFICACION}.")
    spec = Model_pb2.Model()
    spec.ParseFromString(ruta.read_bytes())
    return spec


def leer_metadatos(paquete: Path) -> dict[str, str]:
    return dict(_leer_spec(paquete).description.metadata.userDefined)


def escribir_metadatos(paquete: Path, metadatos: Mapping[str, str]) -> None:
    faltan = [clave for clave in METADATOS if not str(metadatos.get(clave, ""))]
    if faltan:
        raise ArtefactoNoValido(f"Faltan los metadatos {', '.join(faltan)}.")
    spec = _leer_spec(paquete)
    descripcion = spec.description.metadata
    for clave in METADATOS:
        descripcion.userDefined[clave] = str(metadatos[clave])
    descripcion.shortDescription = (
        f"KPoseFormer · {len(metadatos['clases'].split(','))} clases de LSEC · entrada (1,75,96) normalizada · "
        f"Top-1 de prueba {metadatos['cifra']}"
    )
    (Path(paquete) / ESPECIFICACION).write_bytes(spec.SerializeToString())


def unir(elementos: Sequence[str], nombre: str) -> str:
    """Lista separada por comas, como la lee la app; una coma dentro de un elemento la rompería."""
    con_coma = [e for e in elementos if "," in e]
    if con_coma:
        raise ArtefactoNoValido(f"Hay {nombre} con coma ({con_coma[0]!r}): la app las separa por comas.")
    return ",".join(elementos)


def _forma(tipo) -> list[int]:
    return [int(d) for d in tipo.multiArrayType.shape]


def validar(paquete: Path, clases: Sequence[str], glosas: Sequence[str]) -> None:
    """Las comprobaciones de `ModelArtifactDescription` (contracts/artefacto.md A-3)."""
    spec = _leer_spec(paquete)
    entradas = {e.name: e.type for e in spec.description.input}
    salidas = {s.name: s.type for s in spec.description.output}
    if ENTRADA not in entradas or _forma(entradas[ENTRADA]) != list(FORMA_ENTRADA):
        raise ArtefactoNoValido(f"La entrada «{ENTRADA}» no tiene la forma {list(FORMA_ENTRADA)}.")
    if SALIDA not in salidas:
        raise ArtefactoNoValido(f"Falta la salida «{SALIDA}».")
    numero_salidas = 1
    for dimension in _forma(salidas[SALIDA]):
        numero_salidas *= dimension
    if numero_salidas != len(clases):
        raise ArtefactoNoValido(f"El artefacto da {numero_salidas} salidas y hay {len(clases)} clases.")
    for nombre, tipo in ((ENTRADA, entradas[ENTRADA]), (SALIDA, salidas[SALIDA])):
        if tipo.multiArrayType.dataType != FLOAT16:
            raise ArtefactoNoValido(f"«{nombre}» no está declarada en FLOAT16 (ADR-0013, ADR-0014).")
    metadatos = dict(spec.description.metadata.userDefined)
    if metadatos.get("clases", "").split(",") != list(clases):
        raise ArtefactoNoValido("Los metadatos `clases` no son las clases del modelo, en número u orden.")
    leidas = metadatos.get("glosas", "").split(",")
    if leidas != list(glosas):
        raise ArtefactoNoValido("Los metadatos `glosas` no son las glosas del modelo, en número u orden.")
    if any(not glosa.strip() for glosa in leidas):
        raise ArtefactoNoValido("Hay una glosa vacía: la app no la aceptaría.")


def huella_paquete(paquete: Path) -> str:
    """SHA-256 de todos los ficheros del paquete, en orden de ruta, con su ruta relativa."""
    resumen = hashlib.sha256()
    for fichero in sorted(p for p in Path(paquete).rglob("*") if p.is_file()):
        resumen.update(fichero.relative_to(paquete).as_posix().encode())
        resumen.update(hashlib.sha256(fichero.read_bytes()).digest())
    return resumen.hexdigest()


def ficha_artefacto(paquete: Path, avisos: Sequence[str]) -> dict:
    """El contenido de `artefacto.json`."""
    return {
        "paquete": Path(paquete).name,
        "sha256": huella_paquete(paquete),
        "bytes": sum(p.stat().st_size for p in Path(paquete).rglob("*") if p.is_file()),
        "formato": "mlprogram",
        "precision": "float16",
        "objetivo": OBJETIVO,
        "especificacion": _leer_spec(paquete).specificationVersion,
        "metadatos": leer_metadatos(paquete),
        "bibliotecas": versiones_bibliotecas(),
        "avisos": list(avisos),
    }
