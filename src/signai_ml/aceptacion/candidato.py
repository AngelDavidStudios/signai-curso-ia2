"""El candidato de TT-14 en `modelos/reconocedor/` (data-model «Candidato»; research R-05, R-09).

`modelo.pt` sigue el contrato M-2 de TT-14 y `candidato.json` es su ficha. El candidato se identifica
por el SHA-256 de los bytes de `modelo.pt`, que nunca se modifica: es lo que se evalúa, se registra
y se exporta (FR-009).
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..reconocedor import LIMITE_PARAMETROS
from ..reconocedor.modelo import KPoseFormer, cargar, contar_parametros
from ..ventanas import VentanasNoValidas, vocabulario
from . import CANDIDATO
from .errores import CandidatoNoValido

HUELLAS = ("corpus.sha256_ventanas", "corpus.sha256_indice", "particion.huella")
DECISIONES = ("aceptado", "rechazado")
RESTAURAR = "uv run dvc checkout modelos/reconocedor.dvc"


@dataclass(frozen=True)
class Candidato:
    modelo: KPoseFormer
    metadatos: dict
    ficha: dict
    modelo_sha256: str
    clases: tuple[str, ...]
    glosas: tuple[str, ...]
    huellas: dict[str, str]
    parametros: int


def sha256_fichero(ruta: Path) -> str:
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def _no_valido(motivo: str) -> CandidatoNoValido:
    return CandidatoNoValido(f"{motivo} Restaura el candidato con `{RESTAURAR}`.")


def _leer_ficha(ruta: Path) -> dict:
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise _no_valido(f"{CANDIDATO / 'candidato.json'} no es JSON válido ({error}).") from error


def _cargar_modelo(ruta: Path) -> tuple[KPoseFormer, dict]:
    try:
        return cargar(ruta)
    except (ValueError, KeyError, RuntimeError) as error:
        raise _no_valido(f"{CANDIDATO / 'modelo.pt'} no cumple el contrato M-2: {error}") from error


def _comprobar_vocabulario(raiz: Path, clases: tuple, glosas: tuple, lector: Callable) -> None:
    try:
        filas = lector(raiz)
    except VentanasNoValidas as error:
        raise CandidatoNoValido(str(error)) from error
    if clases != tuple(f["clase"] for f in filas) or glosas != tuple(f["glosa"] for f in filas):
        raise _no_valido(
            "Las clases o las glosas de modelo.pt no son las de config/corpus/vocabulario.csv, en número u orden."
        )


def leer(raiz: Path, lector_vocabulario: Callable = vocabulario) -> Candidato:
    """Lee y valida el candidato sin modificar nada."""
    base = Path(raiz) / CANDIDATO
    for fichero in ("modelo.pt", "candidato.json"):
        if not (base / fichero).is_file():
            raise _no_valido(f"Falta {CANDIDATO / fichero}.")
    ficha = _leer_ficha(base / "candidato.json")
    modelo_sha256 = sha256_fichero(base / "modelo.pt")
    modelo, metadatos = _cargar_modelo(base / "modelo.pt")
    clases, glosas = tuple(metadatos["clases"]), tuple(metadatos["glosas"])
    _comprobar_vocabulario(Path(raiz), clases, glosas, lector_vocabulario)
    parametros = contar_parametros(modelo)
    if parametros > LIMITE_PARAMETROS:
        raise _no_valido(f"El modelo tiene {parametros:,} parámetros y el límite es {LIMITE_PARAMETROS:,} (ADR-0013).")
    huellas = ficha.get("huellas") or {}
    if any(clave not in huellas for clave in HUELLAS):
        raise _no_valido(f"La ficha no tiene las huellas {', '.join(HUELLAS)}.")
    if ficha.get("run_id") != metadatos.get("corrida", {}).get("run_id"):
        raise _no_valido("La ficha y modelo.pt no son de la misma corrida de TT-14.")
    return Candidato(
        modelo=modelo,
        metadatos=metadatos,
        ficha=ficha,
        modelo_sha256=modelo_sha256,
        clases=clases,
        glosas=glosas,
        huellas={clave: huellas[clave] for clave in HUELLAS},
        parametros=parametros,
    )


def anotar_evaluacion(
    raiz: Path, *, consulta: int, run_id: str, aciertos: int, total: int, top1: float, decision: str
) -> Path:
    """Escribe en `candidato.json` el resultado de su consulta; `modelo.pt` no se toca (FR-007, FR-010).

    Las transiciones son «pendiente de aceptación (TT-15)» → aceptado | rechazado. Rechazado es
    terminal: el siguiente candidato lo trae TT-14 con otro barrido. Volver a anotar la misma consulta
    conserva lo que añadió el registro (`modelo_registrado`).
    """
    if decision not in DECISIONES:
        raise ValueError(f"Decisión desconocida: {decision!r}")
    ruta = Path(raiz) / CANDIDATO / "candidato.json"
    ficha = _leer_ficha(ruta)
    anterior = ficha.get("aceptacion") or {}
    if ficha.get("estado") == "rechazado" and (anterior.get("consulta"), decision) != (consulta, "rechazado"):
        raise CandidatoNoValido(
            f"El candidato ya fue rechazado en la consulta {anterior.get('consulta')}: es terminal. "
            "Vuelve a TT-14 con un barrido nuevo."
        )
    resultado = {
        "consulta": consulta, "run_id": run_id, "aciertos": aciertos, "total": total, "top1": top1,
        "decision": decision,
    }
    if anterior.get("consulta") == consulta and "modelo_registrado" in anterior:
        resultado["modelo_registrado"] = anterior["modelo_registrado"]
    ficha["estado"] = decision
    ficha["aceptacion"] = resultado
    ruta.write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ruta


def anotar_registro(raiz: Path, *, nombre: str, version: int) -> Path:
    """Añade a `candidato.json` la versión registrada del candidato aceptado (FR-010)."""
    ruta = Path(raiz) / CANDIDATO / "candidato.json"
    ficha = _leer_ficha(ruta)
    if ficha.get("estado") != "aceptado" or "aceptacion" not in ficha:
        raise CandidatoNoValido("Solo se anota la versión registrada de un candidato aceptado.")
    ficha["aceptacion"]["modelo_registrado"] = {"nombre": nombre, "version": version}
    ruta.write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ruta
