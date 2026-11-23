"""El registro de copias: `$SIGNAI_DATA_ROOT/copia/registro.jsonl` (data-model; FR-008, FR-011).

Una línea JSON por operación, que solo crece. Un fallo también se anota. Ningún campo admite
rutas de las grabaciones crudas, nombres de archivo de vídeo ni glosas (FR-016): se rechaza antes
de escribir.
"""

import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errores import ErrorUso, RechazoDominio

TIPOS = ("actualizacion", "envio", "verificacion", "restauracion", "crudas")
RESULTADOS = ("correcto", "fallo")

_DATOS_DE_SENANTES = re.compile(r"Raw-dataset|Lista-Prioritaria|\.(mov|mp4)\b", re.IGNORECASE)


def _textos(valor: Any) -> list[str]:
    if isinstance(valor, str):
        return [valor]
    if isinstance(valor, Mapping):
        return [texto for clave, dato in valor.items() for texto in (*_textos(clave), *_textos(dato))]
    if isinstance(valor, (list, tuple)):
        return [texto for dato in valor for texto in _textos(dato)]
    return []


def sanear(texto: str) -> str:
    """Sustituye por «[ruta omitida]» cada fragmento con rutas de grabaciones o nombres de vídeo.

    Un fallo de DVC o de restic puede nombrar clips; sin esto, el registro rechazaría la entrada y
    el fallo quedaría sin anotar (FR-011, FR-016).
    """
    crudas = os.environ.get("SIGNAI_CRUDAS")
    if crudas:
        texto = re.sub(rf"[^\s,;]*{re.escape(crudas)}[^\s,;]*", "[ruta omitida]", texto)
    return re.sub(rf"[^\s,;]*(?:{_DATOS_DE_SENANTES.pattern})[^\s,;]*", "[ruta omitida]", texto, flags=re.IGNORECASE)


def _sin_datos_de_senantes(valores: list[str]) -> None:
    crudas = os.environ.get("SIGNAI_CRUDAS")
    for valor in valores:
        if _DATOS_DE_SENANTES.search(valor) or (crudas and crudas in valor):
            raise RechazoDominio("el registro no admite rutas de grabaciones, vídeos ni glosas (FR-016)")


@dataclass(frozen=True)
class Entrada:
    fecha: str
    tipo: str
    resultado: str
    instantanea: str | None = None
    ronda: str | None = None
    conjuntos: tuple[str, ...] = ()
    bytes_anadidos: int = 0
    duracion_s: int = 0
    detalle: Mapping[str, Any] = field(default_factory=dict)
    motivo: str | None = None

    def __post_init__(self) -> None:
        if self.tipo not in TIPOS:
            raise ErrorUso(f"tipo de entrada desconocido: {self.tipo!r}")
        if self.resultado not in RESULTADOS:
            raise ErrorUso(f"resultado desconocido: {self.resultado!r}")
        if (self.resultado == "fallo") != (self.motivo is not None):
            raise ErrorUso("el motivo es obligatorio en un fallo y no se anota en un acierto")
        for nombre in ("bytes_anadidos", "duracion_s"):
            valor = getattr(self, nombre)
            if isinstance(valor, bool) or not isinstance(valor, int) or valor < 0:
                raise ErrorUso(f"{nombre} debe ser un entero ≥ 0")
        _sin_datos_de_senantes(_textos(asdict(self)))


class Registro:
    def __init__(self, raiz: Path) -> None:
        self._ruta = Path(raiz) / "copia" / "registro.jsonl"

    def anadir(self, entrada: Entrada) -> None:
        self._ruta.parent.mkdir(parents=True, exist_ok=True)
        datos = asdict(entrada) | {"conjuntos": list(entrada.conjuntos), "detalle": dict(entrada.detalle)}
        with self._ruta.open("a", encoding="utf-8") as archivo:
            archivo.write(json.dumps(datos, ensure_ascii=False) + "\n")

    def leer(self, ultimas: int | None = None) -> list[Entrada]:
        if not self._ruta.exists():
            return []
        lineas = self._ruta.read_text(encoding="utf-8").splitlines()
        entradas = [self._entrada(json.loads(linea)) for linea in lineas if linea.strip()]
        return entradas if ultimas is None else entradas[-ultimas:]

    @staticmethod
    def _entrada(datos: dict[str, Any]) -> Entrada:
        return Entrada(**(datos | {"conjuntos": tuple(datos.get("conjuntos", ()))}))
