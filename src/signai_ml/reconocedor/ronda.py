"""Declaración de rondas en TOML versionado (contracts/ronda.md; research R-05).

    config/reconocedor/<barrido>/ronda-NN.toml

Cada `[[configuracion]]` hereda `base` y solo cambia lo que declara (fusión profunda). La huella del
fichero se registra en cada corrida: una ronda ya lanzada no se edita.
"""

import copy
import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .configuracion import Configuracion, configuracion_desde_dict
from .errores import RondaNoValida

CLAVES = {"barrido", "ronda", "base", "configuracion"}


@dataclass(frozen=True)
class Ronda:
    barrido: str
    numero: int
    configuraciones: tuple[Configuracion, ...]
    huella: str
    ruta: Path


def ruta_barrido(raiz: Path, barrido: str) -> Path:
    return Path(raiz) / "config" / "reconocedor" / barrido


def _fusionar(base: dict, cambios: dict) -> dict:
    resultado = copy.deepcopy(base)
    for clave, valor in cambios.items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = _fusionar(resultado[clave], valor)
        else:
            resultado[clave] = copy.deepcopy(valor)
    return resultado


def _sin_repetidos(configuraciones: list[Configuracion], origen: str) -> None:
    nombres: set[str] = set()
    valores: dict[tuple, str] = {}
    for cfg in configuraciones:
        if cfg.nombre in nombres:
            raise RondaNoValida(f"{origen}: el nombre «{cfg.nombre}» está repetido; cada configuración tiene el suyo.")
        nombres.add(cfg.nombre)
        if cfg.valores() in valores:
            raise RondaNoValida(
                f"{origen}: «{cfg.nombre}» tiene los mismos valores que «{valores[cfg.valores()]}»."
            )
        valores[cfg.valores()] = cfg.nombre


def leer_ronda(ruta: Path) -> Ronda:
    contenido = Path(ruta).read_bytes()
    try:
        datos = tomllib.loads(contenido.decode("utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise RondaNoValida(f"{ruta}: no es TOML válido ({error}).") from error
    sobrantes = sorted(set(datos) - CLAVES)
    if sobrantes:
        raise RondaNoValida(f"{ruta}: clave desconocida «{sobrantes[0]}».")
    barrido, numero = datos.get("barrido"), datos.get("ronda")
    if not isinstance(barrido, str) or not barrido:
        raise RondaNoValida(f"{ruta}: falta `barrido`.")
    if isinstance(numero, bool) or not isinstance(numero, int) or numero < 1:
        raise RondaNoValida(f"{ruta}: `ronda` debe ser un entero ≥ 1.")
    declaradas = datos.get("configuracion", [])
    if not declaradas:
        raise RondaNoValida(f"{ruta}: la ronda no declara ninguna configuración.")
    base = datos.get("base", {})
    configuraciones = []
    for declarada in declaradas:
        declarada = dict(declarada)
        nombre = declarada.pop("nombre", "")
        configuraciones.append(configuracion_desde_dict(nombre, _fusionar(base, declarada)))
    _sin_repetidos(configuraciones, str(ruta))
    return Ronda(barrido, numero, tuple(configuraciones), hashlib.sha256(contenido).hexdigest(), Path(ruta))


def leer_barrido(directorio: Path) -> tuple[Ronda, ...]:
    rondas = tuple(sorted((leer_ronda(r) for r in sorted(Path(directorio).glob("ronda-*.toml"))), key=lambda r: r.numero))
    if not rondas:
        raise RondaNoValida(f"El barrido {Path(directorio).name} no tiene rondas en {directorio}.")
    for esperado, ronda in enumerate(rondas, start=1):
        if ronda.barrido != Path(directorio).name:
            raise RondaNoValida(f"{ronda.ruta}: declara el barrido «{ronda.barrido}» pero está en {Path(directorio).name}.")
        if ronda.numero != esperado:
            raise RondaNoValida(f"Las rondas de {Path(directorio).name} deben ser consecutivas desde 1; falta la {esperado}.")
    _sin_repetidos([c for r in rondas for c in r.configuraciones], f"barrido {Path(directorio).name}")
    return rondas


def buscar_configuracion(rondas: tuple[Ronda, ...], nombre: str) -> tuple[Ronda, Configuracion]:
    for ronda in rondas:
        for cfg in ronda.configuraciones:
            if cfg.nombre == nombre:
                return ronda, cfg
    raise RondaNoValida(f"Ninguna ronda declara la configuración «{nombre}»; declárala y confírmala antes.")
