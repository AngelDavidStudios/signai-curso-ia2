"""`signai-copia`: la copia de seguridad cifrada del equipo de entrenamiento (contracts/cli.md).

La copia se cifra en el equipo con restic y pasa a un iPad por cable con Finder (ADR-0027). La
clave solo existe en papel: **nunca** se acepta como argumento ni como variable de entorno; las
órdenes que la necesitan la piden una vez, sin eco.

Códigos de salida: `0` correcto · `1` rechazo de dominio · `2` error de uso · `3` falta una
herramienta.
"""

import argparse
import getpass
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from . import clave, seguimiento
from .dvc import EstadoDvc, EstadoDvcLocal
from .envio import Clonador, ClonadorApfs, Envios
from .errores import ErrorCopia, ErrorUso, RechazoDominio
from .registro import Registro
from .repositorio import Repositorio, Restic
from .servicio import ServicioCopia

RAIZ_REPO = Path(__file__).resolve().parents[3]

SUBCOMANDOS = (
    "clave",
    "iniciar",
    "actualizar",
    "confirmar",
    "estado",
    "instantaneas",
    "restaurar",
    "comprobar",
    "reubicar",
    "anotar-crudas",
    "registro",
)


@dataclass
class Contexto:
    raiz: Path
    repositorio_para: Callable[[Path, str], Repositorio]
    pedir_clave: Callable[[str], str]
    hay_terminal: Callable[[], bool]
    ahora: Callable[[], datetime]
    salida: TextIO
    error: TextIO
    estado_dvc: EstadoDvc = field(default_factory=lambda: EstadoDvcLocal(RAIZ_REPO))
    clonador: Clonador = field(default_factory=ClonadorApfs)

    @property
    def ruta_repositorio(self) -> Path:
        return self.raiz / "copia" / "repositorio"

    def escribir(self, texto: str = "") -> None:
        print(texto, file=self.salida)

    def servicio(self, repositorio: Repositorio | None = None) -> ServicioCopia:
        return ServicioCopia(
            self.raiz,
            registro=Registro(self.raiz),
            ahora=self.ahora,
            repositorio=repositorio,
            estado_dvc=self.estado_dvc,
            envios=Envios(self.raiz, self.clonador),
        )

    def clave_tecleada(self, mensaje: str = "Clave de la copia (desde el papel): ") -> str:
        if not self.hay_terminal():
            raise ErrorUso("hace falta una terminal para teclear la clave: no se acepta de otra forma")
        return clave.validar(self.pedir_clave(mensaje))


def crear_contexto(raiz: Path) -> Contexto:
    return Contexto(
        raiz=raiz,
        repositorio_para=lambda ruta, clave_usada: Restic(ruta, clave_usada),
        pedir_clave=getpass.getpass,
        hay_terminal=sys.stdin.isatty,
        ahora=lambda: datetime.now(UTC),
        salida=sys.stdout,
        error=sys.stderr,
    )


def _raiz_por_defecto() -> Path:
    return Path(os.environ.get("SIGNAI_DATA_ROOT", Path.home() / "signai-data"))


# ── Subcomandos ───────────────────────────────────────────────────────────────────────────────


def _clave(ctx: Contexto, _: argparse.Namespace) -> int:
    ctx.escribir("Clave nueva de la copia de seguridad:")
    ctx.escribir()
    ctx.escribir(f"    {clave.formatear(clave.generar())}")
    ctx.escribir()
    ctx.escribir("→ escríbela en papel y guárdala en un lugar seguro: no se guarda en ningún sitio.")
    ctx.escribir("  Sin ella, la copia no se puede leer. Después ejecuta `signai-copia iniciar`.")
    return 0


def _iniciar(ctx: Contexto, _: argparse.Namespace) -> int:
    primera = ctx.clave_tecleada()
    segunda = ctx.clave_tecleada("Repite la clave: ")
    if primera != segunda:
        raise RechazoDominio("las dos claves no coinciden: no se ha creado nada")
    ctx.repositorio_para(ctx.ruta_repositorio, primera).iniciar()
    ctx.escribir(f"✓ Repositorio cifrado creado en {ctx.ruta_repositorio}")
    ctx.escribir("→ siguiente paso: `signai-copia actualizar --ronda inicial`")
    return 0


def _registro(ctx: Contexto, opciones: argparse.Namespace) -> int:
    entradas = Registro(ctx.raiz).leer(ultimas=opciones.ultimas)
    if not entradas:
        ctx.escribir("El registro de copias está vacío: sin entradas.")
    for entrada in entradas:
        linea = f"{entrada.fecha}  {entrada.tipo:<13} {entrada.resultado:<8}"
        if entrada.instantanea:
            linea += f"  {entrada.instantanea[:8]}"
        if entrada.ronda:
            linea += f"  ronda={entrada.ronda}"
        if entrada.motivo:
            linea += f"  — {entrada.motivo}"
        ctx.escribir(linea)
    return 0


def _tamano(bytes_: int) -> str:
    for unidad in ("B", "KB", "MB", "GB"):
        if bytes_ < 1000 or unidad == "GB":
            return f"{bytes_:.0f} {unidad}" if unidad == "B" else f"{bytes_:.1f} {unidad}".replace(".", ",")
        bytes_ /= 1000  # type: ignore[assignment]
    return ""  # pragma: no cover


def _actualizar(ctx: Contexto, opciones: argparse.Namespace) -> int:
    repositorio = ctx.repositorio_para(ctx.ruta_repositorio, ctx.clave_tecleada())
    resultado = ctx.servicio(repositorio).actualizar(ronda=opciones.ronda)
    envio = resultado.envio
    cambiados = ", ".join(resultado.conjuntos) or "ninguno"
    ctx.escribir(
        f"✓ Instantánea {resultado.instantanea[:8]} · conjuntos cambiados: {cambiados} · "
        f"añadidos {_tamano(resultado.bytes_anadidos)} · {resultado.duracion_s} s"
    )
    ctx.escribir("✓ Verificación del 5 % sin errores.")
    ctx.escribir()
    ctx.escribir("Envío al iPad:")
    ctx.escribir(f"  carpeta    {envio.ruta}")
    ctx.escribir(f"  archivos   {envio.archivos}   (elementos, con subcarpetas: {envio.elementos})")
    ctx.escribir(f"  tamaño     {_tamano(envio.bytes)}")
    ctx.escribir(f"  espacio libre necesario en el iPad: {_tamano(envio.espacio_necesario)}")
    ctx.escribir()
    ctx.escribir("→ Finder → iPad → Archivos: arrastra la carpeta sobre «Documents».")
    ctx.escribir("→ En el iPad, «Obtener información» de la carpeta, y después:")
    ctx.escribir(f"   uv run signai-copia confirmar --carpeta {envio.carpeta} --archivos <número que muestra el iPad>")
    return 0


def _confirmar(ctx: Contexto, opciones: argparse.Namespace) -> int:
    anterior = ctx.servicio().confirmar(opciones.carpeta, opciones.archivos)
    ctx.escribir(f"✓ Envío {opciones.carpeta} confirmado en el iPad.")
    if anterior and anterior != opciones.carpeta:
        ctx.escribir(f"→ Borra del iPad la carpeta anterior: {anterior}")
    return 0


def _estado(ctx: Contexto, _: argparse.Namespace) -> int:
    estado = ctx.servicio().estado()
    if estado.ultima_actualizacion:
        ctx.escribir(f"Última actualización: {estado.ultima_actualizacion[:16]} (instantánea {(estado.instantanea or '')[:8]})")
    else:
        ctx.escribir("Última actualización: nunca")
    if estado.en_ipad:
        ctx.escribir(f"En el iPad: sí ({estado.carpeta})")
    elif estado.carpeta:
        ctx.escribir(f"En el iPad: no — copia sin enviar al iPad ({estado.carpeta})")
    for nombre, valor in estado.conjuntos.items():
        ctx.escribir(f"{nombre}: {valor}")
    ctx.escribir(f"Corridas en curso: {estado.corridas_en_curso}")
    ctx.escribir(f"Objetos de DVC sin subir: {'sí' if estado.dvc_sin_subir else 'no'}")
    if estado.al_dia:
        ctx.escribir("✓ Todo al día: la copia está en el iPad.")
        return 0
    ctx.escribir("→ Pendiente: dvc push, signai-copia actualizar y el envío al iPad, según lo de arriba.")
    return 1


def _instantaneas(ctx: Contexto, opciones: argparse.Namespace) -> int:
    ruta = opciones.repositorio or ctx.ruta_repositorio
    lista = ctx.repositorio_para(ruta, ctx.clave_tecleada()).instantaneas()
    if not lista:
        ctx.escribir("El repositorio no tiene instantáneas.")
    for instantanea in lista:
        ctx.escribir(
            f"{instantanea.fecha[:19]}  {instantanea.id[:8]}  {_tamano(instantanea.bytes):>9}  {', '.join(instantanea.etiquetas)}"
        )
    return 0


def _restaurar(ctx: Contexto, opciones: argparse.Namespace) -> int:
    repositorio = ctx.repositorio_para(opciones.repositorio, ctx.clave_tecleada())
    ctx.servicio().restaurar(repositorio, opciones.destino, instantanea=opciones.instantanea, ruta=opciones.ruta)
    ctx.escribir(f"✓ Copia leída entera y sin errores; restaurado en {opciones.destino}")
    ctx.escribir(f"→ siguiente paso: uv run signai-copia comprobar --restaurado {opciones.destino}")
    return 0


def _comprobar(ctx: Contexto, opciones: argparse.Namespace) -> int:
    ctx.escribir(ctx.servicio().comprobar(opciones.restaurado))
    ctx.escribir("✓ Restauración idéntica al origen. Copia este resumen en evidencias/prueba-restauracion.md")
    return 0


def _reubicar(ctx: Contexto, opciones: argparse.Namespace) -> int:
    cambiadas = seguimiento.reubicar(opciones.base, opciones.de, opciones.a)
    ctx.escribir(f"✓ {cambiadas} rutas de artefactos reubicadas en {opciones.base.name}")
    return 0


def _anotar_crudas(ctx: Contexto, opciones: argparse.Namespace) -> int:
    ctx.servicio().anotar_crudas(
        opciones.donde,
        fuera=opciones.fuera_del_equipo == "si",
        cifrada=opciones.cifrada == "si",
        completa=opciones.completa == "si",
    )
    ctx.escribir("✓ Copia de las grabaciones crudas comprobada: fuera del equipo, cifrada y completa.")
    return 0


def analizador() -> argparse.ArgumentParser:
    raiz = argparse.ArgumentParser(prog="signai-copia", description=__doc__.split("\n\n")[0])
    raiz.add_argument("--raiz", type=Path, help="raíz de datos (por defecto $SIGNAI_DATA_ROOT o ~/signai-data)")
    sub = raiz.add_subparsers(dest="subcomando", required=True, metavar="subcomando")

    def orden(nombre: str, ayuda: str, funcion: Callable[[Contexto, argparse.Namespace], int]) -> argparse.ArgumentParser:
        analizador_orden = sub.add_parser(nombre, help=ayuda)
        analizador_orden.set_defaults(funcion=funcion)
        return analizador_orden

    orden("clave", "genera una clave nueva y la muestra una sola vez", _clave)
    orden("iniciar", "crea el repositorio cifrado (pide la clave dos veces)", _iniciar)
    orden("actualizar", "copia, verifica y prepara el envío al iPad", _actualizar).add_argument("--ronda")
    confirmar = orden("confirmar", "registra que la carpeta llegó entera al iPad", _confirmar)
    confirmar.add_argument("--carpeta", required=True)
    confirmar.add_argument("--archivos", type=int, required=True)
    orden("estado", "muestra el estado de la copia (sin clave)", _estado)
    orden("instantaneas", "lista las instantáneas", _instantaneas).add_argument("--repositorio", type=Path)
    restaurar = orden("restaurar", "verifica y restaura una copia devuelta desde el iPad", _restaurar)
    restaurar.add_argument("--repositorio", type=Path, required=True)
    restaurar.add_argument("--destino", type=Path, required=True)
    restaurar.add_argument("--instantanea", default="latest")
    restaurar.add_argument("--ruta")
    orden("comprobar", "compara lo restaurado con el origen (sin clave)", _comprobar).add_argument(
        "--restaurado", type=Path, required=True
    )
    reubicar = orden("reubicar", "reescribe las rutas de MLflow restaurado en otra raíz", _reubicar)
    reubicar.add_argument("--base", type=Path, required=True)
    reubicar.add_argument("--de", required=True)
    reubicar.add_argument("--a", required=True)
    crudas = orden("anotar-crudas", "registra la comprobación de la copia de las grabaciones crudas", _anotar_crudas)
    crudas.add_argument("--donde", required=True)
    for opcion in ("--fuera-del-equipo", "--cifrada", "--completa"):
        crudas.add_argument(opcion, choices=("si", "no"), required=True)
    orden("registro", "muestra las últimas entradas del registro de copias", _registro).add_argument(
        "--ultimas", type=int, default=20
    )
    return raiz


def main(argv: Sequence[str] | None = None, contexto: Contexto | None = None) -> int:
    error = contexto.error if contexto else sys.stderr
    try:
        opciones = analizador().parse_args(argv)
    except SystemExit as salida:
        return int(salida.code or 0)
    try:
        if contexto is None:
            contexto = crear_contexto((opciones.raiz or _raiz_por_defecto()).expanduser())
        elif opciones.raiz is not None:
            contexto.raiz = opciones.raiz
        return opciones.funcion(contexto, opciones)
    except ErrorCopia as fallo:
        print(fallo, file=error)
        return fallo.codigo_salida
