"""CLI `signai-aceptacion` (contracts/cli.md).

    signai-aceptacion evaluar     mide el candidato sobre la prueba y decide (una consulta)
    signai-aceptacion registrar   registra el candidato aceptado en el registro de modelos
    signai-aceptacion exportar    lo exporta a Core ML y demuestra la paridad
    signai-aceptacion estado      muestra candidato, consultas, versión y exportación

Código de salida: 0 bien (también «rechazado»), 1 error de los datos o del estado, 2 uso
incorrecto, 3 árbol sucio. No hay ninguna opción para saltarse la guarda del árbol: la consulta a la
prueba es única y su commit tiene que describir lo que se ejecutó.
"""

import argparse
import csv
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..codigo import codigo_permitido
from ..particion.errores import ErrorParticion
from ..particion.lectura import para_evaluacion
from . import ALIAS, CANDIDATO, CLIPS_PRUEBA, MODELO_REGISTRADO, POR_CLASE
from .candidato import Candidato, anotar_evaluacion, anotar_registro, leer, sha256_fichero
from .consultas import EvaluacionRegistrada, siguiente, total
from .datos import cargar_prueba
from .decision import Decision, decidir, umbral_aciertos
from .errores import ArbolSucio, ErrorAceptacion, NoAceptado
from .evaluacion import ResultadoEvaluacion, evaluar, probabilidades
from .estado import cmd_estado, estado_dvc as _estado_dvc
from .exportacion import cmd_exportar
from .formato import decimal, miles, porcentaje
from .conversion import Convertidor, ConvertidorCoreml
from .ejecucion import Ejecutor, EjecutorCoreml
from .registro import RegistroAceptacion, RegistroAceptacionMlflow

ORDENES = {
    "evaluar": "Mide el candidato sobre los 1.800 clips de prueba y decide su aceptación.",
    "registrar": "Registra el candidato aceptado como versión de reconocedor-kposeformer.",
    "exportar": "Exporta la versión aceptada a Core ML y demuestra la paridad.",
    "estado": "Muestra el candidato, las consultas a la prueba, la versión y la exportación.",
}
VERSIONAR_CANDIDATO = "Versiona el candidato:  uv run dvc add modelos/reconocedor && uv run dvc push"


@dataclass
class Contexto:
    raiz: Path
    registro: RegistroAceptacion
    lector: Callable
    comprobar_codigo: Callable[[Path], tuple[str, bool]]
    total: int = CLIPS_PRUEBA
    por_clase: int = POR_CLASE
    convertidor: Convertidor = field(default_factory=ConvertidorCoreml)
    ejecutor: Callable[[str], Ejecutor] = EjecutorCoreml
    estado_dvc: Callable[[Path, Sequence[str]], dict[str, str]] = _estado_dvc


def _raiz_repo() -> Path:
    return Path(__file__).resolve().parents[3]


def _codigo_limpio(raiz: Path) -> tuple[str, bool]:
    return codigo_permitido(raiz, False, ArbolSucio, "Confirma los cambios y vuelve a lanzar.")


# ── Formato ─────────────────────────────────────────────────────────────────────────────────────


def lineas_resultado(
    numero: int, total_consultas: int, candidato: Candidato, aciertos: int, total_clips: int,
    metricas: dict[str, float], decision: str,
) -> list[str]:
    ficha = candidato.ficha
    return [
        f"Consulta {numero} a la prueba ({total_consultas} en total) · candidato {ficha.get('barrido')} / "
        f"{ficha.get('configuracion')} / semilla {ficha.get('semilla')} · modelo {candidato.modelo_sha256[:8]}…",
        f"Top-1 de prueba: {miles(aciertos)} / {miles(total_clips)} = {porcentaje(aciertos / total_clips)}  "
        f"(umbral ≥85 %: {miles(umbral_aciertos(total_clips))})",
        f"F1 macro {decimal(metricas['f1_macro'])} · acierto al 95 % {porcentaje(metricas['acierto_95'])} · "
        f"glosas falsas al 95 % {porcentaje(metricas['falsas_95'])}",
        f"Decisión: {decision.upper()}",
    ]


# ── Artefactos de la consulta (contracts/registro.md, G-1) ──────────────────────────────────────


def escribir_artefactos(
    destino: Path, resultado: ResultadoEvaluacion, candidato: Candidato, lineas: list[str]
) -> list[Path]:
    por_clase = destino / "metricas_por_clase.json"
    por_clase.write_text(
        json.dumps(
            [
                {"clase": c, "glosa": g, "top1": t, "f1": f}
                for c, g, t, f in zip(candidato.clases, candidato.glosas, resultado.top1_por_clase,
                                      resultado.f1_por_clase, strict=True)
            ],
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    confusion = destino / "confusion.csv"
    with open(confusion, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo, lineterminator="\n")
        escritor.writerow(["real\\predicha", *candidato.clases])
        for clase, fila in zip(candidato.clases, resultado.confusion.tolist(), strict=True):
            escritor.writerow([clase, *fila])
    informe = destino / "informe.md"
    informe.write_text(
        "# Evaluación de aceptación (TT-15)\n\n"
        + "\n".join(f"- {linea}" for linea in lineas)
        + "\n\nLa cifra se mide con partición aleatoria estratificada (ADR-0011): describe el reconocimiento "
        "sobre el dominio del corpus, no frente a una persona que el modelo no ha visto. Al reportarla se "
        "declara el número de consultas a la prueba.\n",
        encoding="utf-8",
    )
    return [por_clase, confusion, informe]


# ── evaluar ─────────────────────────────────────────────────────────────────────────────────────


def _etiquetas(candidato: Candidato, commit: str, sucio: bool) -> dict[str, str]:
    ficha = candidato.ficha
    return {
        "modelo.sha256": candidato.modelo_sha256,
        "corrida.tt14": str(ficha.get("run_id")),
        "barrido": str(ficha.get("barrido")),
        "configuracion": str(ficha.get("configuracion")),
        "semilla": str(ficha.get("semilla")),
        **candidato.huellas,
        "codigo.commit": commit,
        "codigo.sucio": str(sucio).lower(),
    }


def _ya_evaluado(ctx: Contexto, candidato: Candidato, terminada: EvaluacionRegistrada, total_consultas: int) -> int:
    aciertos, total_clips = int(terminada.metricas["aciertos"]), int(terminada.metricas["total"])
    anotar_evaluacion(
        ctx.raiz, consulta=terminada.consulta, run_id=terminada.run_id, aciertos=aciertos, total=total_clips,
        top1=terminada.metricas["top1"], decision=str(terminada.decision),
    )
    print(f"El candidato ya se evaluó en la consulta {terminada.consulta} (ya evaluado): no se vuelve a consultar la prueba.")
    for linea in lineas_resultado(
        terminada.consulta, total_consultas, candidato, aciertos, total_clips, terminada.metricas, str(terminada.decision)
    ):
        print(linea)
    return 0


def cmd_evaluar(ctx: Contexto) -> int:
    commit, sucio = ctx.comprobar_codigo(ctx.raiz)
    candidato = leer(ctx.raiz)
    evaluaciones = ctx.registro.evaluaciones()
    proxima = siguiente(evaluaciones, candidato.modelo_sha256)
    total_consultas = max(total(evaluaciones), proxima.numero)
    if proxima.terminada is not None:
        return _ya_evaluado(ctx, candidato, proxima.terminada, total_consultas)

    datos = cargar_prueba(ctx.raiz, candidato, lector=ctx.lector, total=ctx.total, por_clase=ctx.por_clase)
    id_consulta = ctx.registro.abrir_consulta(proxima.numero, _etiquetas(candidato, commit, sucio))
    try:
        resultado = evaluar(probabilidades(candidato.modelo, datos.x), datos.y, len(candidato.clases))
        decision = decidir(resultado.aciertos, resultado.total)
        lineas = lineas_resultado(
            proxima.numero, total_consultas, candidato, resultado.aciertos, resultado.total, resultado.metricas(),
            decision,
        )
        with tempfile.TemporaryDirectory(prefix="signai-aceptacion-") as temporal:
            artefactos = escribir_artefactos(Path(temporal), resultado, candidato, lineas)
            ctx.registro.terminar_consulta(
                id_consulta, resultado.metricas(), decision.value, artefactos, ctx.raiz / CANDIDATO / "modelo.pt"
            )
    except KeyboardInterrupt:
        ctx.registro.interrumpir_consulta(id_consulta, "interrumpida por la persona (Ctrl-C)")
        raise ErrorAceptacion(
            f"Consulta {proxima.numero} interrumpida antes de dar su cifra. Al relanzar se reutiliza su número."
        ) from None
    except BaseException as error:
        ctx.registro.fallar_consulta(id_consulta, str(error) or type(error).__name__)
        raise
    anotar_evaluacion(
        ctx.raiz, consulta=proxima.numero, run_id=id_consulta, aciertos=resultado.aciertos, total=resultado.total,
        top1=resultado.top1, decision=decision.value,
    )
    for linea in lineas:
        print(linea)
    print(VERSIONAR_CANDIDATO)
    if decision is Decision.RECHAZADO:
        print("El candidato no alcanza el umbral: no se registra ni se exporta. Vuelve a TT-14 con un barrido nuevo.")
    return 0


# ── registrar ───────────────────────────────────────────────────────────────────────────────────


def _evaluacion_aceptada(ctx: Contexto, candidato: Candidato) -> EvaluacionRegistrada:
    terminada = siguiente(ctx.registro.evaluaciones(), candidato.modelo_sha256).terminada
    if terminada is None:
        raise NoAceptado(
            "El candidato no tiene una evaluación terminada: lanza antes `uv run signai-aceptacion evaluar`."
        )
    if terminada.decision != Decision.ACEPTADO:
        raise NoAceptado(
            f"El candidato no alcanzó el umbral en la consulta {terminada.consulta} ({terminada.decision}): "
            "no se registra ni se exporta (constitución IV). Vuelve a TT-14 con un barrido nuevo."
        )
    return terminada


def _etiquetas_version(candidato: Candidato, terminada: EvaluacionRegistrada) -> dict[str, str]:
    return {
        "estado": "aceptado",
        "consulta": str(terminada.consulta),
        "aciertos": str(int(terminada.metricas["aciertos"])),
        "top1_prueba": f"{terminada.metricas['top1']:.4f}",
        **{clave: terminada.etiquetas[clave] for clave in (
            "modelo.sha256", "corrida.tt14", "barrido", "configuracion", "semilla",
            "corpus.sha256_ventanas", "corpus.sha256_indice", "particion.huella",
        )},
    }


def cmd_registrar(ctx: Contexto) -> int:
    candidato = leer(ctx.raiz)
    terminada = _evaluacion_aceptada(ctx, candidato)
    anotar_evaluacion(
        ctx.raiz, consulta=terminada.consulta, run_id=terminada.run_id, aciertos=int(terminada.metricas["aciertos"]),
        total=int(terminada.metricas["total"]), top1=terminada.metricas["top1"], decision=str(terminada.decision),
    )
    existente = ctx.registro.version_de(candidato.modelo_sha256)
    version = existente or ctx.registro.registrar_version(terminada.run_id, _etiquetas_version(candidato, terminada))
    with tempfile.TemporaryDirectory(prefix="signai-aceptacion-") as temporal:
        registrado = sha256_fichero(ctx.registro.descargar_modelo(version.version, Path(temporal)))
    if registrado != candidato.modelo_sha256:
        raise ErrorAceptacion(
            f"Los pesos de la versión {version.version} ({registrado[:12]}…) no son los evaluados "
            f"({candidato.modelo_sha256[:12]}…): FR-009. No se anota el registro ni se exporta."
        )
    anotar_registro(ctx.raiz, nombre=MODELO_REGISTRADO, version=version.version)
    if existente:
        print(f"El candidato ya está registrado como versión {version.version} de {MODELO_REGISTRADO}: no se crea otra.")
    print(
        f"Versión {version.version} de {MODELO_REGISTRADO} · alias {ALIAS} · consulta {terminada.consulta} · "
        f"modelo {candidato.modelo_sha256[:8]}… (pesos comprobados)"
    )
    print(VERSIONAR_CANDIDATO)
    return 0


# ── main ────────────────────────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signai-aceptacion",
        description="Evaluación, aceptación y exportación del reconocedor (TT-15).",
    )
    ordenes = parser.add_subparsers(dest="orden", metavar="orden")
    subordenes = {nombre: ordenes.add_parser(nombre, help=ayuda, description=ayuda) for nombre, ayuda in ORDENES.items()}
    exportar = subordenes["exportar"]
    exportar.add_argument("--casos", type=Path, help="Casos de referencia (por defecto, los del contrato de normalización).")
    exportar.add_argument("--version", type=int, help="Versión a exportar (por defecto, la del alias aceptado).")
    return parser


def main(
    argv: Sequence[str] | None = None,
    raiz: Path | None = None,
    registro: RegistroAceptacion | None = None,
    lector: Callable | None = None,
    comprobar_codigo: Callable[[Path], tuple[str, bool]] | None = None,
    total: int | None = None,
    por_clase: int | None = None,
    convertidor: Convertidor | None = None,
    ejecutor: Callable[[str], Ejecutor] | None = None,
    estado_dvc: Callable[[Path, Sequence[str]], dict[str, str]] | None = None,
) -> int:
    parser = _parser()
    try:
        argumentos = parser.parse_args(argv)
    except SystemExit as salida:
        return int(salida.code or 0)
    if argumentos.orden is None:
        parser.print_help()
        return 2
    ctx = Contexto(
        raiz=Path(raiz) if raiz else _raiz_repo(),
        registro=registro or RegistroAceptacionMlflow(),
        lector=lector or para_evaluacion,
        comprobar_codigo=comprobar_codigo or _codigo_limpio,
        total=total or CLIPS_PRUEBA,
        por_clase=por_clase or POR_CLASE,
        convertidor=convertidor or ConvertidorCoreml(),
        ejecutor=ejecutor or EjecutorCoreml,
        estado_dvc=estado_dvc or _estado_dvc,
    )
    try:
        if argumentos.orden == "evaluar":
            return cmd_evaluar(ctx)
        if argumentos.orden == "registrar":
            return cmd_registrar(ctx)
        if argumentos.orden == "exportar":
            return cmd_exportar(ctx, argumentos.casos, argumentos.version)
        return cmd_estado(ctx)
    except (ErrorAceptacion, ErrorParticion) as error:
        print(f"Error: {error}", file=sys.stderr)
        return error.codigo_salida


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
