"""CLI `signai-preproceso` (contracts/cli.md): lote → aprobar-lote → procesar → verificar.

    signai-preproceso lote [--procesos N]
    signai-preproceso aprobar-lote [--rechazar --nota "…"]
    signai-preproceso procesar [--procesos N]
    signai-preproceso verificar

Los datos se resuelven desde la raíz del repositorio (`datos/`); el material desechable (caché de
detecciones y vídeos de revisión), desde `$SIGNAI_DATA_ROOT/preproceso/`. Código de salida: 0 bien,
1 error de los datos o del proceso, 2 uso incorrecto. No hay orden para saltarse la aprobación.
"""

import argparse
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .almacen import Almacen, Versiones, sha256_fichero
from .aprobacion import exigir, invalidar, leer, registrar, vigente
from .cadena import Cadena, Extractor
from .clips import ClipEtiquetado, leer_etiquetas
from .deteccion import ExtractorKeypoints
from .errores import ErrorPreproceso, ErrorUso
from .informe import escribir_informe, informe_clip, informe_lote
from .lote import SEMILLA_LOTE, elegir_lote, elegir_revision, guardar_lote, leer_lote
from .revision import generar_video
from .verificacion import verificar

ORDEN_DVC = "uv run dvc add datos/keypoints datos/ventanas datos/validacion && uv run dvc push"


def _procesos_por_defecto() -> int:
    try:
        rendimiento = int(subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, check=True
        ).stdout)
    except (OSError, ValueError, subprocess.CalledProcessError):
        rendimiento = os.cpu_count() or 2
    return max(1, rendimiento // 2)


def _raiz_repo() -> Path:
    return Path(__file__).resolve().parents[3]


def _desechable() -> Path:
    return Path(os.environ.get("SIGNAI_DATA_ROOT", Path.home() / "signai-data"))


@dataclass
class Contexto:
    raiz: Path = field(default_factory=_raiz_repo)
    desechable: Path = field(default_factory=_desechable)
    extractor: Extractor = field(default_factory=ExtractorKeypoints)
    versiones: Versiones = field(default_factory=Versiones.actuales)
    generar_video: Callable[[Path, np.ndarray, Path], None] = generar_video

    @property
    def datos(self) -> Path:
        return self.raiz / "datos"

    @property
    def validacion(self) -> Path:
        return self.datos / "validacion"

    @property
    def cache(self) -> Path:
        return self.desechable / "preproceso" / "detecciones"

    @property
    def dir_revision(self) -> Path:
        return self.desechable / "preproceso" / "revision"

    def clips(self) -> tuple[ClipEtiquetado, ...]:
        return leer_etiquetas(self.datos / "clips")

    def cadena(self, base: Path) -> Cadena:
        return Cadena(self.extractor, Almacen(base), self.cache, self.versiones)


def _informe_vigente(ctx: Contexto) -> str:
    ruta = ctx.validacion / "informe.json"
    if not ruta.is_file():
        raise ErrorPreproceso("El lote de validación no está procesado: lanza `signai-preproceso lote`.")
    return sha256_fichero(ruta)


def _errores(errores: dict[str, str]) -> None:
    print(f"\n{len(errores)} clip(s) con error (no producen resultado):")
    for clip_id, mensaje in sorted(errores.items()):
        print(f"  - {mensaje}" if mensaje.startswith(clip_id) else f"  - {clip_id}: {mensaje}")


def cmd_lote(ctx: Contexto, procesos: int) -> int:
    clips = ctx.clips()
    ruta_lote = ctx.validacion / "lote.json"
    if ruta_lote.is_file():
        elegidos = set(leer_lote(ruta_lote)["clips"])
        lote = tuple(c for c in clips if c.clip_id in elegidos)
    else:
        lote = elegir_lote(clips)
    print(f"Lote de validación: {len(lote)} clips (uno por clase y grabación, semilla {SEMILLA_LOTE}).")

    cadena = ctx.cadena(ctx.validacion)
    inicio = time.monotonic()
    resumen = cadena.procesar(lote, procesos)
    segundos = time.monotonic() - inicio
    if resumen.errores:
        _errores(resumen.errores)
        return 1
    cadena.ensamblar(lote, {"lote_validacion": True, "semilla_lote": SEMILLA_LOTE})

    informes = [
        informe_clip(r.clip, r.keypoints, r.fallidos, r.ventana.sin_normalizar)
        for r in (resumen.resultados[c.clip_id] for c in lote)
    ]
    corpus = sum(-(-c.n_fotogramas // 2) for c in clips)
    datos_informe = informe_lote(informes, segundos, resumen.fotogramas_extraidos, corpus)
    sha = escribir_informe(ctx.validacion, datos_informe)
    if leer(ctx.validacion) is not None and not vigente(ctx.validacion, ctx.versiones, sha):
        invalidar(ctx.validacion)
        print("Aviso: la aprobación anterior queda invalidada (el lote se ha vuelto a procesar).")

    revision = elegir_revision(informes)
    for r in revision:
        clip = next(c for c in lote if c.clip_id == r["clip_id"])
        destino = ctx.dir_revision / f"{clip.clip_id}.mp4"
        ctx.generar_video(clip.ruta, resumen.resultados[clip.clip_id].keypoints, destino)
        r["video"] = str(destino)
    guardar_lote(ruta_lote, lote, revision)

    print(f"Procesados {len(resumen.procesados)}, ya vigentes {len(resumen.saltados)}.")
    print(f"Anomalías: {datos_informe['anomalias'] or 'ninguna'}")
    if datos_informe["proyeccion_horas"] is not None:
        print(
            f"Velocidad: {datos_informe['fotogramas_por_segundo']} fotogramas/s. "
            f"Proyección del corpus completo ({corpus} fotogramas): {datos_informe['proyeccion_horas']} h."
        )
    else:
        print("Proyección: sin medida (las detecciones salieron de la caché, sin pasar por Vision).")
    print(f"Informe: {ctx.validacion / 'informe.md'}")
    print(f"Vídeos de revisión ({len(revision)}) en {ctx.dir_revision}:")
    for r in revision:
        print(f"  - {r['clip_id']}  ({r['motivo']})")
    print("Mira los vídeos y registra tu decisión con `signai-preproceso aprobar-lote` "
          "(o `--rechazar --nota \"…\"`).")
    return 0


def cmd_aprobar(ctx: Contexto, rechazar: bool, nota: str) -> int:
    sha = _informe_vigente(ctx)
    lote_ids = set(leer_lote(ctx.validacion / "lote.json")["clips"])
    almacen = Almacen(ctx.validacion)
    pendientes = [c.clip_id for c in ctx.clips() if c.clip_id in lote_ids and not almacen.vigente(c, ctx.versiones)]
    if pendientes:
        raise ErrorPreproceso(
            f"El lote no está procesado con la versión vigente de la cadena ({len(pendientes)} clips): "
            "vuelve a lanzar `signai-preproceso lote`."
        )
    datos = registrar(
        ctx.validacion, "rechazado" if rechazar else "aprobado", nota, ctx.versiones, SEMILLA_LOTE, sha
    )
    print(f"Lote {datos['decision']} el {datos['fecha']} ({ctx.validacion / 'aprobacion.json'}).")
    return 0


def cmd_procesar(ctx: Contexto, procesos: int) -> int:
    aprobacion = exigir(ctx.validacion, ctx.versiones, _informe_vigente(ctx))
    clips = ctx.clips()
    print(f"Corpus completo: {len(clips)} clips con {procesos} proceso(s).")
    cadena = ctx.cadena(ctx.datos)
    inicio = time.monotonic()
    resumen = cadena.procesar(clips, procesos)
    horas = (time.monotonic() - inicio) / 3600
    print(f"Procesados {len(resumen.procesados)}, ya vigentes {len(resumen.saltados)} en {horas:.2f} h.")
    if resumen.errores:
        _errores(resumen.errores)
        print("Corrige los errores y vuelve a lanzar `procesar`: continuará por los pendientes.")
        return 1
    ruta = cadena.ensamblar(clips, {
        "semilla_lote": SEMILLA_LOTE,
        "aprobacion_lote": {"fecha": aprobacion["fecha"], "sha256_informe": aprobacion["sha256_informe"]},
    })
    print(f"Ventanas: {len(clips)} en {ruta}. Diferencia con etiquetas.csv: 0.")
    print(f"Comprueba con `signai-preproceso verificar` y versiona con:\n  {ORDEN_DVC}")
    return 0


def cmd_verificar(ctx: Contexto) -> int:
    clips = ctx.clips()
    problemas = verificar(clips, Almacen(ctx.datos))
    if problemas:
        print(f"{len(problemas)} problema(s):")
        for p in problemas:
            print(f"  - {p}")
        return 1
    print(f"Resultado correcto: {len(clips)} ventanas (75, 96), una por clip de etiquetas.csv.")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signai-preproceso", description="Cadena de preprocesamiento del corpus (TT-12)")
    sub = p.add_subparsers(dest="orden", required=True)
    s = sub.add_parser("lote", help="procesa el lote de validación y genera su informe y sus vídeos")
    s.add_argument("--procesos", type=int, default=None)
    s = sub.add_parser("aprobar-lote", help="registra la decisión del responsable sobre el lote")
    s.add_argument("--rechazar", action="store_true")
    s.add_argument("--nota", default="")
    s = sub.add_parser("procesar", help="procesa el corpus completo (exige el lote aprobado)")
    s.add_argument("--procesos", type=int, default=None)
    sub.add_parser("verificar", help="comprueba el resultado sin escribir nada")
    return p


def main(argv: Sequence[str] | None = None, contexto: Contexto | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as salida:
        return int(salida.code or 0)
    ctx = contexto or Contexto()
    procesos = getattr(args, "procesos", None) or _procesos_por_defecto()
    try:
        if args.orden == "lote":
            return cmd_lote(ctx, procesos)
        if args.orden == "aprobar-lote":
            if args.rechazar and not args.nota.strip():
                raise ErrorUso("Para rechazar el lote hace falta una nota: `aprobar-lote --rechazar --nota \"…\"`.")
            return cmd_aprobar(ctx, args.rechazar, args.nota)
        if args.orden == "procesar":
            return cmd_procesar(ctx, procesos)
        return cmd_verificar(ctx)
    except ErrorPreproceso as fallo:
        print(f"Error: {fallo}")
        return fallo.codigo_salida
