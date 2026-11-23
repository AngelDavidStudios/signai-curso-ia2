"""Resumen legible del barrido por rotación (TT-22; contracts/cli.md C-3').

Se regenera desde el registro tras cada ronda; nunca se edita a mano.
"""

from collections.abc import Sequence

from . import LIMITE_PARAMETROS, SEMILLAS
from .seleccion import ROTACIONES, Agregado


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _cuenta(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def generar_resumen(
    barrido: str,
    agregados: Sequence[Agregado],
    mejores: Sequence[tuple[int, float | None]],
    estado: str,
    candidato: Agregado | None,
) -> str:
    rondas = {a.ronda for a in agregados} | {r for r, _ in mejores}
    lineas = [
        f"# Barrido {barrido}",
        "",
        f"- Estado: {estado}",
        f"- {_cuenta(len(agregados), 'configuración', 'configuraciones')} con sus {len(SEMILLAS)} semillas × "
        f"{len(ROTACIONES)} rotaciones en {_cuenta(len(rondas), 'ronda', 'rondas')}",
        f"- Elegible: ≤ {_miles(LIMITE_PARAMETROS)} parámetros (el acierto al 95 % se informa y desempata)",
        "- Cifras: medias de las 6 corridas en las épocas elegidas (E*), validando con el pliegue que no entrenó",
        "- Mejor media elegible de Top-1 tras cada ronda: "
        + (", ".join(f"ronda {r}: {'—' if v is None else f'{v:.3f}'}" for r, v in mejores) or "—"),
    ]
    if candidato:
        lineas.append(
            f"- Candidato: {candidato.configuracion} ({_cuenta(candidato.epocas_elegidas, 'época', 'épocas')}, "
            f"semilla {candidato.semilla_final}, Top-1 {candidato.medias['top1']:.3f}, diferencia entre rotaciones "
            f"{candidato.diferencia_rotaciones:.3f})"
        )
    else:
        lineas.append("- Candidato: ninguno")
    lineas += [
        "",
        "| Configuración | Ronda | E* | Top-1 | A / B | Diferencia | F1 macro | Acierto 95 % | Falsas 95 % | Semilla final | Parámetros | Tamaño | Elegible |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in agregados:
        elegible = "sí" if a.elegible else f"no elegible: {a.motivo}"
        lineas.append(
            f"| {a.configuracion} | {a.ronda} | {a.epocas_elegidas} | {a.medias['top1']:.3f} | "
            f"{a.por_rotacion['a']:.3f} / {a.por_rotacion['b']:.3f} | {a.diferencia_rotaciones:.3f} | "
            f"{a.medias['f1_macro']:.3f} | {a.medias['acierto_95']:.3f} | {a.medias['falsas_95']:.3f} | "
            f"{a.semilla_final} | {_miles(a.parametros)} | {a.parametros * 2 / 1024:.1f} KB | {elegible} |"
        )
    return "\n".join(lineas) + "\n"
