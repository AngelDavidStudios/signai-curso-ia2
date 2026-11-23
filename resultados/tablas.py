"""Reproduce las tablas y la Figura 2 del Entregable 2 a partir de `corridas.csv` y `epocas.csv`.

Para cada configuración del barrido b02 (selección por rotación, research R-05 de TT-22):

1. se toman sus corridas terminadas (3 semillas × 2 rotaciones);
2. la curva de la configuración es la media, época a época, del Top-1 de validación;
3. la época elegida E* es el máximo de esa curva (empates: mayor acierto al 95 % y después la época menor);
4. sus cifras son las medias de las corridas en E*.

Escribe `configuraciones.csv` (todas las configuraciones) y `figura2.csv` (la mejor de cada ronda
completa), e imprime en Markdown las tablas del informe y la comparación pareada.

Uso:
    uv run python resultados/tablas.py
"""

import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

AQUI = Path(__file__).parent
LINEA_BASE = "r01-linea-base"
METRICAS = ("validacion.top1", "validacion.f1_macro", "validacion.acierto_95", "validacion.falsas_95")
CORRIDAS_POR_CONFIGURACION = 6


def leer(nombre: str) -> list[dict]:
    with (AQUI / nombre).open(encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def agregar() -> tuple[list[dict], dict[str, dict]]:
    corridas = [c for c in leer("corridas.csv") if c["barrido"] == "b02" and c["estado"] == "terminada"]
    por_corrida: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for fila in leer("epocas.csv"):
        por_corrida[fila["run_id"]][int(fila["epoca"])] = {m: float(fila[m]) for m in METRICAS if fila[m] != ""}
    grupos: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for corrida in corridas:
        grupos[(int(corrida["ronda"]), corrida["configuracion"])].append(corrida)
    resultado, detalle = [], {}
    for (ronda, nombre), miembros in sorted(grupos.items()):
        series = [por_corrida[c["run_id"]] for c in miembros]
        epocas = sorted(set.intersection(*(set(s) for s in series)))
        media = lambda e, m: st.mean(s[e][m] for s in series)  # noqa: E731
        elegida = max(epocas, key=lambda e: (media(e, "validacion.top1"), media(e, "validacion.acierto_95"), -e))
        top1 = [s[elegida]["validacion.top1"] for s in series]
        por_rotacion = defaultdict(list)
        for corrida, valor in zip(miembros, top1):
            por_rotacion[corrida["rotacion"]].append(valor)
        diferencia = abs(st.mean(por_rotacion["a"]) - st.mean(por_rotacion["b"])) if len(por_rotacion) == 2 else float("nan")
        fila = {
            "ronda": ronda,
            "configuracion": nombre,
            "corridas": len(miembros),
            "epoca_elegida": elegida,
            "epocas": miembros[0]["entrenamiento.epocas"],
            "top1": st.mean(top1),
            "top1_desviacion": st.pstdev(top1),
            "f1_macro": media(elegida, "validacion.f1_macro"),
            "acierto_95": media(elegida, "validacion.acierto_95"),
            "falsas_95": media(elegida, "validacion.falsas_95"),
            "diferencia_rotaciones": diferencia,
            "parametros": miembros[0]["modelo.parametros"],
            "minutos_por_corrida": st.mean(float(c["duracion_s"]) for c in miembros) / 60,
        }
        resultado.append(fila)
        detalle[nombre] = {(c["semilla"], c["rotacion"]): v for c, v in zip(miembros, top1)}
    return resultado, detalle


def escribir(nombre: str, filas: list[dict]) -> None:
    with (AQUI / nombre).open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]), lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)


def f(valor: float) -> str:
    return f"{valor:.3f}".replace(".", ",")


def puntos(valor: float, signo: bool = False) -> str:
    return (f"{100 * valor:+.1f}" if signo else f"{100 * valor:.1f}").replace(".", ",")


def main() -> None:
    configuraciones, detalle = agregar()
    escribir("configuraciones.csv", configuraciones)
    completas = [c for c in configuraciones if c["corridas"] == CORRIDAS_POR_CONFIGURACION]
    mejores = {}
    for c in completas:
        if c["ronda"] not in mejores or c["top1"] > mejores[c["ronda"]]["top1"]:
            mejores[c["ronda"]] = c
    rondas_completas = [r for r in sorted(mejores) if all(
        c["corridas"] == CORRIDAS_POR_CONFIGURACION for c in configuraciones if c["ronda"] == r
    )]
    figura = [mejores[r] for r in rondas_completas]
    escribir("figura2.csv", figura)

    print("## Figura 2: mejor configuración de cada ronda completa\n")
    print("| Ronda | Configuración | Top-1 | Desviación |\n| --- | --- | --- | --- |")
    for c in figura:
        print(f"| {c['ronda']} | {c['configuracion']} | {f(c['top1'])} | {f(c['top1_desviacion'])} |")

    base = next(c for c in configuraciones if c["configuracion"] == LINEA_BASE)
    mejor = max(figura, key=lambda c: c["top1"])
    print(f"\n## Tabla 3.2: línea base frente a {mejor['configuracion']}\n")
    print("| Métrica | Línea base | Mejor |\n| --- | --- | --- |")
    for clave, rotulo in (
        ("top1", "Top-1"),
        ("f1_macro", "F1 macro"),
        ("acierto_95", "Acierto al 95 %"),
        ("falsas_95", "Errores al 95 %"),
    ):
        print(f"| {rotulo} | {f(base[clave])} | {f(mejor[clave])} |")
    print(f"| Diferencia entre rotaciones | {puntos(base['diferencia_rotaciones'])} puntos | {puntos(mejor['diferencia_rotaciones'])} puntos |")
    minutos = lambda c: f"{c['minutos_por_corrida']:.1f}".replace(".", ",")  # noqa: E731
    print(f"| Minutos por corrida | {minutos(base)} | {minutos(mejor)} |")
    print(f"| Época elegida | {base['epoca_elegida']} de {base['epocas']} | {mejor['epoca_elegida']} de {mejor['epocas']} |")

    pares = sorted(set(detalle[LINEA_BASE]) & set(detalle[mejor["configuracion"]]))
    diferencias = [detalle[mejor["configuracion"]][p] - detalle[LINEA_BASE][p] for p in pares]
    t = st.mean(diferencias) / (st.stdev(diferencias) / math.sqrt(len(diferencias)))
    print(
        f"\nComparación pareada ({len(pares)} parejas semilla-rotación): diferencia media "
        f"{puntos(st.mean(diferencias))} puntos (de {puntos(min(diferencias))} a {puntos(max(diferencias))}), "
        f"t({len(pares) - 1}) = {t:.1f}".replace(".", ",")
    )

    print("\n## Tabla 3.3: ablación del aumento de datos (ronda 1)\n")
    print("| Configuración | Top-1 | Cambio (puntos) |\n| --- | --- | --- |")
    for c in sorted((c for c in configuraciones if c["ronda"] == 1), key=lambda c: -c["top1"]):
        print(f"| {c['configuracion']} | {f(c['top1'])} | {puntos(c['top1'] - base['top1'], signo=True)} |")


if __name__ == "__main__":
    main()
