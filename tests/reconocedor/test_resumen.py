"""Resumen del barrido por rotación (TT-22; contracts/cli.md C-3')."""

from signai_ml.reconocedor.resumen import generar_resumen
from signai_ml.reconocedor.seleccion import ResultadoRotacion, agregar, elegir_candidato

ORDEN = {"base": (1, 0), "grande": (1, 1)}


def _res(cfg: str, semilla: int, rotacion: str, top1: float, parametros: int) -> ResultadoRotacion:
    curvas = {"top1": (0.5, top1), "f1_macro": (0.5, top1 - 0.01), "acierto_95": (0.5, 0.8), "falsas_95": (0.1, 0.02)}
    return ResultadoRotacion(f"id-{cfg}-{semilla}-{rotacion}", f"{cfg}-s{semilla}-{rotacion}", cfg, semilla,
                             rotacion, 1, parametros, curvas)


def test_cabecera_y_filas() -> None:
    resultados = [_res("base", s, rot, t, 381_121) for s in (215, 216, 217) for rot, t in (("a", 0.84), ("b", 0.80))]
    resultados += [_res("grande", s, rot, 0.9, 3_000_000) for s in (215, 216, 217) for rot in ("a", "b")]
    agregados = agregar(resultados, ORDEN)
    texto = generar_resumen("b02", agregados, [(1, 0.82)], "abierto", elegir_candidato(agregados))
    assert texto.startswith("# Barrido b02")
    assert "2 configuraciones" in texto and "1 ronda" in texto and "abierto" in texto
    assert "3 semillas × 2 rotaciones" in texto
    assert "| base |" in texto and "| grande |" in texto
    assert "| 0.820 |" in texto  # Top-1 medio de base en E*
    assert "0.840 / 0.800" in texto and "0.040" in texto  # por rotación y diferencia
    assert "381.121" in texto and "744.4 KB" in texto
    assert "no elegible" in texto and "parámetros" in texto
    assert "ronda 1: 0.820" in texto
    assert "Candidato: base (2 épocas, semilla 215" in texto


def test_sin_candidato() -> None:
    texto = generar_resumen("b02", [], [], "abierto", None)
    assert "Candidato: ninguno" in texto and "0 configuraciones" in texto
