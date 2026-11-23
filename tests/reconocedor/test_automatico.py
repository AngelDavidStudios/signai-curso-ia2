"""Ciclo automático del barrido (TT-14 R-16; TT-22: primera ronda de b02, research R-07; contracts/cli.md C-6')."""

import subprocess
from pathlib import Path

import pytest
import torch

from signai_ml.reconocedor import automatico, cli
from signai_ml.reconocedor.automatico import (
    Variacion,
    confirmar_con_git,
    generar_primera_ronda,
    generar_siguiente_ronda,
)
from signai_ml.reconocedor.registro import RegistroEnMemoria
from signai_ml.reconocedor.ronda import leer_barrido, leer_ronda
from signai_ml.reconocedor.seleccion import ResultadoRotacion, agregar

from .test_cli import sembrar

CPU = torch.device("cpu")
PEQUENO = (
    Variacion("entrenamiento", "dropout", (0.0, 0.05, 0.1, 0.2, 0.3)),
    Variacion("forma", "capas", (1, 2, 3)),
    "aumento.escala",
)


def _escribir(directorio: Path, numero: int, texto: str) -> Path:
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"ronda-{numero:02d}.toml"
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def _agregados(rondas, mejor: str, parametros: int = 381_121):
    def curvas(top1: float) -> dict:
        return {"top1": (top1,), "f1_macro": (0.9,), "acierto_95": (0.8,), "falsas_95": (0.0,)}

    resultados = [
        ResultadoRotacion(f"{c.nombre}-{s}-{rot}", f"{c.nombre}-s{s}-{rot}", c.nombre, s, rot, r.numero, parametros,
                          curvas(0.95 if c.nombre == mejor else 0.90))
        for r in rondas for c in r.configuraciones for s in (215, 216, 217) for rot in ("a", "b")
    ]
    orden = {c.nombre: (r.numero, i) for r in rondas for i, c in enumerate(r.configuraciones)}
    return agregar(resultados, orden)


# --- Generación de rondas ------------------------------------------------------------------------


def test_primera_ronda_es_la_de_research_r07(tmp_path: Path) -> None:
    ronda = leer_ronda(_escribir(tmp_path / "b02", 1, generar_primera_ronda("b02")))
    assert (ronda.barrido, ronda.numero) == ("b02", 1)
    por_nombre = {c.nombre: c for c in ronda.configuraciones}
    assert list(por_nombre) == ["r01-linea-base", "r01-espacial", "r01-rotacion", "r01-temporal",
                                "r01-ocultacion", "r01-todo"]
    assert all(c.entrenamiento.epocas == 60 and c.entrenamiento.dropout == 0.2 for c in ronda.configuraciones)
    assert all(c.forma.capas == 4 and c.forma.dimension == 98 for c in ronda.configuraciones)
    aumentos = {n: dict(c.aumento) for n, c in por_nombre.items()}
    assert aumentos["r01-linea-base"] == {}  # la configuración de la versión 1 (R-04)
    assert set(aumentos["r01-espacial"]) == {"desplazamiento", "escala"}
    assert set(aumentos["r01-rotacion"]) == {"rotacion"}
    assert aumentos["r01-rotacion"]["rotacion"].intensidad == 0.15
    assert set(aumentos["r01-temporal"]) == {"velocidad", "recorte_temporal"}
    oculta = aumentos["r01-ocultacion"]["ocultacion"]
    assert (oculta.probabilidad, oculta.intensidad) == (0.3, 0.1)
    assert set(aumentos["r01-todo"]) == {"desplazamiento", "escala", "rotacion", "ocultacion", "velocidad",
                                         "recorte_temporal"}


def test_siguiente_ronda_varia_una_dimension_de_la_mejor(tmp_path: Path) -> None:
    directorio = tmp_path / "b01"
    _escribir(directorio, 1, generar_primera_ronda("b01"))
    rondas = leer_barrido(directorio)
    texto = generar_siguiente_ronda("b01", 2, rondas, _agregados(rondas, "r01-rotacion"))
    _escribir(directorio, 2, texto)
    rondas = leer_barrido(directorio)  # valida que no repite nombres ni valores
    mejor = next(c for c in rondas[0].configuraciones if c.nombre == "r01-rotacion")
    nuevas = rondas[1].configuraciones
    assert 1 <= len(nuevas) <= automatico.POR_RONDA
    for cfg in nuevas:
        assert cfg.nombre.startswith("r02-")
        cambios = sum(a != b for a, b in zip(cfg.aplanar().items(), mejor.aplanar().items()))
        distintas = set(cfg.aplanar().items()) ^ set(mejor.aplanar().items())
        assert distintas and (cambios <= 2 or "aumento" in str(distintas))


def test_respeta_el_limite_de_parametros(tmp_path: Path) -> None:
    directorio = tmp_path / "b01"
    _escribir(directorio, 1, generar_primera_ronda("b01"))
    rondas = leer_barrido(directorio)
    texto = generar_siguiente_ronda("b01", 2, rondas, _agregados(rondas, "r01-linea-base"), limite=390_000)
    ronda = leer_ronda(_escribir(directorio, 2, texto))
    assert all(c.forma.capas <= 4 and c.forma.dimension <= 98 for c in ronda.configuraciones)


def test_sin_elegibles_parte_de_la_mejor_media(tmp_path: Path) -> None:
    directorio = tmp_path / "b01"
    _escribir(directorio, 1, generar_primera_ronda("b01"))
    rondas = leer_barrido(directorio)
    texto = generar_siguiente_ronda("b01", 2, rondas, _agregados(rondas, "r01-linea-base", parametros=3_000_000), PEQUENO)
    assert "a partir de r01-linea-base" in texto


def test_rota_las_dimensiones_entre_rondas(tmp_path: Path) -> None:
    directorio = tmp_path / "b01"
    _escribir(directorio, 1, generar_primera_ronda("b01"))
    rondas = leer_barrido(directorio)
    agregados = _agregados(rondas, "r01-linea-base")
    a = generar_siguiente_ronda("b01", 2, rondas, agregados, por_ronda=1)
    b = generar_siguiente_ronda("b01", 3, rondas, agregados, por_ronda=1)
    assert a.split("[[configuracion]]")[1] != b.split("[[configuracion]]")[1]


def test_espacio_agotado(tmp_path: Path) -> None:
    directorio = tmp_path / "b01"
    _escribir(directorio, 1, generar_primera_ronda("b01"))
    rondas = leer_barrido(directorio)
    assert generar_siguiente_ronda("b01", 2, rondas, _agregados(rondas, "r01-linea-base"), vecindario=()) is None


# --- El ciclo ------------------------------------------------------------------------------------


@pytest.fixture
def ciclo(raiz_sintetica: Path, lector_seleccion, lector_final, monkeypatch):
    """Ciclo con forma diminuta, 1 época, vecindario pequeño y commits y copias falsos."""
    monkeypatch.setattr(automatico, "FORMA_BASE", {"capas": 1, "dimension": 98, "cabezas": 2, "intermedia": 16})
    monkeypatch.setattr(automatico, "EPOCAS_BASE", 1)
    monkeypatch.setattr(automatico, "VECINDARIO", PEQUENO)
    registro = RegistroEnMemoria()
    confirmados: list[str] = []
    copias: list[int] = []

    def lanzar(*argv: str) -> int:
        return cli.main(["automatico", "b01", "--permitir-sucio", *argv], raiz=raiz_sintetica, registro=registro,
                        lector_seleccion=lector_seleccion, lector_final=lector_final, dev=CPU,
                        confirmar=lambda raiz, ruta, msj: confirmados.append(msj),
                        copiar=lambda: copias.append(1) or 0)

    lanzar.registro, lanzar.confirmados, lanzar.copias, lanzar.raiz = registro, confirmados, copias, raiz_sintetica
    return lanzar


def test_ciclo_real_declara_confirma_entrena_y_copia(ciclo) -> None:
    assert ciclo("--max-rondas", "2") == 0
    directorio = ciclo.raiz / "config" / "reconocedor" / "b01"
    assert (directorio / "ronda-01.toml").is_file() and (directorio / "ronda-02.toml").is_file()
    assert ciclo.confirmados[:2] == ["SP-372 · Ronda 1 de b01 declarada", "SP-372 · Ronda 2 de b01 declarada"]
    assert len(ciclo.copias) == 2
    rondas = {c.ronda for c in ciclo.registro.corridas("b01")}
    assert rondas == {1, 2}


def test_ciclo_reanuda_la_ronda_a_medias(ciclo) -> None:
    assert ciclo("--max-rondas", "1") == 0
    ronda_2 = len([c for c in ciclo.registro.corridas("b01") if c.ronda == 2])
    assert ronda_2 == 0  # la ronda 2 se declaró pero no se lanzó
    assert ciclo("--max-rondas", "1") == 0
    assert {c.ronda for c in ciclo.registro.corridas("b01")} == {1, 2}


def test_ciclo_cierra_y_escribe_el_candidato(ciclo, monkeypatch, tmp_path: Path) -> None:
    def ronda_sembrada(ctx, barrido, numero, permitir_sucio):
        ronda = leer_barrido(ctx.raiz / "config" / "reconocedor" / barrido)[numero - 1]
        for cfg in ronda.configuraciones:
            sembrar(ctx.registro, ctx.raiz, numero, cfg.nombre, (0.90, 0.90, 0.90), tmp=tmp_path)
        return 0

    monkeypatch.setattr(cli, "cmd_ronda", ronda_sembrada)
    assert ciclo() == 0
    assert ciclo.registro.estado_barrido("b01")["barrido.estado"] == "cerrado"
    assert (ciclo.raiz / "modelos" / "reconocedor" / "candidato.json").is_file()
    assert len(ciclo.copias) >= 3
    assert ciclo() == 0  # ya cerrado: no hace nada más


def test_ciclo_se_detiene_si_una_corrida_falla(ciclo, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "cmd_ronda", lambda *a, **k: 1)
    assert ciclo() == 1
    assert "fallaron" in capsys.readouterr().err


def test_ciclo_con_el_espacio_agotado(ciclo, monkeypatch, tmp_path: Path, capsys) -> None:
    def ronda_sembrada(ctx, barrido, numero, permitir_sucio):
        ronda = leer_barrido(ctx.raiz / "config" / "reconocedor" / barrido)[numero - 1]
        for i, cfg in enumerate(ronda.configuraciones):
            sembrar(ctx.registro, ctx.raiz, numero, cfg.nombre, (0.80 + i / 100,) * 3, tmp=tmp_path)
        return 0

    monkeypatch.setattr(cli, "cmd_ronda", ronda_sembrada)
    monkeypatch.setattr(automatico, "VECINDARIO", ())
    assert ciclo() == 1
    assert "agotado" in capsys.readouterr().err


def test_ciclo_exige_el_arbol_limpio(raiz_sintetica: Path, lector_seleccion, capsys) -> None:
    assert cli.main(["automatico", "b01"], raiz=raiz_sintetica, registro=RegistroEnMemoria(),
                    lector_seleccion=lector_seleccion, dev=CPU) == 1
    assert "sin confirmar" in capsys.readouterr().err


def test_confirmar_con_git_solo_el_fichero_de_la_ronda(tmp_path: Path, monkeypatch) -> None:
    for variable, valor in (("GIT_AUTHOR_NAME", "P"), ("GIT_AUTHOR_EMAIL", "p@example.invalid"),
                            ("GIT_COMMITTER_NAME", "P"), ("GIT_COMMITTER_EMAIL", "p@example.invalid")):
        monkeypatch.setenv(variable, valor)
    def git(*argumentos: str) -> str:
        return subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=p@example.invalid", "-c",
                               "user.name=P", *argumentos], check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "inicial")
    ruta = _escribir(tmp_path / "config" / "reconocedor" / "b01", 1, "x = 1\n")
    confirmar_con_git(tmp_path, ruta, "SP-372 · Ronda 1 de b01 declarada")
    assert git("log", "-1", "--format=%s").strip() == "SP-372 · Ronda 1 de b01 declarada"
    assert git("status", "--porcelain").strip() == ""
    assert "Co-Authored-By" not in git("log", "-1", "--format=%B")


def test_preparar_copia_pide_la_clave_una_vez(monkeypatch) -> None:
    from types import SimpleNamespace

    pedidas: list[str] = []
    llamadas: list = []

    def crear_contexto(_raiz):
        return SimpleNamespace(clave_tecleada=lambda mensaje: pedidas.append(mensaje) or "clave-de-papel",
                               pedir_clave=None)

    monkeypatch.setattr("signai_ml.copia.cli.crear_contexto", crear_contexto)
    monkeypatch.setattr("signai_ml.copia.cli.main",
                        lambda argv, contexto: llamadas.append((argv, contexto.pedir_clave("¿clave?"))) or 0)
    copiar = automatico.preparar_copia()
    assert copiar() == 0 and copiar() == 0
    assert len(pedidas) == 1
    assert llamadas == [(["actualizar"], "clave-de-papel")] * 2
