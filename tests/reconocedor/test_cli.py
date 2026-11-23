"""CLI `signai-reconocedor` con el protocolo por rotación (TT-22; contracts/cli.md C-1'…C-6').

Sobre el corpus sintético, los lectores falsos de la partición versión 2 (dos pliegues y su unión),
CPU y el registro en memoria (salvo donde se indica MLflow). Las rondas declaran una forma diminuta y
1–2 épocas.
"""

import json
from pathlib import Path

import pytest
import torch

from signai_ml.reconocedor.cli import main
from signai_ml.reconocedor.modelo import cargar
from signai_ml.reconocedor.registro import EstadoCorrida, RegistroEnMemoria, RegistroMlflow
from signai_ml.ventanas import huellas_corpus

CPU = torch.device("cpu")

RONDA_1 = """
barrido = "b01"
ronda = 1

[base.forma]
capas = 1
dimension = 98
cabezas = 2
intermedia = 16

[base.entrenamiento]
tasa = 3e-3
dropout = 0.0
lote = 8
epocas = 2

[[configuracion]]
nombre = "base"

[[configuracion]]
nombre = "lenta"
entrenamiento = { tasa = 1e-3 }
"""


def declarar(raiz: Path, numero: int, texto: str, barrido: str = "b01") -> Path:
    directorio = raiz / "config" / "reconocedor" / barrido
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"ronda-{numero:02d}.toml"
    ruta.write_text(texto, encoding="utf-8")
    return ruta


@pytest.fixture
def raiz(raiz_sintetica: Path) -> Path:
    declarar(raiz_sintetica, 1, RONDA_1)
    return raiz_sintetica


@pytest.fixture
def ejecutar(raiz: Path, lector_seleccion, lector_final):
    registro = RegistroEnMemoria()

    def _ejecutar(*argv: str, registro_=None) -> int:
        return main(list(argv), raiz=raiz, registro=registro_ or registro, lector_seleccion=lector_seleccion,
                    lector_final=lector_final, dev=CPU)

    _ejecutar.registro = registro
    return _ejecutar


def _seleccion(registro, barrido: str = "b01") -> list:
    return [c for c in registro.corridas(barrido) if c.fase == "seleccion"]


def test_ayuda_sin_orden() -> None:
    with pytest.raises(SystemExit) as salida:
        main([])
    assert salida.value.code == 2


def test_corrida_de_seleccion_con_su_rotacion(ejecutar, capsys, tmp_path: Path) -> None:
    assert ejecutar("corrida", "b01", "base", "--semilla", "215", "--rotacion", "b", "--permitir-sucio") == 0
    salida = capsys.readouterr().out
    assert "base-s215-b" in salida and "época elegida" in salida and "Top-1" in salida
    (corrida,) = ejecutar.registro.corridas("b01")
    assert corrida.estado is EstadoCorrida.TERMINADA
    assert (corrida.fase, corrida.rotacion, corrida.semilla) == ("seleccion", "b", 215)
    assert [p for p, _ in ejecutar.registro.historia(corrida.id, "validacion.top1")] == [1, 2]
    por_clase = json.loads(ejecutar.registro.descargar(corrida.id, "por_clase.json", tmp_path).read_text())
    assert set(por_clase) == {"1-Sint1", "2-Sint2", "3-Sint3", "4-Sint4"}


def test_corrida_sin_rotacion_o_con_una_desconocida() -> None:
    for argv in (["corrida", "b01", "base", "--semilla", "215"],
                 ["corrida", "b01", "base", "--semilla", "215", "--rotacion", "c"]):
        with pytest.raises(SystemExit) as salida:
            main(argv)
        assert salida.value.code == 2


def test_semilla_fuera_de_las_fijas(ejecutar, capsys) -> None:
    assert ejecutar("corrida", "b01", "base", "--semilla", "42", "--rotacion", "a", "--permitir-sucio") == 2
    assert "215, 216, 217" in capsys.readouterr().err


def test_configuracion_no_declarada(ejecutar, capsys) -> None:
    assert ejecutar("corrida", "b01", "otra", "--semilla", "215", "--rotacion", "a", "--permitir-sucio") == 1
    assert "otra" in capsys.readouterr().err


def test_barrido_cerrado(ejecutar, capsys) -> None:
    ejecutar.registro.cerrar_barrido("b01", {})
    assert ejecutar("corrida", "b01", "base", "--semilla", "215", "--rotacion", "a", "--permitir-sucio") == 1
    assert "cerrado" in capsys.readouterr().err
    assert ejecutar.registro.corridas("b01") == []


def test_barrido_cerrado_solo_admite_repetir_al_candidato(ejecutar, capsys) -> None:
    """La selección del candidato se repite sobre el barrido cerrado; la repetición no cuenta en el resumen."""
    assert ejecutar("ronda", "b01", "1", "--permitir-sucio") == 0
    ejecutar.registro.cerrar_barrido("b01", {})
    assert ejecutar("resumen", "b01") == 0
    resumen = ejecutar.registro.resumen("b01")
    linea = next(l for l in resumen.splitlines() if l.startswith("- Candidato:"))
    configuracion = linea.split()[2]
    semilla = linea.split("semilla ")[1].split(",")[0]
    original = next(c for c in _seleccion(ejecutar.registro) if c.nombre == f"{configuracion}-s{semilla}-a")
    otra = "lenta" if configuracion == "base" else "base"
    capsys.readouterr()

    assert ejecutar("corrida", "b01", otra, "--semilla", semilla, "--rotacion", "a", "--permitir-sucio") == 1
    assert "candidato" in capsys.readouterr().err
    assert ejecutar("corrida", "b01", configuracion, "--semilla", semilla, "--rotacion", "a", "--permitir-sucio") == 0
    repetidas = [c for c in _seleccion(ejecutar.registro) if c.nombre == original.nombre and c.id != original.id]
    assert len(repetidas) == 1 and repetidas[0].estado is EstadoCorrida.TERMINADA
    assert repetidas[0].etiquetas["repeticion.de"] == original.id
    assert ejecutar("resumen", "b01") == 0
    assert ejecutar.registro.resumen("b01") == resumen


def test_corrida_fallida_por_divergencia(ejecutar, monkeypatch) -> None:
    from signai_ml.reconocedor import bucle

    monkeypatch.setattr(bucle, "_perdida_lote", lambda *a, **k: torch.tensor(float("nan"), requires_grad=True))
    assert ejecutar("corrida", "b01", "base", "--semilla", "215", "--rotacion", "a", "--permitir-sucio") == 1
    (corrida,) = ejecutar.registro.corridas("b01")
    assert corrida.estado is EstadoCorrida.FALLIDA and "época 1" in corrida.etiquetas["motivo"]


def test_corrida_cortada_queda_interrumpida(ejecutar, monkeypatch) -> None:
    from signai_ml.reconocedor import cli

    def cortar(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "entrenar", cortar)
    with pytest.raises(KeyboardInterrupt):
        ejecutar("corrida", "b01", "base", "--semilla", "215", "--rotacion", "a", "--permitir-sucio")
    (corrida,) = ejecutar.registro.corridas("b01")
    assert corrida.estado is EstadoCorrida.INTERRUMPIDA


def test_corrida_con_mlflow(raiz: Path, lector_seleccion, tmp_path: Path) -> None:
    registro = RegistroMlflow(tmp_path / "datos")
    assert main(["corrida", "b01", "base", "--semilla", "216", "--rotacion", "a", "--permitir-sucio"],
                raiz=raiz, registro=registro, lector_seleccion=lector_seleccion, dev=CPU) == 0
    (corrida,) = registro.corridas("b01")
    assert corrida.estado is EstadoCorrida.TERMINADA
    assert [paso for paso, _ in registro.historia(corrida.id, "validacion.top1")] == [1, 2]


# --- Ronda, resumen, cerrar y candidato ------------------------------------------------------------

RONDA_2 = RONDA_1.replace("ronda = 1", "ronda = 2").replace(
    '[[configuracion]]\nnombre = "base"\n\n[[configuracion]]\nnombre = "lenta"\nentrenamiento = { tasa = 1e-3 }\n',
    '[[configuracion]]\nnombre = "capas-2"\nforma = { capas = 2 }\n',
)
RONDA_3 = RONDA_1.replace("ronda = 1", "ronda = 3").replace(
    '[[configuracion]]\nnombre = "base"\n\n[[configuracion]]\nnombre = "lenta"\nentrenamiento = { tasa = 1e-3 }\n',
    '[[configuracion]]\nnombre = "intermedia-32"\nforma = { intermedia = 32 }\n',
)


def sembrar(registro, raiz: Path, numero: int, configuracion: str, tops: tuple[float, float, float],
            acierto: float = 0.8, tmp: Path | None = None, barrido: str = "b01",
            parametros: int = 50_000) -> None:
    """Las 6 corridas de selección terminadas, con una época de validación puesta a mano (sin entrenar)."""
    from signai_ml.reconocedor.ronda import leer_ronda

    ronda = leer_ronda(raiz / "config" / "reconocedor" / barrido / f"ronda-{numero:02d}.toml")
    huellas = {**huellas_corpus(raiz), "particion.huella": "f" * 64}
    for semilla, top1 in zip((215, 216, 217), tops):
        for rotacion in ("a", "b"):
            id_corrida = registro.abrir(barrido, f"{configuracion}-s{semilla}-{rotacion}", {"semilla": semilla}, {
                "barrido": barrido, "ronda": str(numero), "ronda.huella": ronda.huella,
                "configuracion": configuracion, "fase": "seleccion", "rotacion": rotacion,
                "modelo.parametros": str(parametros), "codigo.commit": "abc123", **huellas,
            })
            registro.epoca(id_corrida, 1, {
                "validacion.top1": top1, "validacion.f1_macro": top1, "validacion.acierto_95": acierto,
                "validacion.falsas_95": 0.01,
            })
            registro.terminar(id_corrida, {"elegida.top1": top1, "elegida.epoca": 1}, [], {})


@pytest.fixture
def barrido_estancado(ejecutar, raiz: Path, tmp_path: Path):
    declarar(raiz, 2, RONDA_2)
    declarar(raiz, 3, RONDA_3)
    sembrar(ejecutar.registro, raiz, 1, "base", (0.80, 0.80, 0.80), tmp=tmp_path)
    sembrar(ejecutar.registro, raiz, 1, "lenta", (0.70, 0.70, 0.70), tmp=tmp_path)
    sembrar(ejecutar.registro, raiz, 2, "capas-2", (0.802, 0.803, 0.804), tmp=tmp_path)
    sembrar(ejecutar.registro, raiz, 3, "intermedia-32", (0.80, 0.81, 0.805), tmp=tmp_path)
    return ejecutar


def test_ronda_lanza_cada_configuracion_con_3_semillas_y_2_rotaciones_y_no_repite(ejecutar, capsys) -> None:
    assert ejecutar("ronda", "b01", "1", "--permitir-sucio") == 0
    nombres = sorted(c.nombre for c in ejecutar.registro.corridas("b01"))
    assert nombres == sorted(f"{c}-s{s}-{r}" for c in ("base", "lenta") for s in (215, 216, 217) for r in "ab")
    assert "12 corrida(s) pendientes" in capsys.readouterr().out
    assert ejecutar.registro.resumen("b01").startswith("# Barrido b01")
    assert ejecutar("ronda", "b01", "1", "--permitir-sucio") == 0
    assert len(ejecutar.registro.corridas("b01")) == 12


def test_ronda_marca_interrumpidas_y_las_repite(ejecutar, raiz: Path) -> None:
    from signai_ml.reconocedor.ronda import leer_ronda

    huella = leer_ronda(raiz / "config" / "reconocedor" / "b01" / "ronda-01.toml").huella
    colgada = ejecutar.registro.abrir("b01", "base-s215-a", {"semilla": 215}, {
        "ronda": "1", "ronda.huella": huella, "configuracion": "base", "fase": "seleccion", "rotacion": "a",
    })
    assert ejecutar("ronda", "b01", "1", "--permitir-sucio") == 0
    estados = [(c.id, c.estado) for c in ejecutar.registro.corridas("b01") if c.nombre == "base-s215-a"]
    assert (colgada, EstadoCorrida.INTERRUMPIDA) in estados
    assert sum(e is EstadoCorrida.TERMINADA for _, e in estados) == 1


def test_ronda_modificada_tras_lanzarla(ejecutar, raiz: Path, capsys) -> None:
    assert ejecutar("corrida", "b01", "base", "--semilla", "215", "--rotacion", "a", "--permitir-sucio") == 0
    declarar(raiz, 1, RONDA_1.replace("epocas = 2", "epocas = 3"))
    assert ejecutar("ronda", "b01", "1", "--permitir-sucio") == 1
    assert "cambió después de lanzarse" in capsys.readouterr().err


def test_ronda_que_no_existe(ejecutar, capsys) -> None:
    assert ejecutar("ronda", "b01", "5", "--permitir-sucio") == 1
    assert "ronda 5" in capsys.readouterr().err


def test_resumen(barrido_estancado, capsys) -> None:
    assert barrido_estancado("resumen", "b01") == 0
    salida = capsys.readouterr().out
    assert "4 configuraciones" in salida and "3 rondas" in salida
    assert "Candidato: intermedia-32 (1 época, semilla 216" in salida
    assert barrido_estancado.registro.resumen("b01") == salida.rstrip("\n") + "\n"


def test_cerrar_se_niega_sin_estancamiento(ejecutar, raiz: Path, tmp_path: Path, capsys) -> None:
    declarar(raiz, 2, RONDA_2)
    sembrar(ejecutar.registro, raiz, 1, "base", (0.80, 0.80, 0.80), tmp=tmp_path)
    sembrar(ejecutar.registro, raiz, 1, "lenta", (0.70, 0.70, 0.70), tmp=tmp_path)
    sembrar(ejecutar.registro, raiz, 2, "capas-2", (0.80, 0.80, 0.80), tmp=tmp_path)
    assert ejecutar("cerrar", "b01") == 1
    assert "no se ha estancado" in capsys.readouterr().err
    assert ejecutar.registro.estado_barrido("b01") == {}


def test_cerrar_sin_elegibles(ejecutar, raiz: Path, tmp_path: Path, capsys) -> None:
    sembrar(ejecutar.registro, raiz, 1, "base", (0.80, 0.80, 0.80), parametros=3_000_000, tmp=tmp_path)
    assert ejecutar("cerrar", "b01") == 1
    assert "elegible" in capsys.readouterr().err


def test_cerrar_y_despues_nada_mas(barrido_estancado, capsys) -> None:
    assert barrido_estancado("cerrar", "b01") == 0
    estado = barrido_estancado.registro.estado_barrido("b01")
    assert estado["barrido.estado"] == "cerrado" and estado["cierre.ronda"] == "3"
    assert float(estado["cierre.mejor_antes"]) == pytest.approx(0.80)
    assert float(estado["cierre.mejor_despues"]) == pytest.approx(0.805)
    assert barrido_estancado("ronda", "b01", "1", "--permitir-sucio") == 1
    assert barrido_estancado("corrida", "b01", "base", "--semilla", "215", "--rotacion", "a", "--permitir-sucio") == 1
    assert "cerrado" in capsys.readouterr().err


def test_candidato_solo_con_barrido_cerrado(barrido_estancado, raiz: Path, capsys) -> None:
    assert barrido_estancado("candidato", "b01", "--permitir-sucio") == 1
    assert "cerrar" in capsys.readouterr().err
    assert not (raiz / "modelos" / "reconocedor").exists()


def test_candidato_entrena_el_final_y_escribe_la_ficha_version_2(barrido_estancado, raiz: Path, capsys) -> None:
    assert barrido_estancado("cerrar", "b01") == 0
    assert barrido_estancado("candidato", "b01", "--permitir-sucio") == 0
    destino = raiz / "modelos" / "reconocedor"
    ficha = json.loads((destino / "candidato.json").read_text(encoding="utf-8"))
    assert ficha["version_candidato"] == 2 and ficha["protocolo"] == "rotacion-por-grabacion"
    assert (ficha["barrido"], ficha["configuracion"], ficha["ronda"]) == ("b01", "intermedia-32", 3)
    assert (ficha["semilla"], ficha["epocas_elegidas"]) == (216, 1)
    assert ficha["seleccion"]["top1"] == pytest.approx(0.805)
    assert ficha["seleccion"]["diferencia_rotaciones"] == pytest.approx(0.0)
    assert ficha["seleccion"]["por_rotacion"] == {"a": pytest.approx(0.805), "b": pytest.approx(0.805)}
    assert len(ficha["seleccion"]["corridas"]) == 6
    assert ficha["final"]["clips"] == 32 and ficha["final"]["run_id"]
    assert ficha["huellas"] == {**huellas_corpus(raiz), "particion.huella": "f" * 64}
    assert ficha["estado"] == "pendiente" and ficha["parametros"] > 0
    (final,) = [c for c in barrido_estancado.registro.corridas("b01") if c.fase == "final"]
    assert final.id == ficha["final"]["run_id"] and final.estado is EstadoCorrida.TERMINADA
    _, meta = cargar(destino / "modelo.pt")
    assert meta["corrida"]["epoca"] == 1 and len(meta["glosas"]) == 4
    assert "uv run dvc add modelos/reconocedor" in capsys.readouterr().out


def test_candidato_reutiliza_el_final_ya_entrenado(barrido_estancado) -> None:
    assert barrido_estancado("cerrar", "b01") == 0
    assert barrido_estancado("candidato", "b01", "--permitir-sucio") == 0
    assert barrido_estancado("candidato", "b01", "--permitir-sucio") == 0
    assert len([c for c in barrido_estancado.registro.corridas("b01") if c.fase == "final"]) == 1


def test_candidato_no_pisa_otro_barrido_sin_versionar(barrido_estancado, raiz: Path, capsys) -> None:
    destino = raiz / "modelos" / "reconocedor"
    destino.mkdir(parents=True)
    (destino / "candidato.json").write_text(json.dumps({"barrido": "b00"}), encoding="utf-8")
    assert barrido_estancado("cerrar", "b01") == 0
    assert barrido_estancado("candidato", "b01", "--permitir-sucio") == 1
    assert "b00" in capsys.readouterr().err
    (raiz / "modelos" / "reconocedor.dvc").write_text("outs: []\n", encoding="utf-8")
    assert barrido_estancado("candidato", "b01", "--permitir-sucio") == 0


# --- Guardas de código y de datos ------------------------------------------------------------------


def _repositorio(raiz: Path) -> None:
    import subprocess

    def git(*argumentos: str) -> None:
        subprocess.run(["git", "-C", str(raiz), *argumentos], check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.email=prueba@example.invalid", "-c", "user.name=Prueba", "add", "-A")
    git("-c", "user.email=prueba@example.invalid", "-c", "user.name=Prueba", "commit", "-q", "-m", "inicial")


def test_cambios_sin_confirmar_bloquean(ejecutar, raiz: Path, capsys) -> None:
    _repositorio(raiz)
    (raiz / "sin-confirmar.txt").write_text("x", encoding="utf-8")
    assert ejecutar("corrida", "b01", "base", "--semilla", "215", "--rotacion", "a") == 1
    assert ejecutar("ronda", "b01", "1") == 1
    assert "sin confirmar" in capsys.readouterr().err
    assert ejecutar.registro.corridas("b01") == []


def test_arbol_limpio_registra_el_commit(ejecutar, raiz: Path) -> None:
    import subprocess

    _repositorio(raiz)
    assert ejecutar("corrida", "b01", "base", "--semilla", "215", "--rotacion", "a") == 0
    (corrida,) = ejecutar.registro.corridas("b01")
    commit = subprocess.run(["git", "-C", str(raiz), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert corrida.etiquetas["codigo.commit"] == commit and corrida.etiquetas["codigo.sucio"] == "false"


def test_fuera_de_un_repositorio_sin_permiso_se_niega(ejecutar, capsys) -> None:
    assert ejecutar("corrida", "b01", "base", "--semilla", "215", "--rotacion", "a") == 1
    assert "sin confirmar" in capsys.readouterr().err


def test_datos_distintos_dentro_del_barrido(ejecutar, raiz: Path, capsys) -> None:
    assert ejecutar("corrida", "b01", "base", "--semilla", "215", "--rotacion", "a", "--permitir-sucio") == 0
    indice = raiz / "datos" / "ventanas" / "indice.csv"
    indice.write_text(indice.read_text(encoding="utf-8").replace("SINT_1", "SINT_2"), encoding="utf-8")
    assert ejecutar("corrida", "b01", "base", "--semilla", "216", "--rotacion", "a", "--permitir-sucio") == 1
    assert ejecutar("ronda", "b01", "1", "--permitir-sucio") == 1
    assert "corpus.sha256_indice" in capsys.readouterr().err
    assert len(ejecutar.registro.corridas("b01")) == 1
