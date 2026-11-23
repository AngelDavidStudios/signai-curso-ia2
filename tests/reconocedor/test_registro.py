"""Puerto del registro y sus dos implementaciones (contracts/registro.md; research R-10).

La misma batería pasa sobre el registro en memoria (el que usan las pruebas del bucle) y sobre el de
MLflow (el real), con un almacén temporal.
"""

import json
from pathlib import Path

import pytest

from signai_ml.reconocedor.registro import EstadoCorrida, RegistroEnMemoria, RegistroMlflow


@pytest.fixture(params=["memoria", "mlflow"])
def registro(request: pytest.FixtureRequest, tmp_path: Path):
    return RegistroEnMemoria() if request.param == "memoria" else RegistroMlflow(tmp_path / "datos")


ETIQUETAS = {"barrido": "b01", "ronda": "1", "configuracion": "base", "corpus.sha256_ventanas": "a" * 64}


def _abrir(registro, semilla: int = 215, configuracion: str = "base") -> str:
    return registro.abrir(
        "b01", f"{configuracion}-s{semilla}", {"forma.capas": 4, "semilla": semilla},
        {**ETIQUETAS, "configuracion": configuracion},
    )


def test_abrir_deja_la_corrida_en_curso(registro) -> None:
    id_corrida = _abrir(registro)
    (corrida,) = registro.corridas("b01")
    assert corrida.id == id_corrida
    assert corrida.nombre == "base-s215"
    assert corrida.estado is EstadoCorrida.EN_CURSO
    assert corrida.configuracion == "base" and corrida.semilla == 215 and corrida.ronda == 1
    assert corrida.parametros["forma.capas"] == "4"
    assert corrida.etiquetas["corpus.sha256_ventanas"] == "a" * 64


def test_terminar_con_metricas_artefactos_y_etiquetas(registro, tmp_path: Path) -> None:
    id_corrida = _abrir(registro)
    registro.epoca(id_corrida, 1, {"validacion.top1": 0.5})
    artefacto = tmp_path / "por_clase.json"
    artefacto.write_text(json.dumps({"1-Yo": 1.0}), encoding="utf-8")
    registro.terminar(id_corrida, {"elegida.top1": 0.9}, [artefacto], {"dispositivo": "cpu"})
    (corrida,) = registro.corridas("b01")
    assert corrida.estado is EstadoCorrida.TERMINADA
    assert corrida.metricas["elegida.top1"] == 0.9
    assert corrida.etiquetas["dispositivo"] == "cpu"
    destino = registro.descargar(id_corrida, "por_clase.json", tmp_path / "bajada")
    assert json.loads(destino.read_text(encoding="utf-8")) == {"1-Yo": 1.0}
    assert registro.historia(id_corrida, "validacion.top1") == [(1, 0.5)]


def test_fallar_guarda_el_motivo(registro) -> None:
    id_corrida = _abrir(registro)
    registro.fallar(id_corrida, "pérdida no finita en la época 3")
    (corrida,) = registro.corridas("b01")
    assert corrida.estado is EstadoCorrida.FALLIDA
    assert corrida.etiquetas["motivo"] == "pérdida no finita en la época 3"


def test_marcar_interrumpidas_solo_toca_las_en_curso(registro) -> None:
    en_curso = _abrir(registro, 215)
    terminada = _abrir(registro, 216)
    registro.terminar(terminada, {"elegida.top1": 0.8}, [], {})
    assert registro.marcar_interrumpidas("b01") == [en_curso]
    estados = {c.semilla: c.estado for c in registro.corridas("b01")}
    assert estados == {215: EstadoCorrida.INTERRUMPIDA, 216: EstadoCorrida.TERMINADA}
    assert {c.semilla: c.etiquetas.get("motivo") for c in registro.corridas("b01")}[215] == "interrumpida"


def test_barrido_sin_corridas(registro) -> None:
    assert registro.corridas("b99") == []
    assert registro.marcar_interrumpidas("b99") == []
    assert registro.estado_barrido("b99") == {}


def test_cerrar_barrido_y_resumen(registro) -> None:
    _abrir(registro)
    registro.cerrar_barrido("b01", {"cierre.ronda": "3"})
    assert registro.estado_barrido("b01") == {"barrido.estado": "cerrado", "cierre.ronda": "3"}
    registro.guardar_resumen("b01", "# Resumen\n")
    registro.guardar_resumen("b01", "# Resumen 2\n")
    assert registro.resumen("b01") == "# Resumen 2\n"
    assert len(registro.corridas("b01")) == 1  # el resumen no es una corrida


# --- US3: lo que guarda una corrida real (contracts/registro.md) ----------------------------------

ETIQUETAS_CORRIDA = {
    "barrido", "ronda", "ronda.huella", "configuracion", "codigo.commit", "codigo.sucio", "dispositivo",
    "corpus.sha256_ventanas", "corpus.sha256_indice", "particion.huella", "modelo.parametros",
    "modelo.tamano_kb", "estado", "fase", "rotacion",
}
METRICAS_FINALES = {
    "elegida.epoca", "elegida.top1", "elegida.f1_macro", "elegida.acierto_95", "elegida.falsas_95",
    "elegida.perdida", "duracion_s",
}


@pytest.fixture
def corrida_real(raiz_sintetica: Path, lector_seleccion, tmp_path: Path):
    import torch

    from signai_ml.reconocedor.cli import main

    from .test_cli import RONDA_1, declarar

    declarar(raiz_sintetica, 1, RONDA_1)
    registro = RegistroMlflow(tmp_path / "datos")

    def lanzar() -> int:
        return main(["corrida", "b01", "base", "--semilla", "215", "--rotacion", "a", "--permitir-sucio"],
                    raiz=raiz_sintetica, registro=registro, lector_seleccion=lector_seleccion,
                    dev=torch.device("cpu"))

    assert lanzar() == 0
    return registro, lanzar


def test_corrida_terminada_tiene_todo_el_contrato(corrida_real, tmp_path: Path) -> None:
    registro, _ = corrida_real
    (corrida,) = registro.corridas("b01")
    assert ETIQUETAS_CORRIDA <= set(corrida.etiquetas)
    assert {k for k in corrida.etiquetas if k.startswith("version.")}
    assert METRICAS_FINALES <= set(corrida.metricas)
    assert corrida.parametros["semilla"] == "215"
    assert {"forma.capas", "entrenamiento.tasa", "entrenamiento.epocas"} <= set(corrida.parametros)
    assert corrida.etiquetas["codigo.sucio"] == "true" and corrida.etiquetas["dispositivo"] == "cpu"
    assert int(corrida.etiquetas["modelo.parametros"]) > 0
    assert (corrida.fase, corrida.rotacion) == ("seleccion", "a")
    assert [paso for paso, _ in registro.historia(corrida.id, "validacion.top1")] == [1, 2]
    por_clase = json.loads(registro.descargar(corrida.id, "por_clase.json", tmp_path / "b").read_text())
    assert len(por_clase) == 4


def test_ninguna_clave_habla_de_la_prueba(corrida_real) -> None:
    registro, _ = corrida_real
    (corrida,) = registro.corridas("b01")
    claves = [*corrida.parametros, *corrida.etiquetas, *corrida.metricas, "modelo.pt", "por_clase.json"]
    assert not [c for c in claves if "prueba" in c.lower() or "test" in c.lower()]


def test_repetir_crea_una_corrida_nueva(corrida_real) -> None:
    registro, lanzar = corrida_real
    assert lanzar() == 0
    corridas = registro.corridas("b01")
    assert len(corridas) == 2 and len({c.id for c in corridas}) == 2
    assert all(c.estado is EstadoCorrida.TERMINADA for c in corridas)


def test_fase_y_rotacion_de_una_corrida(registro) -> None:
    sin = _abrir(registro)
    con = registro.abrir("b01", "base-s216-b", {"semilla": 216}, {**ETIQUETAS, "fase": "seleccion", "rotacion": "b"})
    final = registro.abrir("b01", "base-s216-final", {"semilla": 216}, {**ETIQUETAS, "fase": "final"})
    por_id = {c.id: c for c in registro.corridas("b01")}
    assert (por_id[sin].fase, por_id[sin].rotacion) == ("seleccion", None)
    assert (por_id[con].fase, por_id[con].rotacion) == ("seleccion", "b")
    assert (por_id[final].fase, por_id[final].rotacion) == ("final", None)
