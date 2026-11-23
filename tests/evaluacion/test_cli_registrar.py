"""`signai-aceptacion registrar` (contracts/cli.md; spec US2, FR-007…FR-010, SC-002, SC-004)."""

import json
from pathlib import Path

import pytest

from signai_ml.aceptacion import MODELO_REGISTRADO
from signai_ml.aceptacion.candidato import leer, sha256_fichero
from signai_ml.aceptacion.registro import RegistroAceptacionEnMemoria, RegistroAceptacionMlflow

from .test_cli_evaluar import _evaluar, _ficha, _fijar_aciertos


def _registrar(raiz: Path, registro) -> int:
    from signai_ml.aceptacion import cli

    return cli.main(["registrar"], raiz=raiz, registro=registro)


@pytest.fixture(params=["memoria", "mlflow"])
def registro(request):
    return RegistroAceptacionEnMemoria() if request.param == "memoria" else RegistroAceptacionMlflow()


def test_registra_el_aceptado(raiz_prueba: Path, lector_prueba_falso, registro, monkeypatch, capsys) -> None:
    _fijar_aciertos(monkeypatch, 11)
    _evaluar(raiz_prueba, registro, lector_prueba_falso)
    capsys.readouterr()

    assert _registrar(raiz_prueba, registro) == 0

    salida = capsys.readouterr().out
    assert f"Versión 1 de {MODELO_REGISTRADO}" in salida
    assert "alias aceptado" in salida
    version = registro.version_con_alias()
    candidato = leer(raiz_prueba)
    assert version.version == 1
    assert version.etiquetas["modelo.sha256"] == candidato.modelo_sha256
    assert version.etiquetas["consulta"] == "1"
    assert version.etiquetas["aciertos"] == "11"
    assert version.etiquetas["corrida.tt14"] == "c" * 32
    assert version.etiquetas["particion.huella"] == candidato.huellas["particion.huella"]
    assert _ficha(raiz_prueba)["aceptacion"]["modelo_registrado"] == {"nombre": MODELO_REGISTRADO, "version": 1}


def test_los_pesos_registrados_son_los_evaluados(
    raiz_prueba: Path, lector_prueba_falso, registro, monkeypatch, tmp_path: Path
) -> None:
    _fijar_aciertos(monkeypatch, 12)
    _evaluar(raiz_prueba, registro, lector_prueba_falso)
    _registrar(raiz_prueba, registro)

    descargado = registro.descargar_modelo(registro.version_con_alias().version, tmp_path / "descarga")

    assert sha256_fichero(descargado) == leer(raiz_prueba).modelo_sha256  # SC-004


def test_repetir_no_crea_otra_version(raiz_prueba: Path, lector_prueba_falso, registro, monkeypatch, capsys) -> None:
    _fijar_aciertos(monkeypatch, 11)
    _evaluar(raiz_prueba, registro, lector_prueba_falso)
    _registrar(raiz_prueba, registro)
    capsys.readouterr()

    assert _registrar(raiz_prueba, registro) == 0

    assert "ya está registrado" in capsys.readouterr().out
    assert registro.version_con_alias().version == 1


def test_un_rechazado_no_se_registra(raiz_prueba: Path, lector_prueba_falso, registro, monkeypatch, capsys) -> None:
    _fijar_aciertos(monkeypatch, 10)
    _evaluar(raiz_prueba, registro, lector_prueba_falso)
    capsys.readouterr()

    assert _registrar(raiz_prueba, registro) == 1

    assert "no alcanzó" in capsys.readouterr().err
    assert registro.version_con_alias() is None
    assert "modelo_registrado" not in _ficha(raiz_prueba)["aceptacion"]


def test_sin_evaluar_no_se_registra(raiz_prueba: Path, registro, capsys) -> None:
    assert _registrar(raiz_prueba, registro) == 1

    assert "evaluar" in capsys.readouterr().err
    assert registro.version_con_alias() is None


def test_la_ficha_se_completa_aunque_faltara_la_anotacion(
    raiz_prueba: Path, lector_prueba_falso, registro, monkeypatch
) -> None:
    _fijar_aciertos(monkeypatch, 11)
    _evaluar(raiz_prueba, registro, lector_prueba_falso)
    ruta = raiz_prueba / "modelos" / "reconocedor" / "candidato.json"
    ficha = json.loads(ruta.read_text(encoding="utf-8"))
    ficha.pop("aceptacion")
    ficha["estado"] = "pendiente de aceptación (TT-15)"
    ruta.write_text(json.dumps(ficha), encoding="utf-8")

    assert _registrar(raiz_prueba, registro) == 0

    ficha = _ficha(raiz_prueba)
    assert ficha["estado"] == "aceptado"
    assert ficha["aceptacion"]["modelo_registrado"]["version"] == 1
