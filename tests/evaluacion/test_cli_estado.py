"""`signai-aceptacion estado`: solo lectura y nunca consulta la prueba (contracts/cli.md; SC-008)."""

import json
from pathlib import Path

import pytest

from signai_ml.aceptacion import cli
from signai_ml.aceptacion.estado import estado_dvc
from signai_ml.aceptacion.registro import RegistroAceptacionEnMemoria

from .sinteticos import PRUEBA_POR_CLASE, TOTAL_PRUEBA, lector_prueba
from .test_cli_evaluar import _fijar_aciertos


def _prohibido(_raiz):
    raise AssertionError("estado no debe leer la prueba")


def _dvc_al_dia(_raiz: Path, objetivos) -> dict[str, str]:
    return {objetivo: "al día" for objetivo in objetivos}


def _estado(raiz: Path, registro, **kwargs) -> int:
    opciones = {"lector": _prohibido, "estado_dvc": _dvc_al_dia} | kwargs
    return cli.main(["estado"], raiz=raiz, registro=registro, **opciones)


def _evaluar_y_registrar(raiz: Path, registro, monkeypatch, aciertos: int = 11) -> None:
    _fijar_aciertos(monkeypatch, aciertos)
    opciones = {"lector": lector_prueba, "comprobar_codigo": lambda _: ("f" * 40, False),
                "total": TOTAL_PRUEBA, "por_clase": PRUEBA_POR_CLASE}
    assert cli.main(["evaluar"], raiz=raiz, registro=registro, **opciones) == 0
    if aciertos >= 11:
        assert cli.main(["registrar"], raiz=raiz, registro=registro, **opciones) == 0


def test_antes_de_evaluar(raiz_prueba: Path, capsys) -> None:
    assert _estado(raiz_prueba, RegistroAceptacionEnMemoria()) == 0

    salida = capsys.readouterr().out
    assert "Candidato: b01 / r01-base / semilla 216" in salida
    assert "pendiente de aceptación" in salida
    assert "Consultas a la prueba: 0" in salida
    assert "Versión con alias aceptado: ninguna" in salida
    assert "Última exportación: ninguna" in salida


def test_tras_evaluar_y_registrar(raiz_prueba: Path, monkeypatch, capsys) -> None:
    registro = RegistroAceptacionEnMemoria()
    _evaluar_y_registrar(raiz_prueba, registro, monkeypatch)
    capsys.readouterr()

    assert _estado(raiz_prueba, registro) == 0

    salida = capsys.readouterr().out
    assert "estado aceptado" in salida
    assert "Consultas a la prueba: 1 en total" in salida
    assert "consulta-01 · terminada · aceptado · 11 / 12" in salida
    assert "Versión con alias aceptado: 1 de reconocedor-kposeformer · consulta 1 · paridad sin demostrar" in salida
    assert "DVC: modelos/reconocedor al día · modelos/coreml al día" in salida


def test_un_rechazado(raiz_prueba: Path, monkeypatch, capsys) -> None:
    registro = RegistroAceptacionEnMemoria()
    _evaluar_y_registrar(raiz_prueba, registro, monkeypatch, aciertos=9)
    capsys.readouterr()

    _estado(raiz_prueba, registro)

    salida = capsys.readouterr().out
    assert "estado rechazado" in salida
    assert "consulta-01 · terminada · rechazado · 9 / 12" in salida


def test_la_ultima_exportacion(raiz_prueba: Path, monkeypatch, capsys) -> None:
    registro = RegistroAceptacionEnMemoria()
    for paridad in ("fallida", "superada"):
        registro.terminar_exportacion(registro.abrir_exportacion({"modelo.version": "1"}), {}, [], paridad)

    _estado(raiz_prueba, registro)

    assert "Última exportación: exportacion-02 · paridad superada" in capsys.readouterr().out


def test_un_candidato_que_no_se_puede_leer(raiz_prueba: Path, capsys) -> None:
    (raiz_prueba / "modelos" / "reconocedor" / "modelo.pt").unlink()

    assert _estado(raiz_prueba, RegistroAceptacionEnMemoria()) == 0

    assert "Candidato: no válido" in capsys.readouterr().out


def test_dvc_sin_versionar_o_con_cambios(raiz_prueba: Path, capsys) -> None:
    def dvc(_raiz, objetivos):
        return {"modelos/reconocedor": "con cambios", "modelos/coreml": "sin versionar"}

    _estado(raiz_prueba, RegistroAceptacionEnMemoria(), estado_dvc=dvc)

    assert "DVC: modelos/reconocedor con cambios · modelos/coreml sin versionar" in capsys.readouterr().out


def test_estado_dvc_real_sin_punteros(tmp_path: Path) -> None:
    assert estado_dvc(tmp_path, ["modelos/reconocedor", "modelos/coreml"]) == {
        "modelos/reconocedor": "sin versionar", "modelos/coreml": "sin versionar",
    }


def test_estado_dvc_interpreta_la_salida(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for objetivo in ("modelos/reconocedor", "modelos/coreml"):
        (tmp_path / f"{objetivo}.dvc").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f"{objetivo}.dvc").write_text("outs: []\n", encoding="utf-8")

    class Resultado:
        returncode = 0
        stdout = json.dumps({"modelos/coreml.dvc": [{"changed outs": {"coreml": "modified"}}]})

    monkeypatch.setattr("signai_ml.aceptacion.estado.subprocess.run", lambda *a, **k: Resultado())

    assert estado_dvc(tmp_path, ["modelos/reconocedor", "modelos/coreml"]) == {
        "modelos/reconocedor": "al día", "modelos/coreml": "con cambios",
    }


def test_estado_dvc_que_falla(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "modelos").mkdir()
    (tmp_path / "modelos" / "coreml.dvc").write_text("outs: []\n", encoding="utf-8")

    class Resultado:
        returncode = 1
        stdout = ""

    monkeypatch.setattr("signai_ml.aceptacion.estado.subprocess.run", lambda *a, **k: Resultado())

    assert estado_dvc(tmp_path, ["modelos/coreml"]) == {"modelos/coreml": "desconocido"}
