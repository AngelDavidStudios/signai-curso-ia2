"""Lectura del candidato de TT-14 con su huella (spec FR-002, FR-009; data-model «Candidato»)."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from signai_ml.aceptacion.candidato import Candidato, anotar_evaluacion, leer
from signai_ml.aceptacion.errores import CandidatoNoValido
from signai_ml.reconocedor.modelo import KPoseFormer

from .sinteticos import HUELLA_PARTICION, N_CLASES, escribir_candidato


def _modelo_pt(raiz: Path) -> Path:
    return raiz / "modelos" / "reconocedor" / "modelo.pt"


def test_lee_el_candidato_con_su_huella(raiz_prueba: Path) -> None:
    candidato = leer(raiz_prueba)

    assert isinstance(candidato, Candidato)
    assert isinstance(candidato.modelo, KPoseFormer)
    assert not candidato.modelo.training
    assert candidato.modelo_sha256 == hashlib.sha256(_modelo_pt(raiz_prueba).read_bytes()).hexdigest()
    assert candidato.clases == tuple(f"{n}-Sint{n}" for n in range(1, N_CLASES + 1))
    assert candidato.glosas == tuple(f"glosa{n}" for n in range(1, N_CLASES + 1))
    assert candidato.huellas["particion.huella"] == HUELLA_PARTICION
    assert set(candidato.huellas) == {"corpus.sha256_ventanas", "corpus.sha256_indice", "particion.huella"}
    assert candidato.ficha["barrido"] == "b01"
    assert candidato.ficha["estado"] == "pendiente de aceptación (TT-15)"


def test_leer_no_modifica_el_modelo(raiz_prueba: Path) -> None:
    antes = _modelo_pt(raiz_prueba).read_bytes()

    leer(raiz_prueba)

    assert _modelo_pt(raiz_prueba).read_bytes() == antes


@pytest.mark.parametrize("fichero", ["modelo.pt", "candidato.json"])
def test_falta_un_fichero(raiz_prueba: Path, fichero: str) -> None:
    (raiz_prueba / "modelos" / "reconocedor" / fichero).unlink()

    with pytest.raises(CandidatoNoValido, match=fichero):
        leer(raiz_prueba)


def test_formato_desconocido(raiz_prueba: Path) -> None:
    datos = torch.load(_modelo_pt(raiz_prueba), weights_only=True)
    datos["formato"] = "otro/1"
    torch.save(datos, _modelo_pt(raiz_prueba))

    with pytest.raises(CandidatoNoValido, match="formato"):
        leer(raiz_prueba)


def test_clases_distintas_del_vocabulario(raiz_prueba: Path) -> None:
    vocabulario = raiz_prueba / "config" / "corpus" / "vocabulario.csv"
    vocabulario.write_text(vocabulario.read_text().replace("glosa2", "otra"), encoding="utf-8")

    with pytest.raises(CandidatoNoValido, match="vocabulario"):
        leer(raiz_prueba)


def test_demasiados_parametros(raiz_prueba: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("signai_ml.aceptacion.candidato.LIMITE_PARAMETROS", 10)

    with pytest.raises(CandidatoNoValido, match="parámetros"):
        leer(raiz_prueba)


def test_ficha_sin_huellas(raiz_prueba: Path) -> None:
    escribir_candidato(raiz_prueba, huellas={"corpus.sha256_ventanas": "a" * 64})

    with pytest.raises(CandidatoNoValido, match="huellas"):
        leer(raiz_prueba)


def test_ficha_que_no_es_json(raiz_prueba: Path) -> None:
    (raiz_prueba / "modelos" / "reconocedor" / "candidato.json").write_text("{", encoding="utf-8")

    with pytest.raises(CandidatoNoValido, match="candidato.json"):
        leer(raiz_prueba)


def test_ficha_de_otra_corrida(raiz_prueba: Path) -> None:
    ficha = raiz_prueba / "modelos" / "reconocedor" / "candidato.json"
    datos = json.loads(ficha.read_text())
    datos["run_id"] = "z" * 32
    ficha.write_text(json.dumps(datos), encoding="utf-8")

    with pytest.raises(CandidatoNoValido, match="corrida"):
        leer(raiz_prueba)


# ── anotar_evaluacion (FR-007, FR-010; research R-09) ───────────────────────────────────────────


def _anotar(raiz: Path, decision: str = "aceptado", **cambios) -> dict:
    argumentos = {"consulta": 1, "run_id": "r" * 32, "aciertos": 11, "total": 12, "top1": 11 / 12, "decision": decision}
    anotar_evaluacion(raiz, **(argumentos | cambios))
    return json.loads((raiz / "modelos" / "reconocedor" / "candidato.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("decision", ["aceptado", "rechazado"])
def test_anotar_la_evaluacion(raiz_prueba: Path, decision: str) -> None:
    antes = json.loads((raiz_prueba / "modelos" / "reconocedor" / "candidato.json").read_text(encoding="utf-8"))

    ficha = _anotar(raiz_prueba, decision)

    assert ficha["estado"] == decision
    assert ficha["aceptacion"] == {
        "consulta": 1, "run_id": "r" * 32, "aciertos": 11, "total": 12, "top1": pytest.approx(11 / 12),
        "decision": decision,
    }
    assert {k: v for k, v in ficha.items() if k not in ("estado", "aceptacion")} == {
        k: v for k, v in antes.items() if k != "estado"
    }


def test_anotar_no_toca_el_modelo(raiz_prueba: Path) -> None:
    huella = hashlib.sha256(_modelo_pt(raiz_prueba).read_bytes()).hexdigest()

    _anotar(raiz_prueba)

    assert hashlib.sha256(_modelo_pt(raiz_prueba).read_bytes()).hexdigest() == huella


def test_anotar_de_nuevo_conserva_el_registro(raiz_prueba: Path) -> None:
    _anotar(raiz_prueba)
    ficha = raiz_prueba / "modelos" / "reconocedor" / "candidato.json"
    datos = json.loads(ficha.read_text(encoding="utf-8"))
    datos["aceptacion"]["modelo_registrado"] = {"nombre": "reconocedor-kposeformer", "version": 1}
    ficha.write_text(json.dumps(datos), encoding="utf-8")

    assert _anotar(raiz_prueba)["aceptacion"]["modelo_registrado"] == {"nombre": "reconocedor-kposeformer", "version": 1}


def test_rechazado_es_terminal(raiz_prueba: Path) -> None:
    _anotar(raiz_prueba, "rechazado")

    with pytest.raises(CandidatoNoValido, match="rechazado"):
        _anotar(raiz_prueba, "aceptado", consulta=2)


def test_decision_desconocida(raiz_prueba: Path) -> None:
    with pytest.raises(ValueError):
        _anotar(raiz_prueba, "quizas")
