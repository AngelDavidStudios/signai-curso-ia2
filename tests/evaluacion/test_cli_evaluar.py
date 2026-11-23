"""`signai-aceptacion evaluar` de extremo a extremo (contracts/cli.md; spec US1, FR-001…FR-007, SC-003).

Sobre el corpus y el candidato sintéticos (4 clases, 12 clips de prueba), el lector falso, el
registro en memoria y una comprobación del código inyectada. Donde la decisión importa, se fijan las
probabilidades del modelo para que el resultado no dependa de sus pesos.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from signai_ml.aceptacion import cli
from signai_ml.aceptacion.candidato import leer
from signai_ml.aceptacion.consultas import EstadoConsulta
from signai_ml.aceptacion.registro import RegistroAceptacionEnMemoria, RegistroAceptacionMlflow

from .sinteticos import N_CLASES, PRUEBA_POR_CLASE, TOTAL_PRUEBA, escribir_candidato

COMMIT = "f" * 40


def _limpio(_raiz: Path) -> tuple[str, bool]:
    return COMMIT, False


def _evaluar(raiz: Path, registro, lector, **kwargs) -> int:
    opciones = {"comprobar_codigo": _limpio, "total": TOTAL_PRUEBA, "por_clase": PRUEBA_POR_CLASE} | kwargs
    return cli.main(["evaluar"], raiz=raiz, registro=registro, lector=lector, **opciones)


def _fijar_aciertos(monkeypatch: pytest.MonkeyPatch, aciertos: int) -> None:
    """El modelo acierta los `aciertos` primeros clips de prueba, con confianza 0,97, y falla el resto."""

    def probabilidades(_modelo, x, lote=200):
        etiquetas = np.repeat(np.arange(N_CLASES), PRUEBA_POR_CLASE)
        p = np.full((len(x), N_CLASES), 0.01)
        for fila, etiqueta in enumerate(etiquetas):
            p[fila, etiqueta if fila < aciertos else (etiqueta + 1) % N_CLASES] = 0.97
        return p

    monkeypatch.setattr(cli, "probabilidades", probabilidades)


def _ficha(raiz: Path) -> dict:
    return json.loads((raiz / "modelos" / "reconocedor" / "candidato.json").read_text(encoding="utf-8"))


def test_arbol_sucio_no_consulta(raiz_prueba: Path, lector_prueba_falso, capsys) -> None:
    registro = RegistroAceptacionEnMemoria()

    codigo = _evaluar(raiz_prueba, registro, lector_prueba_falso, comprobar_codigo=None)  # la real: no es un repo

    assert codigo == 3
    assert registro.evaluaciones() == []
    assert "sin confirmar" in capsys.readouterr().err


def test_primera_evaluacion_aceptada(raiz_prueba: Path, lector_prueba_falso, monkeypatch, capsys) -> None:
    _fijar_aciertos(monkeypatch, 11)
    registro = RegistroAceptacionEnMemoria()

    assert _evaluar(raiz_prueba, registro, lector_prueba_falso) == 0

    salida = capsys.readouterr().out
    assert "Consulta 1 a la prueba (1 en total)" in salida
    assert "11 / 12 = 91,67 %" in salida
    assert "umbral ≥85 %: 11" in salida
    assert "Decisión: ACEPTADO" in salida
    assert "uv run dvc add modelos/reconocedor" in salida
    [evaluacion] = registro.evaluaciones()
    assert evaluacion.estado is EstadoConsulta.TERMINADA
    assert evaluacion.decision == "aceptado"
    assert evaluacion.metricas["aciertos"] == 11
    assert evaluacion.metricas["total"] == TOTAL_PRUEBA
    assert evaluacion.modelo_sha256 == leer(raiz_prueba).modelo_sha256
    assert evaluacion.etiquetas["codigo.commit"] == COMMIT
    assert evaluacion.etiquetas["corrida.tt14"] == "c" * 32
    assert evaluacion.etiquetas["particion.huella"] == leer(raiz_prueba).huellas["particion.huella"]
    assert set(registro.artefactos(evaluacion.run_id)) == {
        "metricas_por_clase.json", "confusion.csv", "informe.md", "modelo/modelo.pt",
    }
    ficha = _ficha(raiz_prueba)
    assert ficha["estado"] == "aceptado"
    assert ficha["aceptacion"]["consulta"] == 1
    assert ficha["aceptacion"]["run_id"] == evaluacion.run_id
    assert ficha["aceptacion"]["aciertos"] == 11


def test_repetir_no_vuelve_a_consultar(raiz_prueba: Path, lector_prueba_falso, monkeypatch, capsys) -> None:
    _fijar_aciertos(monkeypatch, 11)
    registro = RegistroAceptacionEnMemoria()
    _evaluar(raiz_prueba, registro, lector_prueba_falso)
    capsys.readouterr()

    def lector_prohibido(_raiz):
        raise AssertionError("no debía leer la prueba otra vez")

    assert _evaluar(raiz_prueba, registro, lector_prohibido) == 0

    salida = capsys.readouterr().out
    assert "ya evaluado" in salida
    assert "Consulta 1 a la prueba (1 en total)" in salida
    assert "Decisión: ACEPTADO" in salida
    assert len(registro.evaluaciones()) == 1


def test_por_debajo_del_umbral_se_rechaza(raiz_prueba: Path, lector_prueba_falso, monkeypatch, capsys) -> None:
    _fijar_aciertos(monkeypatch, 10)  # 10/12 = 83,3 % < 85 %
    registro = RegistroAceptacionEnMemoria()

    assert _evaluar(raiz_prueba, registro, lector_prueba_falso) == 0

    assert "Decisión: RECHAZADO" in capsys.readouterr().out
    assert registro.evaluaciones()[0].decision == "rechazado"
    assert _ficha(raiz_prueba)["estado"] == "rechazado"
    assert _evaluar(raiz_prueba, registro, lector_prueba_falso) == 0  # repetir sigue siendo rechazado
    assert _ficha(raiz_prueba)["estado"] == "rechazado"


def test_huellas_distintas_no_abren_consulta(raiz_prueba: Path, lector_prueba_falso, capsys) -> None:
    candidato = leer(raiz_prueba)
    escribir_candidato(raiz_prueba, huellas=candidato.huellas | {"particion.huella": "0" * 64})
    registro = RegistroAceptacionEnMemoria()

    assert _evaluar(raiz_prueba, registro, lector_prueba_falso) == 1

    assert registro.evaluaciones() == []
    assert "particion.huella" in capsys.readouterr().err


def test_una_interrupcion_reutiliza_la_consulta(raiz_prueba: Path, lector_prueba_falso, monkeypatch, capsys) -> None:
    def cortar(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "probabilidades", cortar)
    registro = RegistroAceptacionEnMemoria()

    assert _evaluar(raiz_prueba, registro, lector_prueba_falso) == 1
    assert registro.evaluaciones()[0].estado is EstadoConsulta.INTERRUMPIDA
    assert "interrumpida" in capsys.readouterr().err

    _fijar_aciertos(monkeypatch, 12)
    assert _evaluar(raiz_prueba, registro, lector_prueba_falso) == 0

    assert "Consulta 1 a la prueba (1 en total)" in capsys.readouterr().out
    assert sorted(e.consulta for e in registro.evaluaciones()) == [1, 1]


def test_un_fallo_queda_registrado(raiz_prueba: Path, lector_prueba_falso, monkeypatch) -> None:
    def romper(*_args, **_kwargs):
        raise RuntimeError("se rompió")

    monkeypatch.setattr(cli, "probabilidades", romper)
    registro = RegistroAceptacionEnMemoria()

    with pytest.raises(RuntimeError):
        _evaluar(raiz_prueba, registro, lector_prueba_falso)

    [evaluacion] = registro.evaluaciones()
    assert evaluacion.estado is EstadoConsulta.FALLIDA
    assert evaluacion.etiquetas["estado.motivo"] == "se rompió"
    assert _ficha(raiz_prueba)["estado"] == "pendiente de aceptación (TT-15)"


def test_otro_candidato_es_la_consulta_2(raiz_prueba: Path, lector_prueba_falso, monkeypatch, capsys) -> None:
    _fijar_aciertos(monkeypatch, 9)
    registro = RegistroAceptacionEnMemoria()
    _evaluar(raiz_prueba, registro, lector_prueba_falso)
    escribir_candidato(raiz_prueba, semilla=1)  # TT-14 entrega otro candidato tras el rechazo
    capsys.readouterr()

    _fijar_aciertos(monkeypatch, 12)
    assert _evaluar(raiz_prueba, registro, lector_prueba_falso) == 0

    assert "Consulta 2 a la prueba (2 en total)" in capsys.readouterr().out


def test_los_artefactos_de_la_consulta(raiz_prueba: Path, lector_prueba_falso, monkeypatch, tmp_path) -> None:
    _fijar_aciertos(monkeypatch, 11)
    registro = RegistroAceptacionMlflow()

    assert _evaluar(raiz_prueba, registro, lector_prueba_falso) == 0

    [evaluacion] = registro.evaluaciones()
    from mlflow import MlflowClient

    from signai_ml.seguimiento import uri_seguimiento

    cliente = MlflowClient(tracking_uri=uri_seguimiento())
    destino = Path(cliente.download_artifacts(evaluacion.run_id, "", str(tmp_path)))
    por_clase = json.loads((destino / "metricas_por_clase.json").read_text(encoding="utf-8"))
    assert [c["clase"] for c in por_clase] == list(leer(raiz_prueba).clases)
    assert set(por_clase[0]) == {"clase", "glosa", "top1", "f1"}
    confusion = (destino / "confusion.csv").read_text(encoding="utf-8").splitlines()
    assert len(confusion) == N_CLASES + 1
    assert confusion[0].split(",")[1:] == list(leer(raiz_prueba).clases)
    assert sum(int(v) for fila in confusion[1:] for v in fila.split(",")[1:]) == TOTAL_PRUEBA
    informe = (destino / "informe.md").read_text(encoding="utf-8")
    assert "Consulta 1 a la prueba (1 en total)" in informe
    assert "ADR-0011" in informe
    assert (destino / "modelo" / "modelo.pt").read_bytes() == (
        raiz_prueba / "modelos" / "reconocedor" / "modelo.pt"
    ).read_bytes()


def test_con_el_modelo_real_sin_fijar_probabilidades(raiz_prueba: Path, lector_prueba_falso, capsys) -> None:
    registro = RegistroAceptacionEnMemoria()

    assert _evaluar(raiz_prueba, registro, lector_prueba_falso) == 0

    [evaluacion] = registro.evaluaciones()
    assert 0 <= evaluacion.metricas["aciertos"] <= TOTAL_PRUEBA
    assert evaluacion.decision in ("aceptado", "rechazado")
