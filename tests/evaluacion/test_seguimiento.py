"""Registro de corridas en MLflow (FR-003; contrato entorno-local.md; data-model «Corrida»)."""

import math
from pathlib import Path

import pytest
from mlflow import MlflowClient

from signai_ml.seguimiento import (
    abrir_corrida,
    cerrar_corrida,
    registrar_corrida,
    registrar_epoca,
    uri_seguimiento,
    versiones_bibliotecas,
)

BIBLIOTECAS = {"python", "torch", "coremltools", "numpy", "mlflow", "dvc"}


def _corrida_valida(raiz: Path, **cambios):
    argumentos = {
        "experimento": "prueba",
        "configuracion": {"lr": 0.001, "capas": 4, "optimizador": "adamw"},
        "semilla": 42,
        "metricas": {"top1": 0.87, "perdida": 0.41},
        "raiz": raiz,
    }
    argumentos.update(cambios)
    return registrar_corrida(**argumentos)


class TestUriSeguimiento:
    def test_con_raiz_explicita(self, tmp_path: Path) -> None:
        assert uri_seguimiento(tmp_path) == f"sqlite:///{tmp_path}/mlflow/mlflow.db"

    def test_sin_raiz_usa_signai_data_root(self, raiz_datos: Path) -> None:
        assert uri_seguimiento() == f"sqlite:///{raiz_datos}/mlflow/mlflow.db"

    def test_sin_variable_usa_signai_data_en_el_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIGNAI_DATA_ROOT")
        esperado = Path.home() / "signai-data" / "mlflow" / "mlflow.db"
        assert uri_seguimiento() == f"sqlite:///{esperado}"


class TestVersionesBibliotecas:
    def test_devuelve_exactamente_las_bibliotecas_del_contrato(self) -> None:
        versiones = versiones_bibliotecas()
        assert set(versiones) == BIBLIOTECAS
        assert all(isinstance(v, str) and v for v in versiones.values())


class TestRegistrarCorrida:
    def test_registra_configuracion_semilla_metricas_y_versiones(self, tmp_path: Path) -> None:
        id_corrida = _corrida_valida(tmp_path)

        corrida = MlflowClient(tracking_uri=uri_seguimiento(tmp_path)).get_run(id_corrida)
        assert corrida.data.params == {
            "lr": "0.001",
            "capas": "4",
            "optimizador": "adamw",
            "semilla": "42",
        }
        assert corrida.data.metrics == {"top1": 0.87, "perdida": 0.41}
        etiquetas = {k: v for k, v in corrida.data.tags.items() if k.startswith("version.")}
        assert set(etiquetas) == {f"version.{b}" for b in BIBLIOTECAS}
        assert corrida.info.status == "FINISHED"

    def test_crea_el_almacen_bajo_la_raiz(self, tmp_path: Path) -> None:
        _corrida_valida(tmp_path)
        assert (tmp_path / "mlflow" / "mlflow.db").is_file()
        assert (tmp_path / "mlflow" / "artifacts").is_dir()

    def test_sin_raiz_escribe_en_signai_data_root(self, raiz_datos: Path) -> None:
        _corrida_valida(None)
        assert (raiz_datos / "mlflow" / "mlflow.db").is_file()

    def test_dos_corridas_del_mismo_experimento(self, tmp_path: Path) -> None:
        assert _corrida_valida(tmp_path) != _corrida_valida(tmp_path)

    @pytest.mark.parametrize("experimento", ["", "   "])
    def test_rechaza_experimento_vacio(self, tmp_path: Path, experimento: str) -> None:
        with pytest.raises(ValueError, match="experimento"):
            _corrida_valida(tmp_path, experimento=experimento)

    def test_rechaza_semilla_negativa(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="semilla"):
            _corrida_valida(tmp_path, semilla=-1)

    @pytest.mark.parametrize("semilla", [1.5, "42", True])
    def test_rechaza_semilla_que_no_es_entero(self, tmp_path: Path, semilla: object) -> None:
        with pytest.raises(ValueError, match="semilla"):
            _corrida_valida(tmp_path, semilla=semilla)

    def test_rechaza_sin_metricas(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="métricas"):
            _corrida_valida(tmp_path, metricas={})

    @pytest.mark.parametrize("valor", [math.nan, math.inf, -math.inf])
    def test_rechaza_metricas_no_finitas(self, tmp_path: Path, valor: float) -> None:
        with pytest.raises(ValueError, match="top1"):
            _corrida_valida(tmp_path, metricas={"top1": valor})

    def test_no_registra_nada_si_los_datos_son_invalidos(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _corrida_valida(tmp_path, semilla=-1)
        assert not (tmp_path / "mlflow").exists()


class TestRegistroIncremental:
    """API incremental para TT-14 (specs/006-entrenamiento-reconocedor, research R-10)."""

    def _abrir(self, raiz: Path) -> str:
        return abrir_corrida(
            "reconocedor/b01", "base-s215", {"forma.capas": 4, "semilla": 215}, {"estado": "en_curso"}, raiz=raiz
        )

    def test_abrir_crea_el_run_en_curso_con_parametros_etiquetas_y_versiones(self, tmp_path: Path) -> None:
        corrida = MlflowClient(tracking_uri=uri_seguimiento(tmp_path)).get_run(self._abrir(tmp_path))
        assert corrida.info.status == "RUNNING"
        assert corrida.info.run_name == "base-s215"
        assert corrida.data.params == {"forma.capas": "4", "semilla": "215"}
        assert corrida.data.tags["estado"] == "en_curso"
        assert {k for k in corrida.data.tags if k.startswith("version.")} == {f"version.{b}" for b in BIBLIOTECAS}

    def test_registrar_epoca_usa_la_epoca_como_paso(self, tmp_path: Path) -> None:
        id_corrida = self._abrir(tmp_path)
        registrar_epoca(id_corrida, 1, {"validacion.top1": 0.5}, raiz=tmp_path)
        registrar_epoca(id_corrida, 2, {"validacion.top1": 0.7}, raiz=tmp_path)
        historia = MlflowClient(tracking_uri=uri_seguimiento(tmp_path)).get_metric_history(id_corrida, "validacion.top1")
        assert [(m.step, m.value) for m in historia] == [(1, 0.5), (2, 0.7)]

    def test_registrar_epoca_rechaza_no_finitas(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="validacion.perdida"):
            registrar_epoca(self._abrir(tmp_path), 1, {"validacion.perdida": math.nan}, raiz=tmp_path)

    @pytest.mark.parametrize("estado", ["FINISHED", "FAILED", "KILLED"])
    def test_cerrar_con_metricas_artefactos_y_etiquetas(self, tmp_path: Path, estado: str) -> None:
        id_corrida = self._abrir(tmp_path)
        artefacto = tmp_path / "por_clase.json"
        artefacto.write_text("{}", encoding="utf-8")
        cerrar_corrida(
            id_corrida, estado, metricas={"elegida.top1": 0.9}, artefactos=[artefacto],
            etiquetas={"estado": "terminada"}, raiz=tmp_path,
        )
        cliente = MlflowClient(tracking_uri=uri_seguimiento(tmp_path))
        corrida = cliente.get_run(id_corrida)
        assert corrida.info.status == estado
        assert corrida.data.metrics["elegida.top1"] == 0.9
        assert corrida.data.tags["estado"] == "terminada"
        assert [a.path for a in cliente.list_artifacts(id_corrida)] == ["por_clase.json"]

    def test_cerrar_rechaza_un_estado_desconocido(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="estado"):
            cerrar_corrida(self._abrir(tmp_path), "TERMINADA", raiz=tmp_path)
