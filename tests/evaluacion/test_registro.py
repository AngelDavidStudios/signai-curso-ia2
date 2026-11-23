"""Registro de las consultas en MLflow, parte G-1 (spec FR-006; contracts/registro.md).

La misma batería pasa por el adaptador real (SQLite temporal bajo `SIGNAI_DATA_ROOT`) y por el de
memoria que usan las pruebas de la CLI.
"""

from pathlib import Path

import pytest
from mlflow import MlflowClient

from signai_ml.aceptacion import ALIAS, EXPERIMENTO_ACEPTACION, EXPERIMENTO_EXPORTACION, MODELO_REGISTRADO
from signai_ml.aceptacion.consultas import EstadoConsulta
from signai_ml.aceptacion.registro import RegistroAceptacionEnMemoria, RegistroAceptacionMlflow
from signai_ml.seguimiento import uri_seguimiento

HUELLA = "a" * 64
METRICAS = {"aciertos": 10.0, "total": 12.0, "top1": 10 / 12, "f1_macro": 0.8, "acierto_95": 0.5, "falsas_95": 0.0}


@pytest.fixture(params=["mlflow", "memoria"])
def registro(request):
    return RegistroAceptacionMlflow() if request.param == "mlflow" else RegistroAceptacionEnMemoria()


@pytest.fixture
def artefactos(tmp_path: Path) -> tuple[list[Path], Path]:
    nombres = ["metricas_por_clase.json", "confusion.csv", "informe.md"]
    rutas = []
    for nombre in nombres:
        ruta = tmp_path / nombre
        ruta.write_text("x", encoding="utf-8")
        rutas.append(ruta)
    modelo = tmp_path / "modelo.pt"
    modelo.write_bytes(b"pesos")
    return rutas, modelo


def _etiquetas(**cambios) -> dict:
    return {"modelo.sha256": HUELLA, "corrida.tt14": "c" * 32, "particion.huella": "e" * 64} | cambios


def test_sin_consultas(registro) -> None:
    assert registro.evaluaciones() == []


def test_abrir_exige_la_huella_del_modelo(registro) -> None:
    with pytest.raises(ValueError, match="modelo.sha256"):
        registro.abrir_consulta(1, {"corrida.tt14": "c" * 32})


def test_una_consulta_terminada(registro, artefactos) -> None:
    rutas, modelo = artefactos

    id_consulta = registro.abrir_consulta(1, _etiquetas())
    registro.terminar_consulta(id_consulta, METRICAS, "aceptado", rutas, modelo)

    [evaluacion] = registro.evaluaciones()
    assert evaluacion.run_id == id_consulta
    assert evaluacion.consulta == 1
    assert evaluacion.modelo_sha256 == HUELLA
    assert evaluacion.estado is EstadoConsulta.TERMINADA
    assert evaluacion.decision == "aceptado"
    assert evaluacion.metricas == pytest.approx(METRICAS)
    assert evaluacion.etiquetas["corrida.tt14"] == "c" * 32
    assert set(registro.artefactos(id_consulta)) == {
        "metricas_por_clase.json", "confusion.csv", "informe.md", "modelo/modelo.pt",
    }


@pytest.mark.parametrize(
    ("cerrar", "estado"), [("fallar_consulta", EstadoConsulta.FALLIDA), ("interrumpir_consulta", EstadoConsulta.INTERRUMPIDA)]
)
def test_una_consulta_no_terminada(registro, cerrar: str, estado: EstadoConsulta) -> None:
    id_consulta = registro.abrir_consulta(1, _etiquetas())

    getattr(registro, cerrar)(id_consulta, "motivo de prueba")

    [evaluacion] = registro.evaluaciones()
    assert evaluacion.estado is estado
    assert evaluacion.decision is None
    assert evaluacion.etiquetas["estado.motivo"] == "motivo de prueba"


def test_una_consulta_abierta_esta_en_curso(registro) -> None:
    registro.abrir_consulta(1, _etiquetas())

    assert registro.evaluaciones()[0].estado is EstadoConsulta.EN_CURSO


def test_las_consultas_no_se_borran(registro, artefactos) -> None:
    rutas, modelo = artefactos
    primera = registro.abrir_consulta(1, _etiquetas())
    registro.interrumpir_consulta(primera, "Ctrl-C")
    segunda = registro.abrir_consulta(1, _etiquetas())
    registro.terminar_consulta(segunda, METRICAS, "rechazado", rutas, modelo)

    evaluaciones = {e.run_id: e for e in registro.evaluaciones()}

    assert set(evaluaciones) == {primera, segunda}
    assert evaluaciones[segunda].decision == "rechazado"


def test_mlflow_nombra_y_etiqueta_la_consulta(artefactos) -> None:
    rutas, modelo = artefactos
    registro = RegistroAceptacionMlflow()

    id_consulta = registro.abrir_consulta(3, _etiquetas())
    registro.terminar_consulta(id_consulta, METRICAS, "aceptado", rutas, modelo)

    run = MlflowClient(tracking_uri=uri_seguimiento()).get_run(id_consulta)
    assert run.info.run_name == "consulta-03"
    assert run.info.status == "FINISHED"
    assert run.data.tags["tipo"] == "evaluacion-aceptacion"
    assert run.data.tags["consulta"] == "3"
    assert run.data.tags["decision"] == "aceptado"
    assert "version.torch" in run.data.tags
    experimento = MlflowClient(tracking_uri=uri_seguimiento()).get_experiment(run.info.experiment_id)
    assert experimento.name == EXPERIMENTO_ACEPTACION


# ── G-2 · modelo registrado `reconocedor-kposeformer` (FR-008, FR-009; research R-10) ───────────


def _consulta_terminada(registro, artefactos, decision: str = "aceptado", huella: str = HUELLA) -> str:
    rutas, modelo = artefactos
    id_consulta = registro.abrir_consulta(1, _etiquetas(**{"modelo.sha256": huella}))
    registro.terminar_consulta(id_consulta, METRICAS, decision, rutas, modelo)
    return id_consulta


def _etiquetas_version(**cambios) -> dict:
    return {"estado": "aceptado", "consulta": "1", "aciertos": "10", "top1_prueba": "0.8333", "modelo.sha256": HUELLA} | cambios


def test_sin_versiones(registro) -> None:
    assert registro.version_con_alias() is None
    assert registro.version_de(HUELLA) is None


def test_registrar_una_version(registro, artefactos) -> None:
    id_consulta = _consulta_terminada(registro, artefactos)

    version = registro.registrar_version(id_consulta, _etiquetas_version())

    assert version.version == 1
    assert version.run_id == id_consulta
    assert version.etiquetas["estado"] == "aceptado"
    assert version.etiquetas["modelo.sha256"] == HUELLA
    assert registro.version_con_alias() == version
    assert registro.version_de(HUELLA) == version


def test_una_version_nueva_mueve_el_alias_sin_borrar_la_anterior(registro, artefactos) -> None:
    primera = registro.registrar_version(_consulta_terminada(registro, artefactos), _etiquetas_version())
    otra = "b" * 64
    segunda = registro.registrar_version(
        _consulta_terminada(registro, artefactos, huella=otra), _etiquetas_version(**{"modelo.sha256": otra})
    )

    assert segunda.version == 2
    assert registro.version_con_alias().version == 2
    assert registro.version_de(HUELLA) == primera


def test_no_se_registra_una_consulta_rechazada(registro, artefactos) -> None:
    id_consulta = _consulta_terminada(registro, artefactos, decision="rechazado")

    with pytest.raises(ValueError, match="aceptado"):
        registro.registrar_version(id_consulta, _etiquetas_version())
    assert registro.version_con_alias() is None


def test_no_se_registra_una_consulta_sin_terminar(registro) -> None:
    id_consulta = registro.abrir_consulta(1, _etiquetas())

    with pytest.raises(ValueError, match="aceptado"):
        registro.registrar_version(id_consulta, _etiquetas_version())


def test_descargar_el_modelo_de_una_version(registro, artefactos, tmp_path: Path) -> None:
    _, modelo = artefactos
    version = registro.registrar_version(_consulta_terminada(registro, artefactos), _etiquetas_version())

    descargado = registro.descargar_modelo(version.version, tmp_path / "destino")

    assert descargado.name == "modelo.pt"
    assert descargado.read_bytes() == modelo.read_bytes()


def test_etiquetar_una_version(registro, artefactos) -> None:
    version = registro.registrar_version(_consulta_terminada(registro, artefactos), _etiquetas_version())

    registro.etiquetar_version(version.version, {"paridad": "superada", "coreml.sha256": "c" * 64})

    assert registro.version_con_alias().etiquetas["paridad"] == "superada"
    assert registro.version_con_alias().etiquetas["estado"] == "aceptado"


def test_mlflow_registra_con_alias_y_origen(artefactos) -> None:
    registro = RegistroAceptacionMlflow()
    id_consulta = _consulta_terminada(registro, artefactos)

    registro.registrar_version(id_consulta, _etiquetas_version())

    cliente = MlflowClient(tracking_uri=uri_seguimiento())
    version = cliente.get_model_version_by_alias(MODELO_REGISTRADO, ALIAS)
    assert version.source == f"runs:/{id_consulta}/modelo"
    assert version.run_id == id_consulta


# ── G-3 · exportaciones (FR-013; contracts/registro.md) ─────────────────────────────────────────


@pytest.fixture
def artefactos_exportacion(tmp_path: Path) -> list[Path]:
    paquete = tmp_path / "KPoseFormer.mlpackage"
    (paquete / "Data").mkdir(parents=True)
    (paquete / "Manifest.json").write_text("{}", encoding="utf-8")
    (paquete / "Data" / "model.mlmodel").write_bytes(b"spec")
    rutas = [paquete]
    for nombre in ("paridad.json", "artefacto.json", "avisos.txt"):
        (tmp_path / nombre).write_text("{}", encoding="utf-8")
        rutas.append(tmp_path / nombre)
    return rutas


ETIQUETAS_EXPORTACION = {"modelo.version": "1", "modelo.sha256": HUELLA, "aceptacion.run_id": "r" * 32}
METRICAS_PARIDAD = {"paridad.cpu.prueba.max_diferencia": 0.003, "paridad.cpu.prueba.clase_distinta": 0.0, "coreml.bytes": 814871.0}


def test_sin_exportaciones(registro) -> None:
    assert registro.exportaciones() == []


def test_una_exportacion_terminada(registro, artefactos_exportacion) -> None:
    id_exportacion = registro.abrir_exportacion(ETIQUETAS_EXPORTACION)
    registro.terminar_exportacion(id_exportacion, METRICAS_PARIDAD, artefactos_exportacion, "superada")

    [exportacion] = registro.exportaciones()
    assert exportacion.run_id == id_exportacion
    assert exportacion.nombre == "exportacion-01"
    assert exportacion.estado is EstadoConsulta.TERMINADA
    assert exportacion.paridad == "superada"
    assert exportacion.metricas == pytest.approx(METRICAS_PARIDAD)
    assert exportacion.etiquetas["modelo.version"] == "1"
    assert set(registro.artefactos(id_exportacion)) == {
        "KPoseFormer.mlpackage/Manifest.json", "KPoseFormer.mlpackage/Data/model.mlmodel",
        "paridad.json", "artefacto.json", "avisos.txt",
    }


def test_las_exportaciones_se_numeran(registro, artefactos_exportacion) -> None:
    primera = registro.abrir_exportacion(ETIQUETAS_EXPORTACION)
    registro.terminar_exportacion(primera, METRICAS_PARIDAD, artefactos_exportacion, "fallida")
    segunda = registro.abrir_exportacion(ETIQUETAS_EXPORTACION)

    nombres = {e.run_id: e.nombre for e in registro.exportaciones()}

    assert nombres == {primera: "exportacion-01", segunda: "exportacion-02"}
    assert next(e for e in registro.exportaciones() if e.run_id == primera).paridad == "fallida"


def test_una_exportacion_fallida(registro) -> None:
    id_exportacion = registro.abrir_exportacion(ETIQUETAS_EXPORTACION)

    registro.fallar_exportacion(id_exportacion, "la conversión falló")

    [exportacion] = registro.exportaciones()
    assert exportacion.estado is EstadoConsulta.FALLIDA
    assert exportacion.paridad is None
    assert exportacion.etiquetas["estado.motivo"] == "la conversión falló"


def test_las_exportaciones_no_son_consultas(registro, artefactos_exportacion) -> None:
    id_exportacion = registro.abrir_exportacion(ETIQUETAS_EXPORTACION)
    registro.terminar_exportacion(id_exportacion, METRICAS_PARIDAD, artefactos_exportacion, "superada")

    assert registro.evaluaciones() == []


def test_paridad_desconocida(registro, artefactos_exportacion) -> None:
    id_exportacion = registro.abrir_exportacion(ETIQUETAS_EXPORTACION)

    with pytest.raises(ValueError):
        registro.terminar_exportacion(id_exportacion, METRICAS_PARIDAD, artefactos_exportacion, "regular")


def test_una_version_por_numero(registro, artefactos) -> None:
    version = registro.registrar_version(_consulta_terminada(registro, artefactos), _etiquetas_version())

    assert registro.version(1) == version
    assert registro.version(7) is None


def test_mlflow_guarda_la_exportacion_en_su_experimento(artefactos_exportacion) -> None:
    registro = RegistroAceptacionMlflow()

    id_exportacion = registro.abrir_exportacion(ETIQUETAS_EXPORTACION)
    registro.terminar_exportacion(id_exportacion, METRICAS_PARIDAD, artefactos_exportacion, "superada")

    cliente = MlflowClient(tracking_uri=uri_seguimiento())
    run = cliente.get_run(id_exportacion)
    assert run.info.run_name == "exportacion-01"
    assert run.data.tags["tipo"] == "exportacion"
    assert run.data.tags["paridad"] == "superada"
    assert cliente.get_experiment(run.info.experiment_id).name == EXPERIMENTO_EXPORTACION
