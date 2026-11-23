"""`signai-aceptacion exportar` de extremo a extremo (contracts/cli.md; spec US3, FR-009…FR-014, SC-005, SC-006).

Candidato sintético entrenado (4 clases, predicciones claras), evaluado y registrado con la propia
CLI. La conversión a Core ML y la paridad en CPU y en el Neural Engine son reales.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from evaluacion.sinteticos import PRUEBA_POR_CLASE, TOTAL_PRUEBA, escribir_raiz_prueba, lector_prueba
from signai_ml.aceptacion import COREML, NOMBRE_PAQUETE, cli
from signai_ml.aceptacion.artefacto import huella_paquete, leer_metadatos, validar
from signai_ml.aceptacion.candidato import leer
from signai_ml.aceptacion.ejecucion import EjecutorCoreml
from signai_ml.aceptacion.registro import RegistroAceptacionEnMemoria, RegistroAceptacionMlflow


def _limpio(_raiz: Path) -> tuple[str, bool]:
    return "f" * 40, False


def _main(orden: list[str], raiz: Path, registro, **kwargs) -> int:
    opciones = {
        "lector": lector_prueba, "comprobar_codigo": _limpio, "total": TOTAL_PRUEBA, "por_clase": PRUEBA_POR_CLASE,
    } | kwargs
    return cli.main(orden, raiz=raiz, registro=registro, **opciones)


@pytest.fixture
def raiz(tmp_path: Path) -> Path:
    return escribir_raiz_prueba(tmp_path / "repo", entrenado=True)


@pytest.fixture
def registro():
    return RegistroAceptacionEnMemoria()


@pytest.fixture
def aceptado(raiz: Path, registro, capsys) -> Path:
    assert _main(["evaluar"], raiz, registro) == 0
    assert registro.evaluaciones()[0].decision == "aceptado"
    assert _main(["registrar"], raiz, registro) == 0
    capsys.readouterr()
    return raiz


def _exportar(raiz: Path, registro, casos: Path, *extra: str, **kwargs) -> int:
    return _main(["exportar", "--casos", str(casos), *extra], raiz, registro, **kwargs)


class EjecutorAlterado(EjecutorCoreml):
    """Imita una conversión que predice mal sin dar error: desplaza probabilidad hacia la clase 0."""

    def probabilidades(self, paquete, ventanas):
        p = super().probabilidades(paquete, ventanas)
        p[:, 0] += 0.05
        return p / p.sum(axis=1, keepdims=True)


def test_exporta_con_la_paridad_superada(aceptado: Path, registro, ruta_casos_referencia: Path, capsys) -> None:
    assert _exportar(aceptado, registro, ruta_casos_referencia) == 0

    salida = capsys.readouterr().out
    assert "Versión 1 de reconocedor-kposeformer" in salida
    assert "mlprogram · FLOAT16 · iOS17" in salida
    assert "CPU_ONLY" in salida and "ALL" in salida
    assert "Paridad: SUPERADA" in salida
    assert "uv run dvc add modelos/coreml" in salida
    destino = aceptado / COREML
    paquete = destino / NOMBRE_PAQUETE
    candidato = leer(aceptado)
    validar(paquete, candidato.clases, candidato.glosas)
    metadatos = leer_metadatos(paquete)
    assert metadatos["cifra"] == "1.000 · prueba aleatoria 1.800 · consulta 1"
    assert metadatos["modelo.version"] == "1"
    assert metadatos["modelo.sha256"] == candidato.modelo_sha256
    paridad = json.loads((destino / "paridad.json").read_text(encoding="utf-8"))
    assert paridad["superada"] is True
    assert set(paridad["unidades"]) == {"cpu", "todas"}
    assert set(paridad["unidades"]["cpu"]) == {"prueba", "referencia"}
    assert paridad["unidades"]["cpu"]["prueba"]["casos"] == TOTAL_PRUEBA
    assert paridad["unidades"]["todas"]["referencia"]["casos"] == 12
    artefacto = json.loads((destino / "artefacto.json").read_text(encoding="utf-8"))
    assert artefacto["sha256"] == huella_paquete(paquete)
    version = registro.version_con_alias()
    assert version.etiquetas["paridad"] == "superada"
    assert version.etiquetas["coreml.sha256"] == huella_paquete(paquete)
    [exportacion] = registro.exportaciones()
    assert version.etiquetas["coreml.run_id"] == exportacion.run_id
    assert exportacion.paridad == "superada"
    assert exportacion.metricas["coreml.bytes"] == artefacto["bytes"]
    assert "paridad.todas.prueba.max_diferencia" in exportacion.metricas
    assert len(registro.evaluaciones()) == 1  # exportar no consulta la prueba como evaluación nueva


def test_una_paridad_fallida_no_escribe_nada(aceptado: Path, registro, ruta_casos_referencia: Path, capsys) -> None:
    codigo = _exportar(aceptado, registro, ruta_casos_referencia, ejecutor=EjecutorAlterado)

    assert codigo == 1
    captura = capsys.readouterr()
    assert "Paridad: FALLIDA" in captura.out
    assert "paridad" in captura.err.lower()
    assert not (aceptado / COREML).exists()
    [exportacion] = registro.exportaciones()
    assert exportacion.paridad == "fallida"
    assert "paridad" not in registro.version_con_alias().etiquetas


def test_pesos_distintos_de_los_registrados(aceptado: Path, registro, ruta_casos_referencia: Path, capsys) -> None:
    modelo = aceptado / "modelos" / "reconocedor" / "modelo.pt"
    modelo.write_bytes(modelo.read_bytes() + b"\0")  # el candidato ya no es el registrado

    assert _exportar(aceptado, registro, ruta_casos_referencia) == 1

    assert registro.exportaciones() == []
    assert not (aceptado / COREML).exists()


def test_arbol_sucio(aceptado: Path, registro, ruta_casos_referencia: Path, capsys) -> None:
    assert _exportar(aceptado, registro, ruta_casos_referencia, comprobar_codigo=None) == 3

    assert "sin confirmar" in capsys.readouterr().err
    assert registro.exportaciones() == []


def test_huellas_distintas_no_convierten(aceptado: Path, registro, ruta_casos_referencia: Path, capsys) -> None:
    ficha = aceptado / "modelos" / "reconocedor" / "candidato.json"
    datos = json.loads(ficha.read_text(encoding="utf-8"))
    datos["huellas"]["corpus.sha256_indice"] = "0" * 64
    ficha.write_text(json.dumps(datos), encoding="utf-8")

    assert _exportar(aceptado, registro, ruta_casos_referencia) == 1

    assert "corpus.sha256_indice" in capsys.readouterr().err
    assert registro.exportaciones() == []


def test_casos_de_otra_ruta(aceptado: Path, registro, ruta_casos_referencia: Path, tmp_path: Path) -> None:
    casos = json.loads(ruta_casos_referencia.read_text(encoding="utf-8"))
    casos["casos"] = casos["casos"][:5]
    otra = tmp_path / "casos.json"
    otra.write_text(json.dumps(casos), encoding="utf-8")

    assert _exportar(aceptado, registro, otra) == 0

    paridad = json.loads((aceptado / COREML / "paridad.json").read_text(encoding="utf-8"))
    assert paridad["unidades"]["cpu"]["referencia"]["casos"] == 5


def test_casos_inexistentes(aceptado: Path, registro, tmp_path: Path, capsys) -> None:
    assert _exportar(aceptado, registro, tmp_path / "no-existe.json") == 2

    assert "no-existe.json" in capsys.readouterr().err
    assert registro.exportaciones() == []


def test_sin_version_aceptada(raiz: Path, registro, ruta_casos_referencia: Path, capsys) -> None:
    assert _main(["evaluar"], raiz, registro) == 0  # evaluado, pero sin registrar

    assert _exportar(raiz, registro, ruta_casos_referencia) == 1

    assert "registrar" in capsys.readouterr().err
    assert registro.exportaciones() == []


def test_una_version_concreta(aceptado: Path, registro, ruta_casos_referencia: Path) -> None:
    assert _exportar(aceptado, registro, ruta_casos_referencia, "--version", "1") == 0
    assert _exportar(aceptado, registro, ruta_casos_referencia, "--version", "9") == 1


def test_exportar_de_nuevo_sustituye_el_artefacto(aceptado: Path, registro, ruta_casos_referencia: Path) -> None:
    assert _exportar(aceptado, registro, ruta_casos_referencia) == 0
    (aceptado / COREML / "sobrante.txt").write_text("x", encoding="utf-8")

    assert _exportar(aceptado, registro, ruta_casos_referencia) == 0

    assert not (aceptado / COREML / "sobrante.txt").exists()
    assert [e.nombre for e in sorted(registro.exportaciones(), key=lambda e: e.nombre)] == [
        "exportacion-01", "exportacion-02",
    ]


def test_con_mlflow(aceptado_mlflow: Path, ruta_casos_referencia: Path) -> None:
    registro = RegistroAceptacionMlflow()

    assert _exportar(aceptado_mlflow, registro, ruta_casos_referencia) == 0

    assert registro.version_con_alias().etiquetas["paridad"] == "superada"
    assert np.isclose(registro.exportaciones()[0].metricas["paridad.cpu.prueba.clase_distinta"], 0)


@pytest.fixture
def aceptado_mlflow(raiz: Path, capsys) -> Path:
    registro = RegistroAceptacionMlflow()
    assert _main(["evaluar"], raiz, registro) == 0
    assert _main(["registrar"], raiz, registro) == 0
    capsys.readouterr()
    return raiz
