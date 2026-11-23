"""Paridad real del modelo convertido a Core ML (constitución IV y V; spec FR-012, FR-015, SC-007).

Convierte de verdad en el runner un KPoseFormer diminuto a `mlprogram` en 16 bits. Comprueba que da
los mismos resultados que el original en PyTorch, sobre ventanas que no vio y sobre los 12 casos de
referencia del normalizador, en CPU y con todas las unidades (Neural Engine). Una conversión
divergente tiene que hacer fallar la paridad: ADR-0014 avisa de que ese fallo es silencioso.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from signai_ml.aceptacion import TOLERANCIA_PARIDAD
from signai_ml.aceptacion.conversion import ConvertidorCoreml
from signai_ml.aceptacion.ejecucion import UNIDADES, EjecutorCoreml, casos_referencia
from signai_ml.aceptacion.paridad import comparar, informe_paridad

ALTERACION = 2.0  # se suma al sesgo de la clase 0 de la capa de salida


def _probabilidades_torch(modelo: torch.nn.Module, ventanas: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(modelo(torch.from_numpy(ventanas)), dim=1).double().numpy()


@pytest.fixture(scope="module")
def paquete_correcto(tmp_path_factory, modelo_diminuto, clases_diminuto) -> Path:
    destino = tmp_path_factory.mktemp("coreml-correcto")
    return ConvertidorCoreml().convertir(modelo_diminuto, clases_diminuto, destino).paquete


@pytest.fixture(scope="module")
def paquete_alterado(tmp_path_factory, modelo_diminuto, clases_diminuto) -> Path:
    alterado = copy.deepcopy(modelo_diminuto)
    with torch.no_grad():
        alterado.clasificador.bias[0] += ALTERACION
    destino = tmp_path_factory.mktemp("coreml-alterado")
    return ConvertidorCoreml().convertir(alterado, clases_diminuto, destino).paquete


@pytest.fixture(scope="module")
def referencia(ruta_casos_referencia: Path) -> np.ndarray:
    return casos_referencia(ruta_casos_referencia)


def _informes(paquete: Path, modelo, ventanas_prueba, referencia) -> dict:
    x, y = ventanas_prueba
    original = {"prueba": _probabilidades_torch(modelo, x), "referencia": _probabilidades_torch(modelo, referencia)}
    informes = {}
    for unidad in UNIDADES:
        ejecutor = EjecutorCoreml(unidad)
        informes[unidad] = {
            "prueba": comparar(original["prueba"], ejecutor.probabilidades(paquete, x), y),
            "referencia": comparar(original["referencia"], ejecutor.probabilidades(paquete, referencia)),
        }
    return informes


def test_las_unidades_son_cpu_y_todas() -> None:
    assert UNIDADES == ("cpu", "todas")


def test_los_casos_de_referencia_son_las_12_ventanas_del_contrato(referencia: np.ndarray) -> None:
    assert referencia.shape == (12, 75, 96)
    assert referencia.dtype == np.float32


@pytest.mark.parametrize("unidad", UNIDADES)
def test_el_ejecutor_da_una_distribucion_por_ventana(unidad, paquete_correcto, ventanas_prueba) -> None:
    x, _ = ventanas_prueba

    probabilidades = EjecutorCoreml(unidad).probabilidades(paquete_correcto, x)

    assert probabilidades.shape == (len(x), 10)
    assert np.allclose(probabilidades.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("unidad", UNIDADES)
def test_la_conversion_correcta_supera_la_paridad(
    unidad, paquete_correcto, modelo_diminuto, ventanas_prueba, referencia
) -> None:
    informes = _informes(paquete_correcto, modelo_diminuto, ventanas_prueba, referencia)[unidad]

    for conjunto, informe in informes.items():
        assert informe.superado, (conjunto, informe)
        assert informe.clase_distinta == ()
        assert informe.max_diferencia <= TOLERANCIA_PARIDAD
    assert informes["prueba"].aciertos_original == informes["prueba"].aciertos_exportado


def test_la_conversion_correcta_supera_el_informe_completo(
    paquete_correcto, modelo_diminuto, ventanas_prueba, referencia
) -> None:
    assert informe_paridad(_informes(paquete_correcto, modelo_diminuto, ventanas_prueba, referencia)).superada


def test_una_conversion_divergente_hace_fallar_la_paridad(
    paquete_alterado, modelo_diminuto, ventanas_prueba, referencia
) -> None:
    informes = _informes(paquete_alterado, modelo_diminuto, ventanas_prueba, referencia)

    assert not informe_paridad(informes).superada
    for unidad in UNIDADES:
        assert max(i.max_diferencia for i in informes[unidad].values()) > TOLERANCIA_PARIDAD
