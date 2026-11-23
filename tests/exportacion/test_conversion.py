"""Conversión a Core ML en el formato de programa, 16 bits, iOS 17 (spec FR-011; research R-06; ADR-0014).

La interfaz es la que comprueba `CoreMLSignRecognizer` en la app: entrada `ventana` (1, 75, 96) y
salida `logits` (1, clases), las dos en FLOAT16.
"""

from pathlib import Path

import coremltools as ct
import pytest
import torch

from signai_ml.aceptacion import ENTRADA, NOMBRE_PAQUETE, SALIDA
from signai_ml.aceptacion.conversion import Conversion, ConvertidorCoreml
from signai_ml.aceptacion.errores import ArtefactoNoValido

FLOAT16 = 65552  # ArrayFeatureType.FLOAT16 en la especificación de Core ML


@pytest.fixture(scope="module")
def conversion(tmp_path_factory, modelo_diminuto, clases_diminuto) -> Conversion:
    return ConvertidorCoreml().convertir(modelo_diminuto, clases_diminuto, tmp_path_factory.mktemp("conversion"))


def _spec(conversion: Conversion):
    return ct.models.MLModel(str(conversion.paquete), skip_model_load=True).get_spec()


def test_el_paquete_se_llama_como_lo_busca_la_app(conversion: Conversion) -> None:
    assert conversion.paquete.name == NOMBRE_PAQUETE
    assert conversion.paquete.is_dir()


def test_formato_de_programa_especificacion_8(conversion: Conversion) -> None:
    spec = _spec(conversion)

    assert spec.specificationVersion == 8
    assert spec.WhichOneof("Type") == "mlProgram"


def test_entrada_y_salida_en_16_bits(conversion: Conversion) -> None:
    spec = _spec(conversion)
    [entrada], [salida] = spec.description.input, spec.description.output

    assert entrada.name == ENTRADA
    assert list(entrada.type.multiArrayType.shape) == [1, 75, 96]
    assert entrada.type.multiArrayType.dataType == FLOAT16
    assert salida.name == SALIDA
    assert list(salida.type.multiArrayType.shape) == [1, 10]
    assert salida.type.multiArrayType.dataType == FLOAT16


def test_los_avisos_se_devuelven_sin_lanzar(conversion: Conversion) -> None:
    assert isinstance(conversion.avisos, list)
    assert all(isinstance(aviso, str) for aviso in conversion.avisos)


def test_el_modelo_original_no_cambia(modelo_diminuto, clases_diminuto, tmp_path: Path) -> None:
    antes = {k: v.clone() for k, v in modelo_diminuto.state_dict().items()}

    ConvertidorCoreml().convertir(modelo_diminuto, clases_diminuto, tmp_path)

    assert all(torch.equal(antes[k], v) for k, v in modelo_diminuto.state_dict().items())
    assert not modelo_diminuto.training


def test_clases_que_no_corresponden_a_las_salidas(modelo_diminuto, tmp_path: Path) -> None:
    with pytest.raises(ArtefactoNoValido, match="clases"):
        ConvertidorCoreml().convertir(modelo_diminuto, ["solo-una"], tmp_path)


class _NoConvertible(torch.nn.Module):
    """Un modelo con una operación que coremltools 9.0 no implementa (`histc`; la FFT sí convierte)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.histc(x[0], bins=10).unsqueeze(0)


def test_una_operacion_no_admitida_es_un_artefacto_no_valido(tmp_path: Path) -> None:
    with pytest.raises(ArtefactoNoValido, match="conversión"):
        ConvertidorCoreml().convertir(_NoConvertible().eval(), [f"c{i}" for i in range(10)], tmp_path)
    assert not (tmp_path / NOMBRE_PAQUETE).exists()
