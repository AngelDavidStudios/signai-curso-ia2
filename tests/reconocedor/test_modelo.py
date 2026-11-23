"""KPoseFormer (contracts/modelo.md M-1, M-2; spec FR-001, FR-002; research R-02)."""

from pathlib import Path

import pytest
import torch

from signai_ml.reconocedor.configuracion import Forma
from signai_ml.reconocedor.modelo import FORMATO, KPoseFormer, cargar, contar_parametros, guardar

PARTIDA = Forma(4, 98, 7, 256)


def test_entrada_75_96_salida_100() -> None:
    torch.manual_seed(0)
    modelo = KPoseFormer(PARTIDA).eval()
    logits = modelo(torch.rand(3, 75, 96))
    assert logits.shape == (3, 100)
    assert torch.allclose(torch.softmax(logits, dim=1).sum(dim=1), torch.ones(3))


@pytest.mark.parametrize(("clases", "parametros"), [(100, 381_121), (17, 372_904)])
def test_recuento_de_parametros(clases: int, parametros: int) -> None:
    assert contar_parametros(KPoseFormer(PARTIDA, clases=clases)) == parametros


def test_dimension_distinta_de_98_anade_proyeccion() -> None:
    modelo = KPoseFormer(Forma(2, 64, 8, 128)).eval()
    assert isinstance(modelo.proyeccion, torch.nn.Linear)
    assert modelo(torch.rand(2, 75, 96)).shape == (2, 100)
    assert isinstance(KPoseFormer(PARTIDA).proyeccion, torch.nn.Identity)


def test_cabezas_que_no_dividen_la_dimension() -> None:
    with pytest.raises(ValueError, match="cabezas"):
        KPoseFormer(Forma(4, 98, 8, 256))


def test_pesos_iniciales_solo_de_la_semilla() -> None:
    def pesos(semilla: int) -> torch.Tensor:
        torch.manual_seed(semilla)
        return torch.cat([p.flatten() for p in KPoseFormer(PARTIDA).parameters()])

    assert torch.equal(pesos(215), pesos(215))
    assert not torch.equal(pesos(215), pesos(216))


def test_solo_operaciones_convertibles() -> None:
    prohibidos = (torch.nn.TransformerEncoder, torch.nn.TransformerEncoderLayer, torch.nn.MultiheadAttention)
    assert not any(isinstance(m, prohibidos) for m in KPoseFormer(PARTIDA).modules())


def test_eval_es_determinista_y_admite_ausentes() -> None:
    torch.manual_seed(0)
    modelo = KPoseFormer(PARTIDA, dropout=0.3).eval()
    x = torch.rand(2, 75, 96)
    x[:, :10, :20] = -1.0
    with torch.no_grad():
        a, b = modelo(x), modelo(x)
    assert torch.equal(a, b)
    assert not torch.isnan(a).any()


def test_guardar_y_cargar(tmp_path: Path) -> None:
    torch.manual_seed(0)
    forma = Forma(1, 98, 2, 16)
    modelo = KPoseFormer(forma, clases=4).eval()
    clases = [f"{n}-Sint{n}" for n in range(1, 5)]
    glosas = [f"glosa{n}" for n in range(1, 5)]
    ruta = guardar(tmp_path / "modelo.pt", modelo, forma, clases, glosas,
                   corrida={"barrido": "b01", "semilla": 215}, huellas={"particion.huella": "f" * 64})
    cargado, meta = cargar(ruta)
    assert meta["formato"] == FORMATO
    assert meta["clases"] == clases and meta["glosas"] == glosas
    assert meta["forma"] == {"capas": 1, "dimension": 98, "cabezas": 2, "intermedia": 16}
    x = torch.rand(2, 75, 96)
    with torch.no_grad():
        assert torch.allclose(modelo(x), cargado(x))


def test_cargar_rechaza_otro_formato(tmp_path: Path) -> None:
    ruta = tmp_path / "otro.pt"
    torch.save({"formato": "otro"}, ruta)
    with pytest.raises(ValueError, match="formato"):
        cargar(ruta)


# ── Atención convertible a Core ML (TT-15 · SP-342; research R-06 de 007) ───────────────────────


def _atencion_original(atencion, x: torch.Tensor) -> torch.Tensor:
    """El `forward` de TT-14 tal como se entrenó, copiado literalmente: es el oráculo del cambio."""
    import math

    def separar(y: torch.Tensor) -> torch.Tensor:
        b, t, d = y.shape
        return y.reshape(b, t, atencion.cabezas, d // atencion.cabezas).transpose(1, 2)

    b, t, d = x.shape
    q, k, v = separar(atencion.consulta(x)), separar(atencion.clave(x)), separar(atencion.valor(x))
    pesos = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d // atencion.cabezas), dim=-1)
    return atencion.salida(torch.matmul(pesos, v).transpose(1, 2).reshape(b, t, d))


@pytest.mark.parametrize("forma", [Forma(), Forma(capas=1, dimension=16, cabezas=2, intermedia=16)])
def test_la_atencion_da_lo_mismo_que_la_entrenada(forma: Forma) -> None:
    torch.manual_seed(7)
    modelo = KPoseFormer(forma).eval()
    x = torch.randn(3, 75, forma.dimension)

    with torch.no_grad():
        for capa in modelo.capas:
            assert torch.equal(capa.atencion(x), _atencion_original(capa.atencion, x))


def test_el_state_dict_no_cambia() -> None:
    claves = set(KPoseFormer(Forma()).state_dict())

    assert not any("escala" in c or "por_cabeza" in c for c in claves)
    assert len(claves) == len(KPoseFormer(Forma()).state_dict())


def test_la_forma_de_partida_convierte_a_mlprogram() -> None:
    import coremltools as ct
    import numpy as np

    torch.manual_seed(0)
    modelo = KPoseFormer(Forma()).eval()
    trazado = torch.jit.trace(modelo, torch.zeros(1, 75, 96))

    convertido = ct.convert(
        trazado,
        inputs=[ct.TensorType(name="ventana", shape=(1, 75, 96), dtype=np.float16)],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )

    assert convertido.get_spec().WhichOneof("Type") == "mlProgram"
