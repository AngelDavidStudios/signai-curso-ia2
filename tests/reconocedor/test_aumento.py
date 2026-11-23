"""Aumento de datos (spec FR-006; research R-08): nunca fabrica puntos ni mezcla los `-1`."""

import pytest
import torch

from signai_ml.reconocedor.aumento import aplicar
from signai_ml.reconocedor.configuracion import Aumento

ESPACIALES = ("desplazamiento", "escala", "rotacion", "ocultacion")
TEMPORALES = ("velocidad", "recorte_temporal")
INTENSIDAD = {
    "desplazamiento": 0.1, "escala": 0.2, "rotacion": 0.3, "ocultacion": 0.2, "velocidad": 0.3, "recorte_temporal": 0.6,
}


def _lote() -> torch.Tensor:
    generador = torch.Generator().manual_seed(0)
    x = torch.rand(4, 75, 96, generator=generador)
    puntos = x.view(4, 75, 48, 2)
    puntos[:, ::7, ::5] = -1.0
    return x


def _gen(semilla: int = 1) -> torch.Generator:
    return torch.Generator().manual_seed(semilla)


@pytest.mark.parametrize("tipo", ESPACIALES + TEMPORALES)
def test_determinista_por_semilla(tipo: str) -> None:
    x = _lote()
    aumento = ((tipo, Aumento(1.0, INTENSIDAD[tipo])),)
    assert torch.equal(aplicar(x, aumento, _gen(3)), aplicar(x, aumento, _gen(3)))
    assert not torch.equal(aplicar(x, aumento, _gen(3)), aplicar(x, aumento, _gen(4)))


@pytest.mark.parametrize("tipo", ESPACIALES)
def test_los_ausentes_siguen_ausentes(tipo: str) -> None:
    x = _lote()
    y = aplicar(x, ((tipo, Aumento(1.0, INTENSIDAD[tipo])),), _gen())
    assert torch.all(y[x == -1] == -1)
    assert y.shape == x.shape


def test_desplazamiento_y_escala_cambian_los_presentes() -> None:
    x = _lote()
    for tipo in ("desplazamiento", "escala"):
        y = aplicar(x, ((tipo, Aumento(1.0, INTENSIDAD[tipo])),), _gen())
        assert not torch.equal(y[x != -1], x[x != -1])


def test_ocultacion_solo_convierte_presentes_en_ausentes() -> None:
    x = _lote()
    y = aplicar(x, (("ocultacion", Aumento(1.0, 0.2)),), _gen())
    cambiados = y != x
    assert torch.all(y[cambiados] == -1)
    puntos = y.view(4, 75, 48, 2)
    assert torch.all((puntos[..., 0] == -1) == (puntos[..., 1] == -1))  # el punto entero, x e y
    assert cambiados.any()


@pytest.mark.parametrize("tipo", TEMPORALES)
def test_temporales_usan_el_fotograma_mas_cercano(tipo: str) -> None:
    x = _lote()
    y = aplicar(x, ((tipo, Aumento(1.0, INTENSIDAD[tipo])),), _gen())
    assert y.shape == (4, 75, 96)
    for clip in range(4):
        originales = {tuple(f.tolist()) for f in x[clip]}
        assert all(tuple(f.tolist()) in originales for f in y[clip])


def test_recorte_temporal_toma_un_tramo_de_al_menos_la_intensidad() -> None:
    x = torch.arange(75, dtype=torch.float32).view(1, 75, 1).expand(1, 75, 96).contiguous()
    for semilla in range(20):
        y = aplicar(x, (("recorte_temporal", Aumento(1.0, 0.6)),), _gen(semilla))
        fotogramas = y[0, :, 0]
        assert fotogramas[-1] - fotogramas[0] + 1 >= round(0.6 * 75) - 1
        assert torch.all(fotogramas[1:] >= fotogramas[:-1])


def test_probabilidad_cero_es_la_identidad() -> None:
    x = _lote()
    aumento = tuple((tipo, Aumento(0.0, INTENSIDAD[tipo])) for tipo in ESPACIALES + TEMPORALES)
    assert torch.equal(aplicar(x, aumento, _gen()), x)


def test_sin_aumento_no_copia_ni_cambia() -> None:
    x = _lote()
    assert aplicar(x, (), _gen()) is x


# --- TT-22 · rotación (research R-07) -----------------------------------------------------------------


def _presentes(x: torch.Tensor) -> torch.Tensor:
    return (x.view(x.shape[0], 75, 48, 2) != -1.0).all(dim=-1)


def test_rotacion_conserva_la_norma_de_cada_punto() -> None:
    x = _lote() * 2 - 1  # coordenadas a ambos lados del origen, como tras normalizar (N-5)
    x.view(4, 75, 48, 2)[:, ::7, ::5] = -1.0
    y = aplicar(x, (("rotacion", Aumento(1.0, 0.4)),), _gen())
    presentes = _presentes(x)
    norma = lambda t: t.view(4, 75, 48, 2).norm(dim=-1)  # noqa: E731
    assert torch.allclose(norma(y)[presentes], norma(x)[presentes], atol=1e-5)
    assert not torch.equal(y, x)


def test_rotacion_con_angulo_cero_es_la_identidad(monkeypatch) -> None:
    from signai_ml.reconocedor import aumento

    monkeypatch.setattr(aumento, "_aleatorio", lambda generador, *forma: torch.full(forma, 0.5))
    x = _lote()
    assert torch.allclose(aplicar(x, (("rotacion", Aumento(1.0, 0.4)),), _gen()), x, atol=1e-6)


def test_rotacion_dentro_del_angulo_maximo() -> None:
    import math

    x = torch.zeros(64, 75, 96)
    x.view(64, 75, 48, 2)[..., 0] = 1.0  # todos los puntos en (1, 0)
    y = aplicar(x, (("rotacion", Aumento(1.0, 0.2)),), _gen()).view(64, 75, 48, 2)
    angulos = torch.atan2(y[..., 1], y[..., 0])
    assert angulos.abs().max().item() <= 0.2 + 1e-6
    assert angulos.abs().max().item() > 0.1  # usa el rango, no un ángulo fijo
    assert torch.allclose(angulos[:, :1, :1], angulos[:, :1, :1]) and not math.isnan(angulos.sum().item())
