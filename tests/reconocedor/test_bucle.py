"""Una corrida de selección y el modelo final (TT-14; TT-22, research R-06; tasks T025)."""

import math
from pathlib import Path

import pytest
import torch

from signai_ml.reconocedor import bucle
from signai_ml.reconocedor.bucle import entrenar, entrenar_final
from signai_ml.reconocedor.configuracion import Configuracion, Entrenamiento, Forma, configuracion_desde_dict
from signai_ml.reconocedor.datos import cargar_final, cargar_rotacion
from signai_ml.reconocedor.registro import EstadoCorrida

METRICAS_EPOCA = {
    "entrenamiento.perdida", "entrenamiento.top1", "validacion.perdida", "validacion.top1",
    "validacion.f1_macro", "validacion.acierto_95", "validacion.falsas_95",
}
CPU = torch.device("cpu")


def _cfg(forma: Forma, **entrenamiento) -> Configuracion:
    valores = {"tasa": 3e-3, "epocas": 2, "lote": 8, "dropout": 0.0, **entrenamiento}
    return Configuracion("diminuta", forma, Entrenamiento(**valores), ())


@pytest.fixture
def datos(raiz_sintetica: Path, lector_seleccion):
    return cargar_rotacion(raiz_sintetica, "a", lector_seleccion)


@pytest.fixture
def datos_final(raiz_sintetica: Path, lector_final):
    return cargar_final(raiz_sintetica, lector_final)


def test_registra_cada_epoca_y_elige_la_mejor(datos, forma_diminuta: Forma) -> None:
    epocas: dict[int, dict[str, float]] = {}
    resultado = entrenar(datos, _cfg(forma_diminuta), 215, lambda e, m: epocas.update({e: m}), CPU)
    assert resultado.estado is EstadoCorrida.TERMINADA
    assert sorted(epocas) == [1, 2]
    assert all(set(m) == METRICAS_EPOCA for m in epocas.values())
    mejor = max(epocas, key=lambda e: (epocas[e]["validacion.top1"], epocas[e]["validacion.f1_macro"]))
    assert resultado.epoca_elegida == mejor
    assert resultado.metricas["top1"] == epocas[mejor]["validacion.top1"]
    assert set(resultado.metricas) == {"top1", "f1_macro", "acierto_95", "falsas_95", "perdida"}
    assert len(resultado.por_clase) == 4
    assert resultado.duracion_s > 0


def test_supera_el_azar(datos, forma_diminuta: Forma) -> None:
    resultado = entrenar(datos, _cfg(forma_diminuta, epocas=8), 215, lambda e, m: None, CPU)
    assert resultado.metricas["top1"] > 0.5  # 4 clases: el azar es 0,25


def test_conserva_el_modelo_de_la_epoca_elegida(datos, forma_diminuta: Forma) -> None:
    resultado = entrenar(datos, _cfg(forma_diminuta), 215, lambda e, m: None, CPU)
    modelo = resultado.modelo.eval()
    with torch.no_grad():
        prob = torch.softmax(modelo(datos.x_validacion), dim=1)
    acierto = (prob.argmax(dim=1) == datos.y_validacion).float().mean().item()
    assert acierto == pytest.approx(resultado.metricas["top1"])


def test_misma_semilla_mismas_metricas_en_cpu(datos, forma_diminuta: Forma) -> None:
    def correr(semilla: int) -> list:
        epocas: list = []
        entrenar(datos, _cfg(forma_diminuta), semilla, lambda e, m: epocas.append(m), CPU)
        return epocas

    assert correr(215) == correr(215)
    assert correr(215) != correr(216)


def test_el_aumento_solo_toca_el_entrenamiento(datos, forma_diminuta: Forma, monkeypatch) -> None:
    vistos: list[int] = []
    original = bucle.aplicar

    def espia(x, aumento, generador):
        vistos.append(x.shape[0])
        return original(x, aumento, generador)

    monkeypatch.setattr(bucle, "aplicar", espia)
    cfg = configuracion_desde_dict(
        "aumentada",
        {"forma": vars(forma_diminuta), "entrenamiento": {"epocas": 1, "lote": 8, "dropout": 0.0},
         "aumento": {"escala": {"probabilidad": 1.0, "intensidad": 0.1}}},
    )
    entrenar(datos, cfg, 215, lambda e, m: None, CPU)
    assert sum(vistos) == datos.x_entrenamiento.shape[0]  # solo lotes de entrenamiento


def test_perdida_no_finita_es_fallida(datos, forma_diminuta: Forma, monkeypatch) -> None:
    monkeypatch.setattr(bucle, "_perdida_lote", lambda *a, **k: torch.tensor(math.nan, requires_grad=True))
    resultado = entrenar(datos, _cfg(forma_diminuta), 215, lambda e, m: None, CPU)
    assert resultado.estado is EstadoCorrida.FALLIDA
    assert "época 1" in resultado.motivo and resultado.modelo is None


def test_programacion_constante_y_sin_recorte(datos, forma_diminuta: Forma) -> None:
    cfg = _cfg(forma_diminuta, epocas=1, programacion="constante", recorte_gradiente=0.0, suavizado_etiquetas=0.1)
    assert entrenar(datos, cfg, 215, lambda e, m: None, CPU).estado is EstadoCorrida.TERMINADA


def test_dispositivo_por_defecto(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert bucle.dispositivo() == CPU
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert bucle.dispositivo() == torch.device("mps")


# --- TT-22 · modelo final (research R-06) ----------------------------------------------------------


def test_final_entrena_exactamente_las_epocas_pedidas_sin_validar(datos_final, forma_diminuta: Forma) -> None:
    epocas: dict[int, dict[str, float]] = {}
    resultado = entrenar_final(datos_final, _cfg(forma_diminuta, epocas=5), 215, 3,
                               lambda e, m: epocas.update({e: m}), CPU)
    assert resultado.estado is EstadoCorrida.TERMINADA and resultado.epoca_elegida == 3
    assert sorted(epocas) == [1, 2, 3]
    assert all(set(m) == {"entrenamiento.perdida", "entrenamiento.top1"} for m in epocas.values())
    assert resultado.metricas is None and resultado.modelo is not None and resultado.duracion_s > 0


def test_final_sigue_la_programacion_de_toda_la_configuracion(
    datos_final, forma_diminuta: Forma, monkeypatch
) -> None:
    # La tasa sigue el coseno con T_max = épocas de la configuración, no las pedidas (R-06).
    vistos: list[int] = []
    original = torch.optim.lr_scheduler.CosineAnnealingLR

    def espia(optimizador, T_max, *a, **k):
        vistos.append(T_max)
        return original(optimizador, T_max, *a, **k)

    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", espia)
    entrenar_final(datos_final, _cfg(forma_diminuta, epocas=7), 215, 2, lambda e, m: None, CPU)
    assert vistos == [7]


def test_final_coincide_con_la_seleccion_en_las_mismas_epocas(
    raiz_sintetica: Path, lector_seleccion, forma_diminuta: Forma
) -> None:
    # Con los mismos datos, la misma semilla y la misma programación, parar en E* da los mismos pesos
    # que la selección en su época E*: el final no hace nada distinto salvo no validar.
    datos = cargar_rotacion(raiz_sintetica, "a", lector_seleccion)
    from signai_ml.reconocedor.datos import DatosFinal

    como_final = DatosFinal(datos.x_entrenamiento, datos.y_entrenamiento, datos.clases, datos.glosas, datos.huellas)
    cfg = _cfg(forma_diminuta, epocas=1)
    seleccion = entrenar(datos, cfg, 215, lambda e, m: None, CPU)
    final = entrenar_final(como_final, cfg, 215, 1, lambda e, m: None, CPU)
    for clave, valor in seleccion.modelo.state_dict().items():
        assert torch.equal(valor, final.modelo.state_dict()[clave]), clave


def test_final_con_perdida_no_finita_es_fallido(datos_final, forma_diminuta: Forma, monkeypatch) -> None:
    monkeypatch.setattr(bucle, "_perdida_lote", lambda *a, **k: torch.tensor(math.nan, requires_grad=True))
    resultado = entrenar_final(datos_final, _cfg(forma_diminuta), 215, 2, lambda e, m: None, CPU)
    assert resultado.estado is EstadoCorrida.FALLIDA and resultado.modelo is None


@pytest.mark.parametrize("epocas", [0, 3])
def test_final_epocas_fuera_de_la_configuracion(datos_final, forma_diminuta: Forma, epocas: int) -> None:
    with pytest.raises(ValueError, match="épocas"):
        entrenar_final(datos_final, _cfg(forma_diminuta, epocas=2), 215, epocas, lambda e, m: None, CPU)
