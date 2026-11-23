"""Paridad de la normalización contra los casos de referencia del contrato (N-8; FR-017, SC-004).

Una implementación cumple el contrato si reproduce cada salida esperada con |Δ| ≤ 1·10⁻⁴. La
prueba de mutación demuestra que los casos discriminan: quitar un paso o cambiar el orden falla.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from signai_ml.preproceso import NORMALIZACION
from signai_ml.preproceso.normalizacion import (
    Imputacion,
    NormalizacionHombros,
    Normalizador,
    PasoNormalizacion,
    Remuestreo,
    RespaldoMuneca,
    Secuencia,
    aplicar_pasos,
)

CASOS = Path(__file__).parents[1] / "referencias" / "preprocesamiento" / "normalizacion-casos.json"
# La huella que declara `signai-architecture/contracts/normalizacion.md` (N-8).
SHA256_CONTRATO = "1e2a3f935ff22a8c9b2161dd1c48ee386ba15beb01faa8057ac105d313287660"
TOLERANCIA = 1e-4


@pytest.fixture(scope="module")
def referencia() -> dict:
    return json.loads(CASOS.read_text(encoding="utf-8"))


def test_la_copia_local_es_la_del_contrato() -> None:
    assert hashlib.sha256(CASOS.read_bytes()).hexdigest() == SHA256_CONTRATO


def test_los_casos_son_de_la_version_vigente_y_con_su_tolerancia(referencia: dict) -> None:
    assert referencia["contrato"] == "normalizacion"
    assert referencia["version"] == NORMALIZACION
    assert referencia["tolerancia"] == TOLERANCIA
    assert len(referencia["casos"]) >= 11


def test_reproduce_cada_caso_dentro_de_la_tolerancia(referencia: dict) -> None:
    normalizador = Normalizador()
    for caso in referencia["casos"]:
        ventana = normalizador.normalizar(np.asarray(caso["entrada"], dtype=np.float32))
        esperado = np.asarray(caso["salida"], dtype=np.float32)
        diferencia = float(np.abs(ventana.valores - esperado).max())
        assert diferencia <= TOLERANCIA, f"{caso['nombre']}: |Δ| = {diferencia:.2e}"
        assert ventana.sin_normalizar == caso["sin_normalizar"], caso["nombre"]


class _Identidad:
    def aplicar(self, secuencia: Secuencia) -> Secuencia:
        return secuencia


class _AnchoMedio:
    """Normaliza con la MEDIA del ancho de hombros en lugar de la mediana."""

    def aplicar(self, secuencia: Secuencia) -> Secuencia:
        izq, der = secuencia.valores[:, 0], secuencia.valores[:, 1]
        ancho = float(np.hypot(*(izq - der).T).mean())
        if ancho < 1e-6:
            return secuencia
        return Secuencia((secuencia.valores - ((izq + der) / 2)[:, None, :]) / ancho, secuencia.presente)


class _CentroGlobal:
    """Centra en el centro de hombros MEDIO del tramo en lugar del de cada fotograma."""

    def aplicar(self, secuencia: Secuencia) -> Secuencia:
        izq, der = secuencia.valores[:, 0], secuencia.valores[:, 1]
        ancho = float(np.median(np.hypot(*(izq - der).T)))
        if ancho < 1e-6:
            return secuencia
        return Secuencia((secuencia.valores - ((izq + der) / 2).mean(axis=0)) / ancho, secuencia.presente)


class _RemuestreoVecino:
    """Remuestrea a 75 con el fotograma MÁS PRÓXIMO en lugar de interpolar entre los vecinos."""

    def aplicar(self, secuencia: Secuencia) -> Secuencia:
        origen = secuencia.valores.shape[0]
        indices = np.rint(np.linspace(0, origen - 1, 75)).astype(int)
        return Secuencia(secuencia.valores[indices], secuencia.presente[indices], secuencia.sin_normalizar)


MUTANTES: dict[str, tuple[PasoNormalizacion, ...]] = {
    "remuestreo_vecino": (RespaldoMuneca(), Imputacion(), NormalizacionHombros(), _RemuestreoVecino()),
    "sin_respaldo": (_Identidad(), Imputacion(), NormalizacionHombros(), Remuestreo(75)),
    "sin_imputar": (RespaldoMuneca(), _Identidad(), NormalizacionHombros(), Remuestreo(75)),
    "sin_hombros": (RespaldoMuneca(), Imputacion(), _Identidad(), Remuestreo(75)),
    "ancho_medio": (RespaldoMuneca(), Imputacion(), _AnchoMedio(), Remuestreo(75)),
    "centro_global": (RespaldoMuneca(), Imputacion(), _CentroGlobal(), Remuestreo(75)),
    "remuestrear_antes_de_imputar": (RespaldoMuneca(), Remuestreo(75), Imputacion(), NormalizacionHombros()),
    "imputar_antes_del_respaldo": (Imputacion(), RespaldoMuneca(), NormalizacionHombros(), Remuestreo(75)),
}


@pytest.mark.parametrize("nombre", sorted(MUTANTES))
def test_cada_mutacion_hace_fallar_algun_caso(referencia: dict, nombre: str) -> None:
    pasos = MUTANTES[nombre]
    fallos = 0
    for caso in referencia["casos"]:
        entrada = np.asarray(caso["entrada"], dtype=np.float32)
        try:
            obtenido = aplicar_pasos(pasos, entrada).valores
        except ValueError:
            fallos += 1
            continue
        esperado = np.asarray(caso["salida"], dtype=np.float32)
        if obtenido.shape != esperado.shape or not np.allclose(obtenido, esperado, atol=TOLERANCIA, rtol=0):
            fallos += 1
    assert fallos > 0, f"la mutación {nombre} pasa todos los casos: los casos no la detectan"
