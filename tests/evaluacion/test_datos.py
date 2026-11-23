"""La prueba, solo por su contrato de lectura y ligada al candidato (spec FR-001, FR-002; research R-04)."""

import inspect
from pathlib import Path

import numpy as np
import pytest

from signai_ml.aceptacion.candidato import leer
from signai_ml.aceptacion.datos import DatosPrueba, cargar_prueba
from signai_ml.aceptacion.errores import ErrorAceptacion, HuellasDistintas, PruebaInesperada
from signai_ml.particion.lectura import para_evaluacion
from signai_ml.ventanas import Corpus

from .sinteticos import N_CLASES, PRUEBA_POR_CLASE, TOTAL_PRUEBA, LadoPruebaFalso, escribir_candidato, filas_prueba


def _cargar(raiz: Path, lector, **kwargs) -> DatosPrueba:
    return cargar_prueba(raiz, leer(raiz), lector=lector, total=TOTAL_PRUEBA, por_clase=PRUEBA_POR_CLASE, **kwargs)


def test_por_defecto_usa_el_contrato_de_la_prueba() -> None:
    assert inspect.signature(cargar_prueba).parameters["lector"].default is para_evaluacion


def test_lee_solo_la_prueba(raiz_prueba: Path, lector_prueba_falso) -> None:
    datos = _cargar(raiz_prueba, lector_prueba_falso)

    assert datos.x.shape == (TOTAL_PRUEBA, 75, 96)
    assert datos.x.dtype == np.float32
    assert datos.y.dtype == np.int64
    assert datos.y.tolist() == [c for c in range(N_CLASES) for _ in range(PRUEBA_POR_CLASE)]
    assert np.array_equal(datos.x, np.load(raiz_prueba / "datos" / "ventanas" / "ventanas.npy")[filas_prueba()])
    assert datos.clases == leer(raiz_prueba).clases
    assert datos.glosas == leer(raiz_prueba).glosas
    assert datos.huellas == leer(raiz_prueba).huellas


@pytest.mark.parametrize("clave", ["corpus.sha256_ventanas", "corpus.sha256_indice", "particion.huella"])
def test_huellas_distintas_detienen_antes_de_leer(
    raiz_prueba: Path, lector_prueba_falso, clave: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidato = leer(raiz_prueba)
    escribir_candidato(raiz_prueba, huellas=candidato.huellas | {clave: "0" * 64})

    def no_leer(*_):
        raise AssertionError("no debía leer ninguna ventana")

    monkeypatch.setattr(Corpus, "tomar", no_leer)
    with pytest.raises(HuellasDistintas, match=clave):
        _cargar(raiz_prueba, lector_prueba_falso)


def test_total_inesperado(raiz_prueba: Path) -> None:
    def lector_corto(_raiz):
        return LadoPruebaFalso(filas_prueba()[:-1], "e" * 64)

    with pytest.raises(PruebaInesperada, match=str(TOTAL_PRUEBA)):
        _cargar(raiz_prueba, lector_corto)


def test_clases_desequilibradas(raiz_prueba: Path) -> None:
    filas = filas_prueba().copy()
    filas[0] = filas[PRUEBA_POR_CLASE]  # la clase 0 pierde un clip y la 1 gana otro

    def lector_desequilibrado(_raiz):
        return LadoPruebaFalso(filas, "e" * 64)

    with pytest.raises(PruebaInesperada, match="por clase"):
        _cargar(raiz_prueba, lector_desequilibrado)


def test_corpus_incoherente(raiz_prueba: Path, lector_prueba_falso) -> None:
    candidato = leer(raiz_prueba)
    vocabulario = raiz_prueba / "config" / "corpus" / "vocabulario.csv"
    vocabulario.write_text(vocabulario.read_text().replace("2-Sint2", "2-Otra"), encoding="utf-8")

    with pytest.raises(ErrorAceptacion, match="vocabulario"):
        cargar_prueba(raiz_prueba, candidato, lector=lector_prueba_falso, total=TOTAL_PRUEBA, por_clase=PRUEBA_POR_CLASE)
