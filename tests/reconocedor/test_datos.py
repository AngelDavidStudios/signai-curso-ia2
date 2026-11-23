"""Datos por rotación y del modelo final (TT-22, research R-02, R-06; tasks T022)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from signai_ml.particion.errores import ParticionAusente
from signai_ml.reconocedor.datos import cargar_final, cargar_rotacion
from signai_ml.reconocedor.errores import ErrorReconocedor

from .conftest import HUELLA_PARTICION, pliegues_sinteticos


def _ventanas(raiz: Path) -> np.ndarray:
    return np.load(raiz / "datos" / "ventanas" / "ventanas.npy")


def test_rotacion_a_entrena_con_a_y_valida_con_b(raiz_sintetica: Path, lector_seleccion) -> None:
    datos = cargar_rotacion(raiz_sintetica, "a", lector_seleccion)
    a, b = pliegues_sinteticos()
    ventanas = _ventanas(raiz_sintetica)
    assert np.array_equal(datos.x_entrenamiento.numpy(), ventanas[a])
    assert np.array_equal(datos.x_validacion.numpy(), ventanas[b])
    assert datos.y_validacion.tolist() == [c for c in range(4) for _ in range(4)]
    assert not any("prueba" in nombre for nombre in dir(datos))


def test_rotacion_b_es_la_inversa(raiz_sintetica: Path, lector_seleccion) -> None:
    datos = cargar_rotacion(raiz_sintetica, "b", lector_seleccion)
    a, b = pliegues_sinteticos()
    ventanas = _ventanas(raiz_sintetica)
    assert np.array_equal(datos.x_entrenamiento.numpy(), ventanas[b])
    assert np.array_equal(datos.x_validacion.numpy(), ventanas[a])


def test_rotacion_desconocida(raiz_sintetica: Path, lector_seleccion) -> None:
    with pytest.raises(ErrorReconocedor, match="rotación"):
        cargar_rotacion(raiz_sintetica, "c", lector_seleccion)


def test_final_entrena_con_los_dos_pliegues_sin_validacion(raiz_sintetica: Path, lector_final) -> None:
    datos = cargar_final(raiz_sintetica, lector_final)
    assert datos.x_entrenamiento.shape == (32, 75, 96)
    assert np.array_equal(datos.x_entrenamiento.numpy(), _ventanas(raiz_sintetica))
    assert not hasattr(datos, "x_validacion") and not any("prueba" in n for n in dir(datos))


def test_clases_glosas_y_huellas(raiz_sintetica: Path, lector_seleccion, lector_final) -> None:
    ventanas = raiz_sintetica / "datos" / "ventanas"
    esperadas = {
        "corpus.sha256_ventanas": json.loads((ventanas / "manifiesto.json").read_text())["sha256_ventanas"],
        "corpus.sha256_indice": hashlib.sha256((ventanas / "indice.csv").read_bytes()).hexdigest(),
        "particion.huella": HUELLA_PARTICION,
    }
    for datos in (cargar_rotacion(raiz_sintetica, "a", lector_seleccion), cargar_final(raiz_sintetica, lector_final)):
        assert datos.clases == ("1-Sint1", "2-Sint2", "3-Sint3", "4-Sint4")
        assert datos.glosas == ("glosa1", "glosa2", "glosa3", "glosa4")
        assert datos.huellas == esperadas


def test_propaga_el_error_de_la_particion(raiz_sintetica: Path) -> None:
    def lector_roto(_raiz: Path):
        raise ParticionAusente("No hay partición.")

    with pytest.raises(ParticionAusente):
        cargar_rotacion(raiz_sintetica, "a", lector_roto)
    with pytest.raises(ParticionAusente):
        cargar_final(raiz_sintetica, lector_roto)


def test_vocabulario_que_no_corresponde_al_indice(raiz_sintetica: Path, lector_seleccion) -> None:
    vocabulario = raiz_sintetica / "config" / "corpus" / "vocabulario.csv"
    vocabulario.write_text(vocabulario.read_text().replace("2-Sint2", "2-Otra"), encoding="utf-8")
    with pytest.raises(ErrorReconocedor, match="vocabulario"):
        cargar_rotacion(raiz_sintetica, "a", lector_seleccion)
