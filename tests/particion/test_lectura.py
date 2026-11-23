"""Lectura de la partición versión 2 por consumidor (TT-22; contracts/lectura.md L-1a, L-1b, L-2)."""

import json
from pathlib import Path

import numpy as np
import pytest

from signai_ml import particion
from signai_ml.particion import lectura
from signai_ml.particion.errores import ParticionAusente, ParticionNoValida
from signai_ml.particion.lectura import para_entrenamiento_final, para_evaluacion, para_seleccion

from .sinteticos import escribir_particion_version_1, leer_csv_particion

LECTORES = [para_seleccion, para_entrenamiento_final, para_evaluacion]


def _huella(raiz: Path) -> str:
    return json.loads((raiz / "datos" / "particion" / "manifiesto.json").read_text(encoding="utf-8"))["huella"]


def _filas(raiz: Path, lado: str) -> np.ndarray:
    return np.array([int(f["fila"]) for f in leer_csv_particion(raiz) if f["lado"] == lado], dtype=np.int64)


def _ordenado_int64_solo_lectura(filas: np.ndarray) -> bool:
    return filas.dtype == np.int64 and bool(np.all(np.diff(filas) > 0)) and not filas.flags.writeable


# ── T003 · constantes de la versión 2 ──


def test_constantes_de_la_version_2() -> None:
    assert particion.VERSION_PARTICION == "signai-ml/particion/2"
    assert particion.SEMILLA_PARTICION == 372
    assert [lado.value for lado in particion.Lado] == ["prueba", "pliegue_a", "pliegue_b"]


# ── L-1a · selección ──


def test_seleccion_dos_rotaciones_sin_la_prueba(raiz_con_particion: Path) -> None:
    rotaciones = para_seleccion(raiz_con_particion)
    a, b = _filas(raiz_con_particion, "pliegue_a"), _filas(raiz_con_particion, "pliegue_b")
    assert np.array_equal(rotaciones.a.entrenamiento, a) and np.array_equal(rotaciones.a.validacion, b)
    assert np.array_equal(rotaciones.b.entrenamiento, b) and np.array_equal(rotaciones.b.validacion, a)
    for rotacion in (rotaciones.a, rotaciones.b):
        assert _ordenado_int64_solo_lectura(rotacion.entrenamiento)
        assert _ordenado_int64_solo_lectura(rotacion.validacion)
    prueba = set(_filas(raiz_con_particion, "prueba").tolist())
    assert prueba.isdisjoint(a.tolist()) and prueba.isdisjoint(b.tolist())
    assert rotaciones.huella == _huella(raiz_con_particion)
    assert not hasattr(rotaciones, "prueba"), "la selección no tiene a mano la prueba"


# ── L-1b · entrenamiento final ──


def test_final_es_la_union_de_los_pliegues_sin_validacion(raiz_con_particion: Path) -> None:
    final = para_entrenamiento_final(raiz_con_particion)
    union = np.sort(np.concatenate([_filas(raiz_con_particion, "pliegue_a"), _filas(raiz_con_particion, "pliegue_b")]))
    assert np.array_equal(final.entrenamiento, union) and _ordenado_int64_solo_lectura(final.entrenamiento)
    assert final.huella == _huella(raiz_con_particion)
    assert not hasattr(final, "validacion") and not hasattr(final, "prueba")


# ── L-2 · evaluación ──


def test_evaluacion_solo_la_prueba(raiz_con_particion: Path) -> None:
    lado = para_evaluacion(raiz_con_particion)
    assert np.array_equal(lado.prueba, _filas(raiz_con_particion, "prueba"))
    assert _ordenado_int64_solo_lectura(lado.prueba)
    assert lado.huella == _huella(raiz_con_particion)
    assert not hasattr(lado, "entrenamiento") and not hasattr(lado, "validacion")


# ── Garantías ──


def test_tres_lados_disjuntos_que_cubren_el_corpus(raiz_con_particion: Path) -> None:
    final = para_entrenamiento_final(raiz_con_particion).entrenamiento
    prueba = para_evaluacion(raiz_con_particion).prueba
    todas = np.concatenate([final, prueba])
    assert len(todas) == len(set(todas.tolist())) == 8996
    assert set(todas.tolist()) == set(range(8996))


def test_ninguna_grabacion_en_dos_lados(raiz_con_particion: Path) -> None:
    lados: dict[tuple[str, str], set[str]] = {}
    for fila in leer_csv_particion(raiz_con_particion):
        lados.setdefault((fila["clase"], fila["grabacion"]), set()).add(fila["lado"])
    assert len(lados) == 300 and all(len(v) == 1 for v in lados.values())


def test_la_lectura_de_la_version_1_ya_no_existe() -> None:
    assert not hasattr(lectura, "para_entrenamiento")
    assert not hasattr(lectura, "LadosEntrenamiento")


@pytest.mark.parametrize("leer", LECTORES)
def test_la_version_1_se_rechaza_con_la_orden_de_migrar(raiz_sintetica: Path, leer) -> None:
    escribir_particion_version_1(raiz_sintetica)
    with pytest.raises(ParticionNoValida, match="signai-particion generar"):
        leer(raiz_sintetica)


@pytest.mark.parametrize("leer", LECTORES)
def test_sin_particion_ni_puntero_pide_generarla(raiz_sintetica: Path, leer) -> None:
    with pytest.raises(ParticionAusente, match="signai-particion generar"):
        leer(raiz_sintetica)


@pytest.mark.parametrize("leer", LECTORES)
def test_con_puntero_sin_datos_pide_restaurarla(raiz_sintetica: Path, leer) -> None:
    (raiz_sintetica / "datos" / "particion.dvc").write_text("outs:\n- path: particion\n", encoding="utf-8")
    with pytest.raises(ParticionAusente, match=r"uv run dvc checkout datos/particion\.dvc"):
        leer(raiz_sintetica)


@pytest.mark.parametrize("leer", LECTORES)
def test_con_la_ligadura_rota_no_devuelve_nada(raiz_con_particion: Path, leer) -> None:
    ruta = raiz_con_particion / "datos" / "ventanas" / "manifiesto.json"
    manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
    manifiesto["sha256_ventanas"] = "0" * 64
    ruta.write_text(json.dumps(manifiesto), encoding="utf-8")
    with pytest.raises(ParticionNoValida):
        leer(raiz_con_particion)
