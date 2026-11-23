"""Sorteo por grabación entera (TT-22, research R-01; spec FR-001, FR-002, FR-003)."""

import hashlib
import random
from collections import defaultdict

import pytest

from signai_ml.particion import SEMILLA_PARTICION, Lado
from signai_ml.particion.errores import IndiceNoValido
from signai_ml.particion.sorteo import clave_sorteo, sortear

from .sinteticos import RECUENTOS_CORPUS, indice_sintetico

FILAS = indice_sintetico(RECUENTOS_CORPUS)


def _lados_por_grabacion(filas, asignacion: dict[str, Lado]) -> dict[tuple[str, str], set[Lado]]:
    lados: dict[tuple[str, str], set[Lado]] = defaultdict(set)
    for fila in filas:
        lados[(fila["clase"], fila["grabacion"])].add(asignacion[fila["clip_id"]])
    return lados


def test_clave_es_el_sha256_de_semilla_clase_y_grabacion() -> None:
    assert clave_sorteo(372, "1-Sint", "SINT_1_0") == hashlib.sha256(b"372|1-Sint|SINT_1_0").hexdigest()
    assert clave_sorteo(372, "1-Sint", "SINT_1_0") == (
        "54ce6f3f626f09ed99eac680731e3e3105b7b14b0ad80f38c6cab8eff78d330d"
    )


def test_cada_clip_recibe_un_lado_y_aparecen_los_tres() -> None:
    asignacion = sortear(FILAS, SEMILLA_PARTICION)
    assert set(asignacion) == {f["clip_id"] for f in FILAS}
    assert set(asignacion.values()) == set(Lado)


def test_todos_los_clips_de_una_grabacion_van_al_mismo_lado() -> None:
    for grabacion, lados in _lados_por_grabacion(FILAS, sortear(FILAS, SEMILLA_PARTICION)).items():
        assert len(lados) == 1, grabacion


def test_cada_clase_tiene_una_grabacion_en_cada_lado() -> None:
    por_clase: dict[str, list[Lado]] = defaultdict(list)
    for (clase, _), lados in _lados_por_grabacion(FILAS, sortear(FILAS, SEMILLA_PARTICION)).items():
        por_clase[clase].extend(lados)
    assert len(por_clase) == 100
    for clase, lados in por_clase.items():
        assert sorted(lados) == sorted(Lado), clase


def test_orden_por_clave_prueba_pliegue_a_pliegue_b() -> None:
    asignacion = sortear(FILAS, SEMILLA_PARTICION)
    grabaciones = sorted(
        {f["grabacion"] for f in FILAS if f["clase"] == "5-Sint"},
        key=lambda g: hashlib.sha256(f"{SEMILLA_PARTICION}|5-Sint|{g}".encode()).hexdigest(),
    )
    lado_de = {f["grabacion"]: asignacion[f["clip_id"]] for f in FILAS if f["clase"] == "5-Sint"}
    assert [lado_de[g] for g in grabaciones] == [Lado.PRUEBA, Lado.PLIEGUE_A, Lado.PLIEGUE_B]


def test_vector_dorado_con_cuatro_grabaciones() -> None:
    # Si cambia el algoritmo, cambia este reparto y la partición fijada dejaría de reproducirse.
    # La cuarta grabación va al pliegue A: más de 3 se reparten alternando A y B (R-01).
    filas = [
        {"clip_id": f"1-Sint__G{g}__rep{r:02d}", "clase": "1-Sint", "grabacion": f"G{g}"}
        for g in range(4) for r in range(1, 3)
    ]
    asignacion = sortear(filas, 372)
    lado_de = {f["grabacion"]: asignacion[f["clip_id"]] for f in filas}
    assert lado_de == {"G3": Lado.PRUEBA, "G1": Lado.PLIEGUE_A, "G2": Lado.PLIEGUE_B, "G0": Lado.PLIEGUE_A}


def test_determinista_e_independiente_del_orden_de_las_filas() -> None:
    barajadas = list(FILAS)
    random.Random(1).shuffle(barajadas)
    assert sortear(barajadas, SEMILLA_PARTICION) == sortear(FILAS, SEMILLA_PARTICION)


def test_solo_dependen_la_clase_y_la_grabacion() -> None:
    # Ni la repetición, ni la duración, ni el md5 intervienen; tampoco quién seña (FR-003).
    alteradas = [
        {**f, "repeticion": "99", "n_fotogramas_60": "7", "n_fotogramas_30": "3", "md5_clip": "0" * 32}
        for f in FILAS
    ]
    assert sortear(alteradas, SEMILLA_PARTICION) == sortear(FILAS, SEMILLA_PARTICION)


def test_el_mismo_nombre_de_grabacion_en_dos_clases_son_dos_grabaciones() -> None:
    assert clave_sorteo(372, "1-Sint", "IMG_1") != clave_sorteo(372, "2-Sint", "IMG_1")


def test_otra_semilla_da_otro_reparto() -> None:
    assert sortear(FILAS, SEMILLA_PARTICION) != sortear(FILAS, SEMILLA_PARTICION + 1)


def test_una_clase_con_dos_grabaciones_se_rechaza() -> None:
    filas = indice_sintetico(RECUENTOS_CORPUS, grabaciones={"7-Sint": 2})
    with pytest.raises(IndiceNoValido, match=r"7-Sint.*2 grabaciones"):
        sortear(filas, SEMILLA_PARTICION)
