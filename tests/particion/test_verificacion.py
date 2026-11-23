"""Ligadura con el corpus y verificación completa de la partición versión 2 (TT-22; contracts/lectura.md)."""

import json
from pathlib import Path

import numpy as np
import pytest

from signai_ml.particion import Lado
from signai_ml.particion.errores import ParticionAusente, ParticionNoValida
from signai_ml.particion.verificacion import comprobar_ligadura, verificar

from .sinteticos import (
    RECUENTOS_CORPUS,
    escribir_indice,
    escribir_particion_version_1,
    grabaciones_por_lado,
    indice_sintetico,
    intercambiar_grabaciones,
    leer_csv_particion,
    reescribir_particion,
)

RESTAURAR = r"uv run dvc checkout datos/ventanas\.dvc datos/particion\.dvc"
MIGRAR = r"la partición es la versión 1 \(por clip\); genera la versión 2: `signai-particion generar`"
NOMBRES = ("versión", "huella", "corpus de origen", "filas = índice", "grabaciones y disjunción", "re-sorteo")


def _ventanas(raiz: Path) -> Path:
    return raiz / "datos" / "ventanas"


def _manifiesto_particion(raiz: Path) -> Path:
    return raiz / "datos" / "particion" / "manifiesto.json"


def _editar_json(ruta: Path, clave: str, valor: object) -> None:
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    datos[clave] = valor
    ruta.write_text(json.dumps(datos), encoding="utf-8")


# ── comprobar_ligadura ──


def test_ligadura_bien(raiz_con_particion: Path) -> None:
    leida = comprobar_ligadura(raiz_con_particion)
    assert len(leida.asignaciones) == 8996


def test_ligadura_sin_particion(raiz_sintetica: Path) -> None:
    with pytest.raises(ParticionAusente):
        comprobar_ligadura(raiz_sintetica)


def test_ligadura_rechaza_la_version_1_con_la_orden_de_migrar(raiz_sintetica: Path) -> None:
    escribir_particion_version_1(raiz_sintetica)
    with pytest.raises(ParticionNoValida, match=MIGRAR):
        comprobar_ligadura(raiz_sintetica)


def test_ligadura_huella_de_particion_csv(raiz_con_particion: Path) -> None:
    ruta = raiz_con_particion / "datos" / "particion" / "particion.csv"
    ruta.write_text(ruta.read_text(encoding="utf-8").replace(",prueba\n", ",pliegue_a\n", 1), encoding="utf-8")
    with pytest.raises(ParticionNoValida, match=rf"(?s)particion\.csv.*{RESTAURAR}"):
        comprobar_ligadura(raiz_con_particion)


def test_ligadura_huella_de_las_ventanas(raiz_con_particion: Path) -> None:
    _editar_json(_ventanas(raiz_con_particion) / "manifiesto.json", "sha256_ventanas", "0" * 64)
    with pytest.raises(ParticionNoValida, match=rf"(?s)ventanas.*{RESTAURAR}"):
        comprobar_ligadura(raiz_con_particion)


def test_ligadura_huella_del_indice(raiz_con_particion: Path) -> None:
    filas = indice_sintetico(RECUENTOS_CORPUS)
    filas[0]["glosa"] = "otra"
    escribir_indice(_ventanas(raiz_con_particion), filas)
    with pytest.raises(ParticionNoValida, match=rf"(?s)indice\.csv.*{RESTAURAR}"):
        comprobar_ligadura(raiz_con_particion)


def test_ligadura_fila_con_otro_clip(raiz_con_particion: Path) -> None:
    filas = leer_csv_particion(raiz_con_particion)
    filas[0]["clip_id"], filas[1]["clip_id"] = filas[1]["clip_id"], filas[0]["clip_id"]
    reescribir_particion(raiz_con_particion, filas)
    with pytest.raises(ParticionNoValida, match=rf"(?s)fila 0.*{RESTAURAR}"):
        comprobar_ligadura(raiz_con_particion)


def test_ligadura_fila_con_otra_grabacion(raiz_con_particion: Path) -> None:
    filas = leer_csv_particion(raiz_con_particion)
    filas[0]["grabacion"] = "OTRA"
    reescribir_particion(raiz_con_particion, filas)
    with pytest.raises(ParticionNoValida, match=rf"(?s)fila 0.*OTRA.*{RESTAURAR}"):
        comprobar_ligadura(raiz_con_particion)


# ── verificar ──


def test_verificar_bien(raiz_con_particion: Path) -> None:
    informe = verificar(raiz_con_particion)
    assert informe.ok
    assert [c.nombre for c in informe.comprobaciones] == list(NOMBRES)
    manifiesto = json.loads(_manifiesto_particion(raiz_con_particion).read_text(encoding="utf-8"))
    assert informe.huella == manifiesto["huella"]
    assert informe.totales == {Lado(lado): n for lado, n in manifiesto["recuentos"]["total"].items()}
    assert sum(informe.totales.values()) == 8996
    assert "re-sorteo idéntico (8996/8996)" in informe.comprobacion("re-sorteo").detalle


def test_verificar_la_version_1_falla_en_la_version(raiz_sintetica: Path) -> None:
    escribir_particion_version_1(raiz_sintetica)
    with pytest.raises(ParticionNoValida, match=MIGRAR):
        verificar(raiz_sintetica)


def test_verificar_recalcula_la_huella_real_de_las_ventanas(raiz_con_particion: Path) -> None:
    with (_ventanas(raiz_con_particion) / "ventanas.npy").open("wb") as f:
        np.save(f, np.ones((8996, 1), dtype=np.float32), allow_pickle=False)
    informe = verificar(raiz_con_particion)
    assert not informe.ok
    assert not informe.comprobacion("corpus de origen").ok


def test_verificar_detecta_una_grabacion_partida(raiz_con_particion: Path) -> None:
    filas = leer_csv_particion(raiz_con_particion)
    next(f for f in filas if f["clase"] == "2-Sint" and f["lado"] == "prueba")["lado"] = "pliegue_a"
    reescribir_particion(raiz_con_particion, filas)
    comprobacion = verificar(raiz_con_particion).comprobacion("grabaciones y disjunción")
    assert not comprobacion.ok and "2-Sint" in comprobacion.detalle


def test_verificar_detecta_una_clase_sin_grabacion_de_prueba(raiz_con_particion: Path) -> None:
    filas = leer_csv_particion(raiz_con_particion)
    for fila in filas:
        if fila["clase"] == "4-Sint" and fila["lado"] == "prueba":
            fila["lado"] = "pliegue_b"
    reescribir_particion(raiz_con_particion, filas)
    comprobacion = verificar(raiz_con_particion).comprobacion("grabaciones y disjunción")
    assert not comprobacion.ok and "4-Sint" in comprobacion.detalle


def test_verificar_detecta_un_intercambio_de_grabaciones(raiz_con_particion: Path) -> None:
    # Cada grabación sigue entera y cada clase tiene una por lado: solo el re-sorteo lo detecta.
    filas = intercambiar_grabaciones(leer_csv_particion(raiz_con_particion), "3-Sint", "prueba", "pliegue_a")
    assert all(len(g) == 1 for g in grabaciones_por_lado(filas, "3-Sint").values())
    reescribir_particion(raiz_con_particion, filas)
    informe = verificar(raiz_con_particion)
    assert informe.comprobacion("huella").ok
    assert not informe.comprobacion("re-sorteo").ok


def test_verificar_no_escribe_nada(raiz_con_particion: Path) -> None:
    datos = raiz_con_particion / "datos"
    antes = {r: (r.stat().st_mtime_ns, r.read_bytes()) for r in datos.rglob("*") if r.is_file()}
    verificar(raiz_con_particion)
    despues = {r: (r.stat().st_mtime_ns, r.read_bytes()) for r in datos.rglob("*") if r.is_file()}
    assert despues == antes


def test_verificar_sin_particion(raiz_sintetica: Path) -> None:
    with pytest.raises(ParticionAusente):
        verificar(raiz_sintetica)


def test_verificar_con_el_indice_ilegible_sigue_informando(raiz_con_particion: Path) -> None:
    (_ventanas(raiz_con_particion) / "indice.csv").unlink()
    informe = verificar(raiz_con_particion)
    assert not informe.comprobacion("corpus de origen").ok
    assert "indice.csv" in informe.comprobacion("filas = índice").detalle
    assert informe.comprobacion("grabaciones y disjunción").ok
    assert informe.comprobacion("re-sorteo").ok
