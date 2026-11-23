"""Escritura y lectura de la partición versión 2 (TT-22; data-model «Partición versión 2»)."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from signai_ml.particion import SEMILLA_PARTICION, VERSION_PARTICION, Lado
from signai_ml.particion import almacen
from signai_ml.particion.almacen import escribir_particion, leer_particion, ruta_particion
from signai_ml.particion.errores import ErrorParticion
from signai_ml.particion.indice import leer_corpus
from signai_ml.particion.sorteo import sortear

from .sinteticos import escribir_particion_version_1

REGLAS = {
    "unidad": "(clase, grabacion)",
    "orden_sorteo": "sha256(semilla|clase|grabacion)",
    "reparto": "1.ª prueba, 2.ª pliegue_a, 3.ª pliegue_b; más de 3: alternando pliegue_a y pliegue_b",
}


@pytest.fixture
def escrita(raiz_sintetica: Path) -> tuple[Path, dict]:
    corpus = leer_corpus(raiz_sintetica)
    base = ruta_particion(raiz_sintetica)
    manifiesto = escribir_particion(base, sortear(corpus.filas, SEMILLA_PARTICION), corpus, SEMILLA_PARTICION)
    return base, manifiesto


def test_ruta_de_la_particion(raiz_sintetica: Path) -> None:
    assert ruta_particion(raiz_sintetica) == raiz_sintetica / "datos" / "particion"


def test_particion_csv_canonico_con_grabacion(escrita: tuple[Path, dict], raiz_sintetica: Path) -> None:
    base, _ = escrita
    contenido = (base / "particion.csv").read_bytes()
    lineas = contenido.decode("utf-8").split("\n")
    assert lineas[0] == "fila,clip_id,clase,grabacion,lado"
    assert b"\r" not in contenido and contenido.endswith(b"\n")
    filas = [linea.split(",") for linea in lineas[1:-1]]
    assert [int(f[0]) for f in filas] == list(range(8996))
    indice = leer_corpus(raiz_sintetica).filas
    assert [(f[1], f[2], f[3]) for f in filas] == [(i["clip_id"], i["clase"], i["grabacion"]) for i in indice]
    assert {f[4] for f in filas} == {"prueba", "pliegue_a", "pliegue_b"}


def test_manifiesto_version_2(escrita: tuple[Path, dict], raiz_sintetica: Path) -> None:
    base, devuelto = escrita
    manifiesto = json.loads((base / "manifiesto.json").read_text(encoding="utf-8"))
    corpus = leer_corpus(raiz_sintetica)
    assert manifiesto == devuelto
    assert manifiesto["version_particion"] == VERSION_PARTICION == "signai-ml/particion/2"
    assert manifiesto["semilla"] == SEMILLA_PARTICION == 372
    assert manifiesto["reglas"] == REGLAS
    assert manifiesto["origen"] == {
        "sha256_ventanas": corpus.sha256_ventanas, "sha256_indice": corpus.sha256_indice, "ventanas": 8996,
    }
    total = manifiesto["recuentos"]["total"]
    assert set(total) == {"prueba", "pliegue_a", "pliegue_b"} and sum(total.values()) == 8996
    por_clase = manifiesto["recuentos"]["por_clase"]
    assert len(por_clase) == 100
    for clase, lados in por_clase.items():
        assert set(lados) == {"prueba", "pliegue_a", "pliegue_b"}, clase
        assert len(lados["prueba"]["grabaciones"]) == 1, clase
        assert sum(lado["clips"] for lado in lados.values()) in (89, 90), clase
    assert por_clase["1-Sint"]["prueba"]["clips"] == 30
    assert datetime.fromisoformat(manifiesto["fecha"]).tzinfo is not None
    assert manifiesto["huella"] == hashlib.sha256((base / "particion.csv").read_bytes()).hexdigest()


def test_informe_por_grabacion(escrita: tuple[Path, dict]) -> None:
    base, manifiesto = escrita
    informe = (base / "informe.md").read_text(encoding="utf-8")
    assert "signai-ml/particion/2" in informe and "372" in informe
    lados = manifiesto["recuentos"]["por_clase"]["1-Sint"]
    celdas = " | ".join(f"{lados[l]['grabaciones'][0]} ({lados[l]['clips']})" for l in ("prueba", "pliegue_a", "pliegue_b"))
    assert f"| 1-Sint | {celdas} |" in informe


def test_leer_devuelve_lo_escrito_con_la_grabacion(escrita: tuple[Path, dict], raiz_sintetica: Path) -> None:
    base, manifiesto = escrita
    leida = leer_particion(base)
    assert leida.manifiesto == manifiesto
    assert len(leida.asignaciones) == 8996
    primera = leida.asignaciones[0]
    assert (primera.fila, primera.clip_id, primera.clase, primera.grabacion) == (
        0, "1-Sint__SINT_1_0__rep01", "1-Sint", "SINT_1_0",
    )
    assert isinstance(primera.lado, Lado)
    esperado = sortear(leer_corpus(raiz_sintetica).filas, SEMILLA_PARTICION)
    assert {a.clip_id: a.lado for a in leida.asignaciones} == esperado


def test_escritura_atomica(raiz_sintetica: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = leer_corpus(raiz_sintetica)
    base = ruta_particion(raiz_sintetica)

    def falla(_: dict) -> str:
        raise RuntimeError("corte simulado")

    monkeypatch.setattr(almacen, "informe_markdown", falla)
    with pytest.raises(RuntimeError):
        escribir_particion(base, sortear(corpus.filas, SEMILLA_PARTICION), corpus, SEMILLA_PARTICION)
    assert not base.exists()
    assert not list(base.parent.glob(".particion*")), "sin temporales olvidados"


def test_no_sobrescribe_sin_permiso(escrita: tuple[Path, dict], raiz_sintetica: Path) -> None:
    base, _ = escrita
    corpus = leer_corpus(raiz_sintetica)
    with pytest.raises(ErrorParticion, match="ya existe"):
        escribir_particion(base, sortear(corpus.filas, SEMILLA_PARTICION), corpus, SEMILLA_PARTICION)


def test_sustituye_la_version_1_de_golpe(raiz_sintetica: Path) -> None:
    base = escribir_particion_version_1(raiz_sintetica)
    corpus = leer_corpus(raiz_sintetica)
    manifiesto = escribir_particion(
        base, sortear(corpus.filas, SEMILLA_PARTICION), corpus, SEMILLA_PARTICION, sustituir=True
    )
    assert manifiesto["version_particion"] == "signai-ml/particion/2"
    assert json.loads((base / "manifiesto.json").read_text(encoding="utf-8")) == manifiesto
    assert "grabacion" in (base / "particion.csv").read_text(encoding="utf-8").split("\n")[0]
    assert not list(base.parent.glob(".particion*")), "sin temporales ni la versión 1 olvidada"
