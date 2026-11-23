"""CLI `signai-particion` con la partición versión 2 (TT-22; contracts/cli.md P-1', P-2')."""

import json
from pathlib import Path

import pytest

from signai_ml.particion.cli import main

from .sinteticos import (
    RECUENTOS_CORPUS,
    escribir_corpus,
    escribir_indice,
    escribir_particion_version_1,
    indice_sintetico,
    intercambiar_grabaciones,
    leer_csv_particion,
    reescribir_particion,
)


def _particion(raiz: Path) -> Path:
    return raiz / "datos" / "particion"


def _manifiesto(raiz: Path) -> dict:
    return json.loads((_particion(raiz) / "manifiesto.json").read_text(encoding="utf-8"))


def _contenido(raiz: Path) -> dict[str, bytes]:
    return {r.name: r.read_bytes() for r in sorted(_particion(raiz).iterdir())}


def _totales(manifiesto: dict) -> str:
    t = manifiesto["recuentos"]["total"]
    miles = lambda n: f"{n:,}".replace(",", ".")  # noqa: E731
    return f"{miles(t['prueba'])} prueba · {miles(t['pliegue_a'])} pliegue A · {miles(t['pliegue_b'])} pliegue B"


# ── P-1' · generar ──


def test_generar_reparte_por_grabacion_y_explica_como_versionar(
    raiz_sintetica: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["generar"], raiz=raiz_sintetica) == 0
    salida = capsys.readouterr().out
    manifiesto = _manifiesto(raiz_sintetica)
    assert manifiesto["version_particion"] == "signai-ml/particion/2"
    assert _totales(manifiesto) in salida
    assert "8.996 ventanas, 100 clases, 300 grabaciones" in salida
    assert manifiesto["huella"] in salida
    for orden in ("uv run dvc add datos/particion", "uv run dvc push", "git add datos/particion.dvc",
                  "uv run signai-copia actualizar"):
        assert orden in salida
    assert {r.name for r in _particion(raiz_sintetica).iterdir()} == {
        "particion.csv", "manifiesto.json", "informe.md",
    }


def test_generar_se_niega_con_una_clase_de_dos_grabaciones(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    raiz = escribir_corpus(tmp_path, indice_sintetico(RECUENTOS_CORPUS, grabaciones={"7-Sint": 2}))
    assert main(["generar"], raiz=raiz) == 1
    salida = capsys.readouterr().out
    assert "7-Sint" in salida and "2 grabaciones" in salida
    assert not _particion(raiz).exists()


def test_generar_con_un_indice_no_valido_no_escribe_nada(
    raiz_sintetica: Path, capsys: pytest.CaptureFixture
) -> None:
    filas = indice_sintetico(RECUENTOS_CORPUS)
    filas[1]["clip_id"] = filas[0]["clip_id"]
    escribir_indice(raiz_sintetica / "datos" / "ventanas", filas)
    assert main(["generar"], raiz=raiz_sintetica) == 1
    assert filas[0]["clip_id"] in capsys.readouterr().out
    assert not _particion(raiz_sintetica).exists()


def test_generar_sustituye_la_version_1_y_lo_dice(raiz_sintetica: Path, capsys: pytest.CaptureFixture) -> None:
    escribir_particion_version_1(raiz_sintetica)
    (raiz_sintetica / "datos" / "particion.dvc").write_text("outs:\n- path: particion\n", encoding="utf-8")
    assert main(["generar"], raiz=raiz_sintetica) == 0
    salida = capsys.readouterr().out
    assert "sustituye a la versión 1" in salida and "historial de DVC" in salida
    assert _manifiesto(raiz_sintetica)["version_particion"] == "signai-ml/particion/2"


def test_generar_se_niega_si_la_version_2_ya_esta_fijada(
    raiz_con_particion: Path, capsys: pytest.CaptureFixture
) -> None:
    antes = _contenido(raiz_con_particion)
    capsys.readouterr()
    assert main(["generar"], raiz=raiz_con_particion) == 1
    assert "La partición ya está fijada" in capsys.readouterr().out
    assert _contenido(raiz_con_particion) == antes


def test_generar_con_puntero_sin_datos_pide_restaurarla_antes(
    raiz_sintetica: Path, capsys: pytest.CaptureFixture
) -> None:
    # Sin los datos no se sabe qué versión es: hay que descargarla primero.
    (raiz_sintetica / "datos" / "particion.dvc").write_text("outs:\n- path: particion\n", encoding="utf-8")
    assert main(["generar"], raiz=raiz_sintetica) == 1
    assert "uv run dvc checkout datos/particion.dvc" in capsys.readouterr().out
    assert not _particion(raiz_sintetica).exists()


@pytest.mark.parametrize("opcion", [["--semilla", "7"], ["--forzar"]])
def test_no_hay_semilla_ni_forzado(raiz_sintetica: Path, opcion: list[str]) -> None:
    assert main(["generar", *opcion], raiz=raiz_sintetica) == 2
    assert not _particion(raiz_sintetica).exists()


# ── P-2' · verificar ──


def test_verificar_bien(raiz_con_particion: Path, capsys: pytest.CaptureFixture) -> None:
    manifiesto = _manifiesto(raiz_con_particion)
    capsys.readouterr()
    assert main(["verificar"], raiz=raiz_con_particion) == 0
    salida = capsys.readouterr().out
    assert salida.count("✓") == 6
    assert _totales(manifiesto) in salida
    assert manifiesto["huella"] in salida


def test_verificar_mal_dice_que_falla_y_como_restaurar(
    raiz_con_particion: Path, capsys: pytest.CaptureFixture
) -> None:
    filas = intercambiar_grabaciones(leer_csv_particion(raiz_con_particion), "3-Sint", "prueba", "pliegue_a")
    reescribir_particion(raiz_con_particion, filas)
    capsys.readouterr()
    assert main(["verificar"], raiz=raiz_con_particion) == 1
    salida = capsys.readouterr().out
    assert "✗" in salida and "re-sorteo" in salida
    assert "uv run dvc checkout datos/ventanas.dvc datos/particion.dvc" in salida


def test_verificar_la_version_1_pide_migrar(raiz_sintetica: Path, capsys: pytest.CaptureFixture) -> None:
    escribir_particion_version_1(raiz_sintetica)
    assert main(["verificar"], raiz=raiz_sintetica) == 1
    assert "signai-particion generar" in capsys.readouterr().out


def test_verificar_sin_particion(raiz_sintetica: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["verificar"], raiz=raiz_sintetica) == 1
    assert "signai-particion generar" in capsys.readouterr().out


def test_sin_orden_es_uso_incorrecto(raiz_sintetica: Path) -> None:
    assert main([], raiz=raiz_sintetica) == 2
