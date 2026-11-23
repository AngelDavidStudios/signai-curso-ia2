"""Aprobación del lote de validación (FR-021, SC-005; research R-13)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from signai_ml.preproceso.almacen import Versiones
from signai_ml.preproceso.aprobacion import exigir, invalidar, leer, registrar, vigente
from signai_ml.preproceso.errores import ErrorUso, FaltaAprobacion

V = Versiones("signai-ml/preproceso/1", "signai-ml/keypoints/1", "1.0.0", "1.0.0", "27.2.0")


def test_aprobar_registra_decision_fecha_versiones_semilla_e_informe(tmp_path: Path) -> None:
    r = registrar(tmp_path, "aprobado", "", V, semilla_lote=213, sha256_informe="s" * 64)
    datos = json.loads((tmp_path / "aprobacion.json").read_text())
    assert datos == r
    assert datos["decision"] == "aprobado" and datos["semilla_lote"] == 213
    assert datos["version_cadena"] == V.version_cadena and datos["macos"] == "27.2.0"
    assert datos["sha256_informe"] == "s" * 64
    assert "T" in datos["fecha"] and ("+" in datos["fecha"][10:] or "-" in datos["fecha"][10:])


def test_rechazar_exige_nota(tmp_path: Path) -> None:
    with pytest.raises(ErrorUso, match="nota"):
        registrar(tmp_path, "rechazado", "", V, 213, "s" * 64)
    registrar(tmp_path, "rechazado", "la mano derecha sale en el bloque izquierdo", V, 213, "s" * 64)
    assert leer(tmp_path)["decision"] == "rechazado"


def test_una_decision_nueva_guarda_la_anterior(tmp_path: Path) -> None:
    registrar(tmp_path, "rechazado", "mal", V, 213, "s" * 64)
    registrar(tmp_path, "aprobado", "", V, 213, "s" * 64)
    anteriores = list((tmp_path / "aprobaciones-anteriores").glob("*.json"))
    assert len(anteriores) == 1 and json.loads(anteriores[0].read_text())["decision"] == "rechazado"


def test_vigente_y_exigir(tmp_path: Path) -> None:
    with pytest.raises(FaltaAprobacion, match="aprobar-lote"):
        exigir(tmp_path, V, "s" * 64)
    registrar(tmp_path, "aprobado", "", V, 213, "s" * 64)
    assert vigente(tmp_path, V, "s" * 64)
    exigir(tmp_path, V, "s" * 64)
    assert not vigente(tmp_path, replace(V, version_cadena="signai-ml/preproceso/2"), "s" * 64)
    with pytest.raises(FaltaAprobacion, match="versión"):
        exigir(tmp_path, replace(V, version_cadena="signai-ml/preproceso/2"), "s" * 64)
    with pytest.raises(FaltaAprobacion, match="informe"):
        exigir(tmp_path, V, "t" * 64)


def test_un_rechazo_no_habilita(tmp_path: Path) -> None:
    registrar(tmp_path, "rechazado", "mal", V, 213, "s" * 64)
    with pytest.raises(FaltaAprobacion, match="rechazado"):
        exigir(tmp_path, V, "s" * 64)


def test_invalidar_mueve_la_decision_a_anteriores(tmp_path: Path) -> None:
    assert not invalidar(tmp_path)
    registrar(tmp_path, "aprobado", "", V, 213, "s" * 64)
    assert invalidar(tmp_path)
    assert leer(tmp_path) is None
