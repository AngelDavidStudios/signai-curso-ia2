"""Humo del entorno de preprocesamiento: macOS arm64 nativo, sin contenedores (FR-009; R-10)."""

import platform

import signai_ml


def test_el_paquete_se_importa() -> None:
    assert signai_ml.__version__


def test_corre_nativo_en_macos_arm64() -> None:
    assert platform.system() == "Darwin"
    assert platform.machine() == "arm64"
