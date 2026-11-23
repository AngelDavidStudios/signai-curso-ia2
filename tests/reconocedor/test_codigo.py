"""Guarda de árbol limpio, común a TT-14 y TT-15 (`signai_ml.codigo`; research R-02 de TT-15)."""

import subprocess
from pathlib import Path

import pytest

from signai_ml.codigo import codigo_permitido, estado_codigo


class ArbolSucioDePrueba(Exception):
    pass


def _git(raiz: Path, *argumentos: str) -> None:
    subprocess.run(["git", "-C", str(raiz), *argumentos], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    raiz = tmp_path / "repo"
    raiz.mkdir()
    _git(raiz, "init", "-q")
    _git(raiz, "config", "user.email", "prueba@example.invalid")
    _git(raiz, "config", "user.name", "Prueba")
    (raiz / "a.txt").write_text("a", encoding="utf-8")
    _git(raiz, "add", "a.txt")
    _git(raiz, "commit", "-q", "-m", "inicio")
    return raiz


def test_repositorio_limpio(repo: Path) -> None:
    commit, sucio = estado_codigo(repo)

    assert len(commit) == 40
    assert sucio is False


def test_repositorio_sucio(repo: Path) -> None:
    (repo / "b.txt").write_text("b", encoding="utf-8")

    assert estado_codigo(repo)[1] is True


def test_fuera_de_un_repositorio(tmp_path: Path) -> None:
    assert estado_codigo(tmp_path) == ("desconocido", True)


def test_limpio_se_permite(repo: Path) -> None:
    commit, sucio = codigo_permitido(repo, False, ArbolSucioDePrueba, "Vuelve a lanzar.")

    assert commit == estado_codigo(repo)[0]
    assert sucio is False


def test_sucio_lanza_el_error_que_se_pide(repo: Path) -> None:
    (repo / "a.txt").write_text("cambio", encoding="utf-8")

    with pytest.raises(ArbolSucioDePrueba, match="sin confirmar.*Vuelve a lanzar"):
        codigo_permitido(repo, False, ArbolSucioDePrueba, "Vuelve a lanzar.")


def test_sucio_se_deja_pasar_marcado_si_se_permite(repo: Path) -> None:
    (repo / "a.txt").write_text("cambio", encoding="utf-8")

    assert codigo_permitido(repo, True, ArbolSucioDePrueba, "Vuelve a lanzar.")[1] is True
