"""Precondición de la copia: nada pendiente de `dvc push` (research R-02).

El ejecutor se inyecta: ninguna prueba llama a DVC ni toca el remoto real.
"""

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from signai_ml.copia.dvc import EstadoDvcLocal
from signai_ml.copia.errores import FaltaHerramienta, RechazoDominio


class Ejecutor:
    def __init__(self, codigo: int = 0, stdout: str = "{}", stderr: str = "") -> None:
        self.codigo, self.stdout, self.stderr = codigo, stdout, stderr
        self.llamadas: list[tuple[Sequence[str], dict[str, object]]] = []

    def __call__(self, orden: Sequence[str], **opciones: object) -> subprocess.CompletedProcess[str]:
        self.llamadas.append((orden, opciones))
        return subprocess.CompletedProcess(orden, self.codigo, self.stdout, self.stderr)


def test_compara_con_el_remoto_desde_la_raiz_del_repositorio(tmp_path: Path) -> None:
    ejecutor = Ejecutor()
    EstadoDvcLocal(tmp_path, ejecutar=ejecutor).sin_subir()

    orden, opciones = ejecutor.llamadas[0]
    assert list(orden) == [sys.executable, "-m", "dvc", "status", "--cloud", "--json"]
    assert opciones["cwd"] == tmp_path


def test_en_sincronia_no_hay_nada_sin_subir(tmp_path: Path) -> None:
    assert EstadoDvcLocal(tmp_path, ejecutar=Ejecutor(stdout="{}")).sin_subir() is False


def test_con_objetos_nuevos_hay_algo_sin_subir(tmp_path: Path) -> None:
    pendiente = json.dumps({"datos/clips.dvc": [{"changed checksum": "datos/clips"}]})
    assert EstadoDvcLocal(tmp_path, ejecutar=Ejecutor(stdout=pendiente)).sin_subir() is True


def test_sin_dvc_falta_herramienta(tmp_path: Path) -> None:
    ejecutor = Ejecutor(codigo=1, stdout="", stderr="/usr/bin/python3: No module named dvc")
    with pytest.raises(FaltaHerramienta, match="dvc"):
        EstadoDvcLocal(tmp_path, ejecutar=ejecutor).sin_subir()


def test_un_error_de_dvc_es_rechazo_de_dominio(tmp_path: Path) -> None:
    ejecutor = Ejecutor(codigo=255, stdout="", stderr="ERROR: config file error: remote 'local' doesn't exist")
    with pytest.raises(RechazoDominio, match="remote 'local'"):
        EstadoDvcLocal(tmp_path, ejecutar=ejecutor).sin_subir()


# ── Comparación con un almacén restaurado (R-09) ───────────────────────────────────────────────


class EjecutorSecuencia:
    """Devuelve una respuesta por llamada y recuerda las órdenes."""

    def __init__(self, *respuestas: tuple[int, str, str]) -> None:
        self.respuestas = list(respuestas)
        self.ordenes: list[list[str]] = []

    def __call__(self, orden: Sequence[str], **_: object) -> subprocess.CompletedProcess[str]:
        self.ordenes.append(list(orden))
        codigo, stdout, stderr = self.respuestas.pop(0)
        return subprocess.CompletedProcess(orden, codigo, stdout, stderr)


def test_ausentes_usa_un_remoto_temporal_y_lo_retira(tmp_path: Path) -> None:
    almacen = tmp_path / "restaurado" / "dvc-store"
    salida = json.dumps({"datos/clips/1-Yo/x.mp4": "new", "datos/clips/2-Tu/y.mp4": "new"})
    ejecutor = EjecutorSecuencia((0, "", ""), (0, salida, ""), (0, "", ""))

    assert EstadoDvcLocal(tmp_path, ejecutar=ejecutor).ausentes_en(almacen) == 2

    dvc = [sys.executable, "-m", "dvc"]
    assert ejecutor.ordenes == [
        [*dvc, "remote", "add", "--local", "-f", "prueba-restauracion", str(almacen)],
        [*dvc, "status", "--cloud", "--remote", "prueba-restauracion", "--json"],
        [*dvc, "remote", "remove", "--local", "prueba-restauracion"],
    ]


def test_ausentes_retira_el_remoto_aunque_falle(tmp_path: Path) -> None:
    ejecutor = EjecutorSecuencia((0, "", ""), (1, "", "ERROR: algo"), (0, "", ""))

    with pytest.raises(RechazoDominio):
        EstadoDvcLocal(tmp_path, ejecutar=ejecutor).ausentes_en(tmp_path / "x")
    assert ejecutor.ordenes[-1][-3:] == ["remove", "--local", "prueba-restauracion"]


def test_ausentes_cero_si_todo_esta(tmp_path: Path) -> None:
    ejecutor = EjecutorSecuencia((0, "", ""), (0, "{}", ""), (0, "", ""))
    assert EstadoDvcLocal(tmp_path, ejecutar=ejecutor).ausentes_en(tmp_path / "x") == 0
