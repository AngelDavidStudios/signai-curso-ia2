"""Adaptador `Restic` del puerto `Repositorio` (research R-01, R-07, R-08).

La mitad de las pruebas usa restic real con un repositorio local en `tmp_path`; la otra mitad,
un ejecutor inyectado para comprobar cómo se construyen las órdenes (sobre todo, que la clave
nunca aparece en `argv`).
"""

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from signai_ml.copia.errores import FaltaHerramienta, RechazoDominio
from signai_ml.copia.repositorio import Restic

RUTAS = ("dvc-store", "mlflow")


class Ejecutor:
    def __init__(self, codigo: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.codigo, self.stdout, self.stderr = codigo, stdout, stderr
        self.llamadas: list[tuple[Sequence[str], dict[str, object]]] = []

    def __call__(self, orden: Sequence[str], **opciones: object) -> subprocess.CompletedProcess[str]:
        self.llamadas.append((orden, opciones))
        return subprocess.CompletedProcess(orden, self.codigo, self.stdout, self.stderr)


# ── Construcción de las órdenes (sin restic) ──────────────────────────────────────────────────


def test_la_clave_solo_viaja_en_el_entorno_del_proceso_hijo(tmp_path: Path, clave_prueba: str) -> None:
    ejecutor = Ejecutor(stdout="[]")
    Restic(tmp_path / "repo", clave_prueba, ejecutar=ejecutor).instantaneas()

    orden, opciones = ejecutor.llamadas[0]
    assert all(clave_prueba not in argumento for argumento in orden)
    entorno = opciones["env"]
    assert isinstance(entorno, dict)
    assert entorno["RESTIC_PASSWORD"] == clave_prueba
    assert entorno["RESTIC_REPOSITORY"] == str(tmp_path / "repo")


def test_ignora_otras_fuentes_de_clave_del_entorno(
    tmp_path: Path, clave_prueba: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESTIC_PASSWORD_FILE", "/tmp/otra")
    monkeypatch.setenv("RESTIC_PASSWORD_COMMAND", "echo otra")
    ejecutor = Ejecutor(stdout="[]")
    Restic(tmp_path / "repo", clave_prueba, ejecutar=ejecutor).instantaneas()

    entorno = ejecutor.llamadas[0][1]["env"]
    assert isinstance(entorno, dict)
    assert "RESTIC_PASSWORD_FILE" not in entorno
    assert "RESTIC_PASSWORD_COMMAND" not in entorno


def test_olvidar_agrupa_solo_por_maquina(tmp_path: Path, clave_prueba: str) -> None:
    # Por defecto restic agrupa también por rutas, y una instantánea con otras rutas nunca se olvidaría.
    ejecutor = Ejecutor()
    Restic(tmp_path / "repo", clave_prueba, ejecutar=ejecutor).olvidar(dentro_de="30d", ultimas=1)

    orden = list(ejecutor.llamadas[0][0])
    assert orden[1:] == ["forget", "--group-by", "host", "--keep-within", "30d", "--keep-last", "1", "--prune"]


def test_copia_desde_la_raiz_con_rutas_relativas(tmp_path: Path, clave_prueba: str) -> None:
    ejecutor = Ejecutor(stdout='{"message_type":"summary","snapshot_id":"abc","data_added":7}\n')
    resultado = Restic(tmp_path / "repo", clave_prueba, ejecutar=ejecutor).copiar(
        RUTAS, base=tmp_path, etiquetas=("signai", "inicial"), excluir=("revision",)
    )

    orden, opciones = ejecutor.llamadas[0]
    assert opciones["cwd"] == tmp_path
    assert list(orden[-2:]) == list(RUTAS)
    assert "--tag" in orden and "inicial" in orden
    assert (resultado.instantanea, resultado.bytes_anadidos) == ("abc", 7)


def test_sin_restic_falta_herramienta(tmp_path: Path, clave_prueba: str) -> None:
    def sin_restic(orden: Sequence[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(orden[0])

    with pytest.raises(FaltaHerramienta, match="brew install restic"):
        Restic(tmp_path / "repo", clave_prueba, ejecutar=sin_restic).instantaneas()


def test_verificar_sin_opciones_solo_comprueba_la_estructura(tmp_path: Path, clave_prueba: str) -> None:
    ejecutor = Ejecutor()
    Restic(tmp_path / "repo", clave_prueba, ejecutar=ejecutor).verificar()
    assert list(ejecutor.llamadas[0][0]) == ["restic", "check"]


def test_un_error_de_restic_es_rechazo_de_dominio(tmp_path: Path, clave_prueba: str) -> None:
    ejecutor = Ejecutor(codigo=1, stderr="Fatal: algo salió mal\n")
    with pytest.raises(RechazoDominio, match="algo salió mal"):
        Restic(tmp_path / "repo", clave_prueba, ejecutar=ejecutor).verificar(subconjunto="5%")


# ── Integración con restic real ──────────────────────────────────────────────────────────────


@pytest.fixture
def repositorio(tmp_path: Path, clave_prueba: str, restic_disponible: str) -> Restic:
    repositorio = Restic(tmp_path / "repo", clave_prueba)
    repositorio.iniciar()
    return repositorio


def test_inicia_un_repositorio_y_no_lo_repite(repositorio: Restic) -> None:
    assert repositorio.existe()
    with pytest.raises(RechazoDominio, match="ya existe"):
        repositorio.iniciar()


def test_copia_y_lista_las_instantaneas(repositorio: Restic, raiz: Path) -> None:
    resultado = repositorio.copiar(RUTAS, base=raiz, etiquetas=("signai", "inicial"))

    [instantanea] = repositorio.instantaneas()
    assert instantanea.id == resultado.instantanea
    assert set(instantanea.etiquetas) == {"signai", "inicial"}
    assert instantanea.bytes > 0
    assert resultado.bytes_anadidos > 0


def test_restaura_completo_y_por_ruta(repositorio: Restic, raiz: Path, tmp_path: Path) -> None:
    resultado = repositorio.copiar(RUTAS, base=raiz, etiquetas=("signai",))
    un_objeto = next((raiz / "dvc-store").rglob("*/*/*/*"))
    relativa = un_objeto.relative_to(raiz).as_posix()

    repositorio.restaurar(resultado.instantanea, tmp_path / "completo")
    repositorio.restaurar(resultado.instantanea, tmp_path / "suelto", ruta=relativa)

    assert (tmp_path / "completo" / "mlflow" / "mlflow.db").is_file()
    assert not (tmp_path / "completo" / "revision").exists()
    restaurados = [p for p in (tmp_path / "suelto").rglob("*") if p.is_file()]
    assert [p.relative_to(tmp_path / "suelto").as_posix() for p in restaurados] == [relativa]
    assert restaurados[0].read_bytes() == un_objeto.read_bytes()


def test_verifica_por_muestreo_y_completo(repositorio: Restic, raiz: Path) -> None:
    repositorio.copiar(RUTAS, base=raiz, etiquetas=("signai",))
    repositorio.verificar(subconjunto="5%")
    repositorio.verificar(completo=True)


def test_olvida_lo_antiguo_y_conserva_la_ultima(repositorio: Restic, raiz: Path) -> None:
    repositorio.copiar(("dvc-store",), base=raiz, etiquetas=("signai",), cuando="2026-08-01 10:00:00")
    reciente = repositorio.copiar(RUTAS, base=raiz, etiquetas=("signai",))

    repositorio.olvidar(dentro_de="30d", ultimas=1)

    assert [i.id for i in repositorio.instantaneas()] == [reciente.instantanea]


def test_una_clave_equivocada_no_abre_el_repositorio(repositorio: Restic, tmp_path: Path) -> None:
    with pytest.raises(RechazoDominio, match="clave"):
        Restic(tmp_path / "repo", "OTRA" * 7).instantaneas()


def test_un_repositorio_inexistente_se_nombra(tmp_path: Path, clave_prueba: str, restic_disponible: str) -> None:
    sin_repositorio = Restic(tmp_path / "no-existe", clave_prueba)
    assert not sin_repositorio.existe()
    with pytest.raises(RechazoDominio, match="no hay ningún repositorio"):
        sin_repositorio.instantaneas()
