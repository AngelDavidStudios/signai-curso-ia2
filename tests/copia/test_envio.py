"""El envío al iPad: carpeta fechada, resumen y confirmación (data-model, «Envío al iPad»; R-03, R-05)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from signai_ml.copia.envio import ClonadorApfs, Envios, contar, nombre_carpeta
from signai_ml.copia.errores import RechazoDominio

MOMENTO = datetime(2026, 10, 18, 21, 30, tzinfo=UTC)


class ClonadorFalso:
    """Copia de verdad con `shutil`: lo que importa aquí es qué se clona y adónde."""

    def __init__(self) -> None:
        self.llamadas: list[tuple[Path, Path]] = []

    def clonar(self, origen: Path, destino: Path) -> None:
        import shutil

        self.llamadas.append((origen, destino))
        shutil.copytree(origen, destino)


@pytest.fixture
def repositorio(raiz: Path) -> Path:
    """Un árbol con la forma de un repositorio de restic: subcarpetas y archivos."""
    repositorio = raiz / "copia" / "repositorio"
    for relativa in ("config", "keys/k1", "data/00/a", "data/01/b", "data/01/c", "index/i1", "snapshots/s1"):
        archivo = repositorio / relativa
        archivo.parent.mkdir(parents=True, exist_ok=True)
        archivo.write_bytes(b"x" * 10)
    (repositorio / "locks").mkdir()
    return repositorio


def test_nombre_fechado_unico_por_actualizacion() -> None:
    assert nombre_carpeta(MOMENTO) == "signai-copia-20261018-2130"


def test_cuenta_archivos_elementos_y_bytes(repositorio: Path) -> None:
    # 7 archivos y 8 subcarpetas (keys, data, data/00, data/01, index, snapshots y locks, más locks vacía).
    archivos, elementos, total = contar(repositorio)
    assert archivos == 7
    assert elementos == 7 + 7
    assert total == 70


def test_prepara_un_clon_en_para_ipad(raiz: Path, repositorio: Path) -> None:
    clonador = ClonadorFalso()
    envio = Envios(raiz, clonador).preparar("abc123", MOMENTO)

    destino = raiz / "copia" / "para-ipad" / "signai-copia-20261018-2130"
    assert clonador.llamadas == [(repositorio, destino)]
    assert (envio.carpeta, envio.instantanea) == ("signai-copia-20261018-2130", "abc123")
    assert (envio.archivos, envio.elementos, envio.bytes) == (7, 14, 70)
    assert envio.ruta == destino
    assert envio.espacio_necesario == 140


def test_no_pisa_un_envio_con_el_mismo_nombre(raiz: Path, repositorio: Path) -> None:
    envios = Envios(raiz, ClonadorFalso())
    envios.preparar("abc", MOMENTO)
    with pytest.raises(RechazoDominio, match="ya existe"):
        envios.preparar("def", MOMENTO)


@pytest.mark.parametrize("recuento", [7, 14])
def test_confirma_con_archivos_o_con_elementos_y_borra_los_anteriores(
    raiz: Path, repositorio: Path, recuento: int
) -> None:
    envios = Envios(raiz, ClonadorFalso())
    anterior = envios.preparar("antes", datetime(2026, 10, 1, 8, 0, tzinfo=UTC))
    nuevo = envios.preparar("ahora", MOMENTO)

    archivos, total = envios.confirmar(nuevo.carpeta, recuento)

    assert (archivos, total) == (7, 70)
    assert nuevo.ruta.exists()
    assert not anterior.ruta.exists()


def test_un_recuento_distinto_se_rechaza_sin_borrar_nada(raiz: Path, repositorio: Path) -> None:
    envios = Envios(raiz, ClonadorFalso())
    anterior = envios.preparar("antes", datetime(2026, 10, 1, 8, 0, tzinfo=UTC))
    nuevo = envios.preparar("ahora", MOMENTO)

    with pytest.raises(RechazoDominio, match="vuelve a arrastrarla"):
        envios.confirmar(nuevo.carpeta, 5)
    assert anterior.ruta.exists()
    assert nuevo.ruta.exists()


def test_una_carpeta_desconocida_se_rechaza(raiz: Path, repositorio: Path) -> None:
    with pytest.raises(RechazoDominio, match="no hay ningún envío"):
        Envios(raiz, ClonadorFalso()).confirmar("signai-copia-20990101-0000", 7)


def test_el_clon_apfs_reproduce_el_arbol(repositorio: Path, tmp_path: Path) -> None:
    destino = tmp_path / "clon"
    ClonadorApfs().clonar(repositorio, destino)

    assert contar(destino) == contar(repositorio)
    assert (destino / "data" / "01" / "c").read_bytes() == b"x" * 10


def test_el_clon_apfs_falla_con_un_origen_inexistente(tmp_path: Path) -> None:
    with pytest.raises(RechazoDominio, match="no se pudo clonar"):
        ClonadorApfs().clonar(tmp_path / "no-existe", tmp_path / "clon")


def test_sin_cp_falta_herramienta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from signai_ml.copia.errores import FaltaHerramienta

    monkeypatch.setenv("PATH", str(tmp_path / "vacio"))
    with pytest.raises(FaltaHerramienta, match="cp"):
        ClonadorApfs().clonar(tmp_path, tmp_path / "clon")


def test_confirmar_ignora_lo_que_no_es_un_envio(raiz: Path, repositorio: Path) -> None:
    envios = Envios(raiz, ClonadorFalso())
    nuevo = envios.preparar("ahora", MOMENTO)
    ajeno = raiz / "copia" / "para-ipad" / "signai-copia-nota.txt"
    ajeno.write_text("no es una carpeta", encoding="utf-8")

    envios.confirmar(nuevo.carpeta, nuevo.archivos)
    assert ajeno.exists()


def test_confirmar_un_envio_antiguo_no_borra_el_nuevo(raiz: Path, repositorio: Path) -> None:
    envios = Envios(raiz, ClonadorFalso())
    mas_antiguo = envios.preparar("a", datetime(2026, 9, 1, 8, 0, tzinfo=UTC))
    antiguo = envios.preparar("b", datetime(2026, 10, 1, 8, 0, tzinfo=UTC))
    nuevo = envios.preparar("c", MOMENTO)

    envios.confirmar(antiguo.carpeta, antiguo.archivos)

    assert not mas_antiguo.ruta.exists()
    assert antiguo.ruta.exists()
    assert nuevo.ruta.exists()
