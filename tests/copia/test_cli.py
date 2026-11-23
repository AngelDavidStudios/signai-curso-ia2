"""`signai-copia`: reglas comunes, `clave`, `iniciar` y `registro` (contracts/cli.md)."""

import io
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from signai_ml.copia import clave
from signai_ml.copia.cli import SUBCOMANDOS, Contexto, main
from signai_ml.copia.errores import FaltaHerramienta, RechazoDominio
from signai_ml.copia.registro import Entrada, Registro

from .conftest import base_mlflow
from .dobles import ClonadorCopia, DvcFalso
from .dobles import RepositorioFalso as RepositorioCompleto


class RepositorioFalso:
    """Doble del puerto `Repositorio`: recuerda con qué clave se abrió y si se inició."""

    def __init__(self, ruta: Path, clave_usada: str, existente: bool = False) -> None:
        self.ruta, self.clave_usada, self.iniciado = ruta, clave_usada, existente

    def existe(self) -> bool:
        return self.iniciado

    def iniciar(self) -> None:
        if self.iniciado:
            raise RechazoDominio("el repositorio ya existe")
        self.iniciado = True


class Mundo:
    """El contexto de la CLI con dobles y la salida capturada."""

    def __init__(self, raiz: Path, claves: list[str], terminal: bool = True) -> None:
        self.claves = iter(claves)
        self.preguntas: list[str] = []
        self.repositorios: list[RepositorioFalso] = []
        self.salida, self.error = io.StringIO(), io.StringIO()
        self.contexto = Contexto(
            raiz=raiz,
            repositorio_para=self._repositorio,
            pedir_clave=self._pedir,
            hay_terminal=lambda: terminal,
            ahora=lambda: datetime(2026, 10, 18, 21, 0, tzinfo=UTC),
            salida=self.salida,
            error=self.error,
        )

    def _pedir(self, mensaje: str) -> str:
        self.preguntas.append(mensaje)
        return next(self.claves)

    def _repositorio(self, ruta: Path, clave_usada: str) -> RepositorioFalso:
        repositorio = RepositorioFalso(ruta, clave_usada)
        self.repositorios.append(repositorio)
        return repositorio

    def ejecutar(self, *argumentos: str) -> int:
        return main(list(argumentos), self.contexto)

    @property
    def texto(self) -> str:
        return self.salida.getvalue() + self.error.getvalue()


@pytest.fixture
def mundo(raiz: Path, clave_prueba: str) -> Iterator[Mundo]:
    yield Mundo(raiz, [clave.formatear(clave_prueba)] * 4)


# ── Reglas comunes ────────────────────────────────────────────────────────────────────────────


def test_declara_los_once_subcomandos_del_contrato() -> None:
    assert SUBCOMANDOS == (
        "clave",
        "iniciar",
        "actualizar",
        "confirmar",
        "estado",
        "instantaneas",
        "restaurar",
        "comprobar",
        "reubicar",
        "anotar-crudas",
        "registro",
    )


def test_ayuda_lista_los_subcomandos(mundo: Mundo, capsys: pytest.CaptureFixture[str]) -> None:
    assert mundo.ejecutar("--help") == 0
    ayuda = capsys.readouterr().out
    for subcomando in SUBCOMANDOS:
        assert subcomando in ayuda


@pytest.mark.parametrize("argumentos", [("desconocido",), ("iniciar", "--clave", "X"), ("iniciar", "--password", "X")])
def test_error_de_uso_y_nunca_la_clave_como_argumento(mundo: Mundo, argumentos: tuple[str, ...]) -> None:
    assert mundo.ejecutar(*argumentos) == 2
    assert mundo.repositorios == []


def test_sin_terminal_no_pide_la_clave(raiz: Path, clave_prueba: str) -> None:
    mundo = Mundo(raiz, [clave_prueba], terminal=False)
    assert mundo.ejecutar("iniciar") == 2
    assert mundo.preguntas == []
    assert "terminal" in mundo.texto


@pytest.mark.parametrize(
    ("error", "codigo"),
    [(RechazoDominio("x"), 1), (FaltaHerramienta("restic"), 3)],
)
def test_cada_error_tiene_su_codigo(mundo: Mundo, error: Exception, codigo: int) -> None:
    def falla(ruta: Path, clave_usada: str) -> RepositorioFalso:
        raise error

    mundo.contexto.repositorio_para = falla
    assert mundo.ejecutar("iniciar") == codigo


def test_raiz_sustituye_a_la_de_datos(mundo: Mundo, tmp_path: Path) -> None:
    otra = tmp_path / "otra"
    assert mundo.ejecutar("--raiz", str(otra), "iniciar") == 0
    assert mundo.repositorios[0].ruta == otra / "copia" / "repositorio"


# ── clave ─────────────────────────────────────────────────────────────────────────────────────


def test_clave_se_muestra_una_vez_y_no_se_guarda(mundo: Mundo, raiz: Path) -> None:
    antes = sorted(p for p in raiz.rglob("*"))
    assert mundo.ejecutar("clave") == 0

    mostrada = next(linea for linea in mundo.salida.getvalue().splitlines() if linea.count("-") == 6)
    assert clave.validar(mostrada)
    assert "escríbela en papel" in mundo.texto
    assert "no se guarda en ningún sitio" in mundo.texto
    assert sorted(p for p in raiz.rglob("*")) == antes


# ── iniciar ───────────────────────────────────────────────────────────────────────────────────


def test_iniciar_pide_la_clave_dos_veces_y_crea_el_repositorio(mundo: Mundo, raiz: Path, clave_prueba: str) -> None:
    assert mundo.ejecutar("iniciar") == 0

    assert len(mundo.preguntas) == 2
    [repositorio] = mundo.repositorios
    assert repositorio.ruta == raiz / "copia" / "repositorio"
    assert repositorio.clave_usada == clave_prueba
    assert repositorio.iniciado
    assert clave_prueba not in mundo.texto


def test_iniciar_rechaza_dos_claves_distintas(raiz: Path, clave_prueba: str) -> None:
    mundo = Mundo(raiz, [clave_prueba, clave.generar()])
    assert mundo.ejecutar("iniciar") == 1
    assert mundo.repositorios == []
    assert "no coinciden" in mundo.texto


def test_iniciar_rechaza_una_clave_mal_formada(raiz: Path) -> None:
    mundo = Mundo(raiz, ["ABC", "ABC"])
    assert mundo.ejecutar("iniciar") == 1
    assert mundo.repositorios == []


# ── registro ──────────────────────────────────────────────────────────────────────────────────


def test_registro_muestra_las_ultimas_entradas(mundo: Mundo, raiz: Path) -> None:
    registro = Registro(raiz)
    for indice in range(3):
        registro.anadir(
            Entrada(fecha=f"2026-10-1{indice}T00:00:00+00:00", tipo="actualizacion", resultado="correcto")
        )

    assert mundo.ejecutar("registro", "--ultimas", "2") == 0
    assert "2026-10-11" in mundo.texto
    assert "2026-10-12" in mundo.texto
    assert "2026-10-10" not in mundo.texto


def test_registro_muestra_instantanea_ronda_y_motivo(mundo: Mundo, raiz: Path) -> None:
    registro = Registro(raiz)
    registro.anadir(
        Entrada(fecha="2026-10-18T00:00:00+00:00", tipo="actualizacion", resultado="correcto",
                instantanea="0123456789abcdef", ronda="barrido-1")
    )
    registro.anadir(
        Entrada(fecha="2026-10-18T01:00:00+00:00", tipo="actualizacion", resultado="fallo",
                motivo="hay corridas en curso")
    )

    assert mundo.ejecutar("registro") == 0
    assert "01234567" in mundo.texto
    assert "ronda=barrido-1" in mundo.texto
    assert "hay corridas en curso" in mundo.texto


def test_sin_contexto_usa_la_raiz_indicada(raiz: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--raiz", str(raiz), "registro"]) == 0
    assert "sin entradas" in capsys.readouterr().out


def test_sin_contexto_ni_raiz_usa_signai_data_root(
    raiz: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SIGNAI_DATA_ROOT", str(raiz))
    assert main(["registro"]) == 0
    assert "sin entradas" in capsys.readouterr().out


def test_registro_vacio(mundo: Mundo) -> None:
    assert mundo.ejecutar("registro") == 0
    assert "sin entradas" in mundo.texto


# ── anotar-crudas ─────────────────────────────────────────────────────────────────────────────


def test_anotar_crudas_correcta(mundo: Mundo, raiz: Path) -> None:
    codigo = mundo.ejecutar(
        "anotar-crudas", "--donde", "disco externo", "--fuera-del-equipo", "si", "--cifrada", "si", "--completa", "si"
    )
    assert codigo == 0
    assert Registro(raiz).leer()[-1].tipo == "crudas"
    assert mundo.preguntas == []


def test_anotar_crudas_pendiente(mundo: Mundo) -> None:
    codigo = mundo.ejecutar(
        "anotar-crudas", "--donde", "disco externo", "--fuera-del-equipo", "si", "--cifrada", "no", "--completa", "si"
    )
    assert codigo == 1
    assert "cifrada" in mundo.texto


def test_anotar_crudas_con_rutas_es_error_de_uso(mundo: Mundo) -> None:
    codigo = mundo.ejecutar(
        "anotar-crudas", "--donde", "/Volumes/X", "--fuera-del-equipo", "si", "--cifrada", "si", "--completa", "si"
    )
    assert codigo == 2


# ── actualizar y confirmar ────────────────────────────────────────────────────────────────────


@pytest.fixture
def mundo_copia(raiz: Path, clave_prueba: str) -> Mundo:
    mundo = Mundo(raiz, [clave.formatear(clave_prueba)] * 4)
    mundo.contexto.estado_dvc = DvcFalso()
    mundo.contexto.clonador = ClonadorCopia()
    repositorio = RepositorioCompleto(raiz)

    def repositorio_para(ruta: Path, clave_usada: str) -> RepositorioCompleto:
        mundo.repositorios.append(repositorio)  # type: ignore[arg-type]
        repositorio.clave_usada = clave_usada  # type: ignore[attr-defined]
        return repositorio

    mundo.contexto.repositorio_para = repositorio_para  # type: ignore[assignment]
    return mundo


def test_actualizar_pide_la_clave_una_vez_y_muestra_el_resumen_de_envio(
    mundo_copia: Mundo, clave_prueba: str
) -> None:
    assert mundo_copia.ejecutar("actualizar", "--ronda", "inicial") == 0

    assert len(mundo_copia.preguntas) == 1
    texto = mundo_copia.texto
    assert "signai-copia-20261018-2100" in texto
    assert "archivos" in texto and "elementos" in texto
    assert "espacio libre necesario en el iPad" in texto
    assert "signai-copia confirmar --carpeta signai-copia-20261018-2100 --archivos" in texto
    assert "datos-versionados" in texto
    assert clave_prueba not in texto


def test_actualizar_sin_datos_de_dvc_subidos_sale_con_1(mundo_copia: Mundo) -> None:
    mundo_copia.contexto.estado_dvc = DvcFalso(sin_subir=True)
    assert mundo_copia.ejecutar("actualizar") == 1
    assert "dvc push" in mundo_copia.texto


def test_confirmar_no_pide_la_clave_y_acepta_el_recuento(mundo_copia: Mundo, raiz: Path) -> None:
    mundo_copia.ejecutar("actualizar")
    [actualizacion] = Registro(raiz).leer()
    archivos = actualizacion.detalle["archivos"]
    preguntas = len(mundo_copia.preguntas)

    assert mundo_copia.ejecutar("confirmar", "--carpeta", "signai-copia-20261018-2100", "--archivos", str(archivos)) == 0
    assert len(mundo_copia.preguntas) == preguntas
    assert "confirmado" in mundo_copia.texto


def test_confirmar_con_otro_recuento_sale_con_1(mundo_copia: Mundo) -> None:
    mundo_copia.ejecutar("actualizar")
    assert mundo_copia.ejecutar("confirmar", "--carpeta", "signai-copia-20261018-2100", "--archivos", "9999") == 1
    assert "vuelve a arrastrarla" in mundo_copia.texto


def test_confirmar_un_segundo_envio_avisa_de_borrar_el_anterior(mundo_copia: Mundo, raiz: Path) -> None:
    for momento in (datetime(2026, 10, 18, 21, 0, tzinfo=UTC), datetime(2026, 10, 19, 9, 0, tzinfo=UTC)):
        mundo_copia.contexto.ahora = lambda momento=momento: momento
        mundo_copia.ejecutar("actualizar")
        actualizacion = Registro(raiz).leer()[-1]
        carpeta, archivos = actualizacion.detalle["carpeta"], actualizacion.detalle["archivos"]
        assert mundo_copia.ejecutar("confirmar", "--carpeta", carpeta, "--archivos", str(archivos)) == 0

    assert "Borra del iPad la carpeta anterior: signai-copia-20261018-2100" in mundo_copia.texto


# ── instantaneas, restaurar, comprobar y reubicar ─────────────────────────────────────────────


def test_instantaneas_pide_la_clave_y_lista(mundo_copia: Mundo, tmp_path: Path) -> None:
    mundo_copia.ejecutar("actualizar", "--ronda", "r1")
    preguntas = len(mundo_copia.preguntas)

    assert mundo_copia.ejecutar("instantaneas", "--repositorio", str(tmp_path / "devuelto")) == 0
    assert len(mundo_copia.preguntas) == preguntas + 1
    assert "nueva1" in mundo_copia.texto
    assert "ronda=r1" in mundo_copia.texto
    assert "2,0 KB" in mundo_copia.texto


def test_restaurar_pide_la_clave_una_vez(mundo_copia: Mundo, raiz: Path) -> None:
    destino = raiz / "copia" / "restauracion-20261018" / "datos"
    codigo = mundo_copia.ejecutar(
        "restaurar", "--repositorio", str(raiz / "x"), "--destino", str(destino), "--ruta", "dvc-store/a"
    )
    assert codigo == 0
    assert len(mundo_copia.preguntas) == 1
    assert "restaurado" in mundo_copia.texto


def test_restaurar_sobre_la_raiz_es_error_de_uso(mundo_copia: Mundo, raiz: Path) -> None:
    assert mundo_copia.ejecutar("restaurar", "--repositorio", str(raiz / "x"), "--destino", str(raiz)) == 2


def test_comprobar_no_pide_la_clave(mundo_copia: Mundo, raiz: Path) -> None:
    restaurado = raiz / "copia" / "restauracion-20261018" / "datos"
    for ruta in ("dvc-store", "mlflow"):
        shutil.copytree(raiz / ruta, restaurado / ruta)

    assert mundo_copia.ejecutar("comprobar", "--restaurado", str(restaurado)) == 0
    assert mundo_copia.preguntas == []
    assert "| Faltan | 0 |" in mundo_copia.texto


def test_comprobar_con_diferencias_sale_con_1(mundo_copia: Mundo, raiz: Path) -> None:
    restaurado = raiz / "copia" / "restauracion-20261018" / "datos"
    shutil.copytree(raiz / "mlflow", restaurado / "mlflow")
    assert mundo_copia.ejecutar("comprobar", "--restaurado", str(restaurado)) == 1


def test_reubicar(mundo: Mundo, tmp_path: Path) -> None:
    db = base_mlflow(tmp_path / "m" / "mlflow.db")
    de = (tmp_path / "m" / "artifacts").as_uri()
    assert mundo.ejecutar("reubicar", "--base", str(db), "--de", de, "--a", "file:///nuevo") == 0
    assert "3" in mundo.texto


def test_reubicar_algo_que_no_es_mlflow_sale_con_1(mundo: Mundo, tmp_path: Path) -> None:
    falsa = tmp_path / "x.db"
    falsa.write_bytes(b"nada")
    assert mundo.ejecutar("reubicar", "--base", str(falsa), "--de", "a", "--a", "b") == 1


def test_instantaneas_de_un_repositorio_vacio(mundo_copia: Mundo) -> None:
    assert mundo_copia.ejecutar("instantaneas") == 0
    assert "no tiene instantáneas" in mundo_copia.texto


# ── estado ────────────────────────────────────────────────────────────────────────────────────


def test_estado_no_pide_la_clave_y_sale_con_1_si_no_esta_al_dia(mundo_copia: Mundo) -> None:
    mundo_copia.ejecutar("actualizar")
    preguntas = len(mundo_copia.preguntas)

    assert mundo_copia.ejecutar("estado") == 1
    assert len(mundo_copia.preguntas) == preguntas
    assert "copia sin enviar al iPad" in mundo_copia.texto


def test_estado_al_dia_sale_con_0(mundo_copia: Mundo, raiz: Path) -> None:
    mundo_copia.ejecutar("actualizar")
    actualizacion = Registro(raiz).leer()[-1]
    mundo_copia.ejecutar(
        "confirmar", "--carpeta", actualizacion.detalle["carpeta"], "--archivos", str(actualizacion.detalle["archivos"])
    )

    assert mundo_copia.ejecutar("estado") == 0
    assert "Todo al día" in mundo_copia.texto


def test_estado_sin_copias(mundo_copia: Mundo) -> None:
    assert mundo_copia.ejecutar("estado") == 1
    assert "nunca" in mundo_copia.texto
    assert "seguimiento: pendiente" in mundo_copia.texto
