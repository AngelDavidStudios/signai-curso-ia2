"""Casos de uso de la copia (contracts/cli.md): grabaciones crudas, actualizar y confirmar."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from signai_ml.copia.envio import ClonadorApfs, Envios
from signai_ml.copia.errores import ErrorUso, RechazoDominio
from signai_ml.copia.manifiesto import leer as leer_manifiesto
from signai_ml.copia.registro import Registro
from signai_ml.copia.repositorio import Restic
from signai_ml.copia.servicio import ServicioCopia

from .conftest import base_mlflow
from .dobles import ClonadorCopia, DvcFalso, RepositorioFalso

AHORA = datetime(2026, 10, 18, 21, 0, tzinfo=UTC)


@pytest.fixture
def servicio(raiz: Path) -> ServicioCopia:
    return ServicioCopia(raiz, registro=Registro(raiz), ahora=lambda: AHORA)


def test_anota_la_copia_de_las_crudas_cuando_cumple(servicio: ServicioCopia, raiz: Path) -> None:
    servicio.anotar_crudas("disco externo cifrado", fuera=True, cifrada=True, completa=True)

    [entrada] = Registro(raiz).leer()
    assert (entrada.tipo, entrada.resultado, entrada.fecha) == ("crudas", "correcto", AHORA.isoformat())
    assert entrada.detalle == {
        "donde": "disco externo cifrado",
        "fuera_del_equipo": True,
        "cifrada": True,
        "completa": True,
    }


@pytest.mark.parametrize(
    ("fuera", "cifrada", "completa", "falta"),
    [(False, True, True, "fuera del equipo"), (True, False, True, "cifrada"), (True, True, False, "completa")],
)
def test_si_algo_no_cumple_queda_pendiente_y_anotado(
    servicio: ServicioCopia, raiz: Path, fuera: bool, cifrada: bool, completa: bool, falta: str
) -> None:
    with pytest.raises(RechazoDominio, match=falta):
        servicio.anotar_crudas("disco externo", fuera=fuera, cifrada=cifrada, completa=completa)

    [entrada] = Registro(raiz).leer()
    assert entrada.resultado == "fallo"
    assert "pendiente de TT-21" in (entrada.motivo or "")


@pytest.mark.parametrize("donde", ["/Volumes/Copia", "~/Copia", "disco/carpeta"])
def test_donde_se_describe_sin_rutas(servicio: ServicioCopia, raiz: Path, donde: str) -> None:
    with pytest.raises(ErrorUso, match="sin rutas"):
        servicio.anotar_crudas(donde, fuera=True, cifrada=True, completa=True)
    assert Registro(raiz).leer() == []


# ── SP-293: actualizar y confirmar ────────────────────────────────────────────────────────────


def _servicio(raiz: Path, repositorio: object, dvc: object | None = None) -> ServicioCopia:
    return ServicioCopia(
        raiz,
        registro=Registro(raiz),
        ahora=lambda: AHORA,
        repositorio=repositorio,  # type: ignore[arg-type]
        estado_dvc=dvc or DvcFalso(),  # type: ignore[arg-type]
        envios=Envios(raiz, ClonadorCopia()),
    )


def test_con_datos_de_dvc_sin_subir_no_copia_y_anota_el_fallo(raiz: Path) -> None:
    repositorio = RepositorioFalso(raiz)
    with pytest.raises(RechazoDominio, match="uv run dvc push"):
        _servicio(raiz, repositorio, DvcFalso(sin_subir=True)).actualizar()

    assert repositorio.copias == []
    [entrada] = Registro(raiz).leer()
    assert (entrada.tipo, entrada.resultado) == ("actualizacion", "fallo")
    assert "dvc push" in (entrada.motivo or "")


def test_la_primera_copia_lleva_la_etiqueta_inicial(raiz: Path) -> None:
    repositorio = RepositorioFalso(raiz)
    _servicio(raiz, repositorio).actualizar()
    assert repositorio.copias[0]["etiquetas"] == ("signai", "inicial")


def test_la_ronda_siempre_gana(raiz: Path) -> None:
    repositorio = RepositorioFalso(raiz)
    _servicio(raiz, repositorio).actualizar(ronda="barrido-1")
    assert repositorio.copias[0]["etiquetas"] == ("signai", "ronda=barrido-1")


def test_las_siguientes_sin_ronda_solo_llevan_signai(raiz: Path) -> None:
    repositorio = RepositorioFalso(raiz, instantaneas=1)
    _servicio(raiz, repositorio).actualizar()
    assert repositorio.copias[0]["etiquetas"] == ("signai",)


def test_actualizar_copia_verifica_guarda_el_manifiesto_y_prepara_el_envio(raiz: Path) -> None:
    repositorio = RepositorioFalso(raiz)
    resultado = _servicio(raiz, repositorio).actualizar(ronda="r1")

    copia = repositorio.copias[0]
    assert copia["rutas"] == ("dvc-store", "mlflow")
    assert copia["base"] == raiz
    assert str(raiz / "revision") in copia["excluir"]
    assert repositorio.verificaciones == ["5%"]
    assert leer_manifiesto(raiz) is not None
    assert resultado.envio.carpeta == "signai-copia-20261018-2100"
    assert resultado.envio.ruta.is_dir()
    assert resultado.conjuntos == ("datos-versionados", "seguimiento")

    [entrada] = Registro(raiz).leer()
    assert (entrada.tipo, entrada.resultado, entrada.instantanea, entrada.ronda) == (
        "actualizacion",
        "correcto",
        "nueva1",
        "r1",
    )
    assert entrada.conjuntos == ("datos-versionados", "seguimiento")
    assert entrada.bytes_anadidos == 1234
    assert entrada.duracion_s >= 0
    assert entrada.detalle["carpeta"] == "signai-copia-20261018-2100"


def test_la_segunda_copia_incluye_el_registro_y_solo_lo_que_cambio(raiz: Path) -> None:
    repositorio = RepositorioFalso(raiz)
    servicio = _servicio(raiz, repositorio)
    servicio.actualizar(ronda="r1")
    (raiz / "mlflow" / "mlflow.db").unlink()
    base_mlflow(raiz / "mlflow" / "mlflow.db", corridas=3)  # una corrida más

    servicio_despues = ServicioCopia(
        raiz,
        registro=Registro(raiz),
        ahora=lambda: datetime(2026, 10, 18, 22, 0, tzinfo=UTC),
        repositorio=repositorio,  # type: ignore[arg-type]
        estado_dvc=DvcFalso(),  # type: ignore[arg-type]
        envios=Envios(raiz, ClonadorCopia()),
    )
    resultado = servicio_despues.actualizar(ronda="r2")

    assert repositorio.copias[1]["rutas"] == ("dvc-store", "mlflow", "copia/registro.jsonl")
    assert resultado.conjuntos == ("seguimiento",)


def test_un_error_al_copiar_se_anota_y_no_prepara_envio(raiz: Path) -> None:
    class RepositorioRoto(RepositorioFalso):
        def copiar(self, rutas, **opciones):  # type: ignore[no-untyped-def]
            raise RechazoDominio("restic backup falló: disco lleno")

    with pytest.raises(RechazoDominio, match="disco lleno"):
        _servicio(raiz, RepositorioRoto(raiz)).actualizar()

    [entrada] = Registro(raiz).leer()
    assert entrada.resultado == "fallo"
    assert not (raiz / "copia" / "para-ipad").exists() or not any((raiz / "copia" / "para-ipad").iterdir())


def test_confirmar_anota_el_envio(raiz: Path) -> None:
    servicio = _servicio(raiz, RepositorioFalso(raiz))
    resultado = servicio.actualizar(ronda="r1")

    servicio.confirmar(resultado.envio.carpeta, resultado.envio.archivos)

    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado, entrada.instantanea) == ("envio", "correcto", "nueva1")
    assert entrada.detalle == {
        "carpeta": resultado.envio.carpeta,
        "archivos": resultado.envio.archivos,
        "bytes": resultado.envio.bytes,
    }


def test_confirmar_con_otro_recuento_anota_el_fallo(raiz: Path) -> None:
    servicio = _servicio(raiz, RepositorioFalso(raiz))
    resultado = servicio.actualizar()

    with pytest.raises(RechazoDominio):
        servicio.confirmar(resultado.envio.carpeta, 999)

    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado) == ("envio", "fallo")


def test_confirmar_devuelve_la_carpeta_anterior_confirmada(raiz: Path) -> None:
    servicio = _servicio(raiz, RepositorioFalso(raiz))
    primero = servicio.actualizar()
    assert servicio.confirmar(primero.envio.carpeta, primero.envio.archivos) is None

    despues = ServicioCopia(
        raiz,
        registro=Registro(raiz),
        ahora=lambda: datetime(2026, 10, 19, 9, 0, tzinfo=UTC),
        repositorio=RepositorioFalso(raiz, instantaneas=1),  # type: ignore[arg-type]
        estado_dvc=DvcFalso(),  # type: ignore[arg-type]
        envios=Envios(raiz, ClonadorCopia()),
    )
    segundo = despues.actualizar()
    assert despues.confirmar(segundo.envio.carpeta, segundo.envio.archivos) == primero.envio.carpeta


def test_primera_copia_y_envio_real_local(raiz: Path, clave_prueba: str, restic_disponible: str, tmp_path: Path) -> None:
    repositorio = Restic(raiz / "copia" / "repositorio", clave_prueba)
    repositorio.iniciar()
    servicio = ServicioCopia(
        raiz,
        registro=Registro(raiz),
        ahora=lambda: AHORA,
        repositorio=repositorio,
        estado_dvc=DvcFalso(),  # type: ignore[arg-type]
        envios=Envios(raiz, ClonadorApfs()),
    )

    resultado = servicio.actualizar(ronda="inicial")
    servicio.confirmar(resultado.envio.carpeta, resultado.envio.archivos)

    # El envío es un repositorio completo: restaurarlo reproduce el origen sin lo excluido.
    devuelto = Restic(resultado.envio.ruta, clave_prueba)
    devuelto.restaurar("latest", tmp_path / "restaurado")
    assert (tmp_path / "restaurado" / "mlflow" / "mlflow.db").read_bytes() == (raiz / "mlflow" / "mlflow.db").read_bytes()
    assert len(list((tmp_path / "restaurado" / "dvc-store").rglob("*/*/*/*"))) == 20
    assert not (tmp_path / "restaurado" / "revision").exists()


def test_sin_dependencias_es_error_de_uso(raiz: Path) -> None:
    solo_registro = ServicioCopia(raiz, registro=Registro(raiz), ahora=lambda: AHORA)
    with pytest.raises(ErrorUso, match="repositorio"):
        solo_registro.actualizar()
    with pytest.raises(ErrorUso, match="envíos"):
        solo_registro.confirmar("signai-copia-20261018-2100", 1)


# ── SP-294: restaurar y comprobar ─────────────────────────────────────────────────────────────


def _restaurado_identico(raiz: Path) -> Path:
    destino = raiz / "copia" / "restauracion-20261018" / "datos"
    for ruta in ("dvc-store", "mlflow"):
        shutil.copytree(raiz / ruta, destino / ruta)
    return destino


@pytest.mark.parametrize("relativo", ["", "dvc-store", "copia", "copia/repositorio/x", "copia/otra-cosa"])
def test_restaurar_no_admite_destinos_dentro_de_la_raiz(raiz: Path, relativo: str) -> None:
    repositorio = RepositorioFalso(raiz)
    with pytest.raises(ErrorUso, match="destino"):
        _servicio(raiz, repositorio).restaurar(repositorio, raiz / relativo)
    assert repositorio.restauraciones == []


def test_restaurar_admite_copia_restauracion(raiz: Path) -> None:
    repositorio = RepositorioFalso(raiz)
    destino = raiz / "copia" / "restauracion-20261018" / "datos"

    _servicio(raiz, repositorio).restaurar(repositorio, destino, instantanea="abc", ruta="dvc-store/x")

    assert repositorio.verificaciones == ["completa"]
    assert repositorio.restauraciones == [("abc", destino, "dvc-store/x")]
    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado) == ("verificacion", "correcto")


def test_restaurar_fuera_de_la_raiz(raiz: Path, tmp_path: Path) -> None:
    repositorio = RepositorioFalso(raiz)
    _servicio(raiz, repositorio).restaurar(repositorio, tmp_path / "fuera")
    assert repositorio.restauraciones[0][0] == "latest"


def test_restaurar_no_pisa_un_destino_con_contenido(raiz: Path, tmp_path: Path) -> None:
    ocupado = tmp_path / "ocupado"
    ocupado.mkdir()
    (ocupado / "algo").write_text("x", encoding="utf-8")
    repositorio = RepositorioFalso(raiz)

    with pytest.raises(RechazoDominio, match="no está vacío"):
        _servicio(raiz, repositorio).restaurar(repositorio, ocupado)


def test_restaurar_anota_el_fallo_de_la_verificacion(raiz: Path, tmp_path: Path) -> None:
    class RepositorioDanado(RepositorioFalso):
        def verificar(self, *, subconjunto: str | None = None, completo: bool = False) -> None:
            raise RechazoDominio("restic check falló: pack dañado")

    with pytest.raises(RechazoDominio, match="pack dañado"):
        _servicio(raiz, RepositorioDanado(raiz)).restaurar(RepositorioDanado(raiz), tmp_path / "d")
    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado) == ("verificacion", "fallo")


def test_comprobar_una_restauracion_identica(raiz: Path) -> None:
    restaurado = _restaurado_identico(raiz)
    dvc = DvcFalso()

    resumen = _servicio(raiz, RepositorioFalso(raiz), dvc).comprobar(restaurado)

    assert dvc.almacenes == [restaurado / "dvc-store"]
    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado) == ("restauracion", "correcto")
    assert entrada.detalle["faltan"] == entrada.detalle["sobran"] == entrada.detalle["distintos"] == 0
    assert entrada.detalle["archivos"] == 21
    assert entrada.detalle["dvc_ausentes"] == 0
    assert entrada.detalle["corridas_origen"] == entrada.detalle["corridas_restauradas"] == 2
    assert entrada.detalle["integridad"] == "ok"
    assert str(raiz) not in json.dumps(entrada.detalle)
    assert "| Faltan | 0 |" in resumen
    assert str(raiz) not in resumen


def test_comprobar_detecta_diferencias_y_las_anota(raiz: Path) -> None:
    restaurado = _restaurado_identico(raiz)
    next((restaurado / "dvc-store").rglob("*/*/*/*")).unlink()

    with pytest.raises(RechazoDominio, match="faltan 1"):
        _servicio(raiz, RepositorioFalso(raiz)).comprobar(restaurado)
    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado, entrada.detalle["faltan"]) == ("restauracion", "fallo", 1)


def test_comprobar_falla_si_dvc_echa_en_falta_objetos(raiz: Path) -> None:
    restaurado = _restaurado_identico(raiz)
    with pytest.raises(RechazoDominio, match="DVC"):
        _servicio(raiz, RepositorioFalso(raiz), DvcFalso(ausentes=3)).comprobar(restaurado)


def test_comprobar_falla_si_el_seguimiento_no_cuadra(raiz: Path) -> None:
    restaurado = _restaurado_identico(raiz)
    (restaurado / "mlflow" / "mlflow.db").unlink()
    base_mlflow(restaurado / "mlflow" / "mlflow.db", corridas=1)
    with pytest.raises(RechazoDominio, match="corridas"):
        _servicio(raiz, RepositorioFalso(raiz)).comprobar(restaurado)


def test_restaurar_y_comprobar_real_local(raiz: Path, clave_prueba: str, restic_disponible: str) -> None:
    repositorio = Restic(raiz / "copia" / "repositorio", clave_prueba)
    repositorio.iniciar()
    servicio = ServicioCopia(
        raiz,
        registro=Registro(raiz),
        ahora=lambda: AHORA,
        repositorio=repositorio,
        estado_dvc=DvcFalso(),  # type: ignore[arg-type]
        envios=Envios(raiz, ClonadorApfs()),
    )
    primera = servicio.actualizar(ronda="r1")
    objeto = sorted((raiz / "dvc-store").rglob("*/*/*/*"))[0]
    original = objeto.read_bytes()
    relativa = objeto.relative_to(raiz).as_posix()

    # La copia vuelve «del iPad»: un clon de la carpeta de envío, dentro de copia/restauracion-*.
    vuelta = raiz / "copia" / "restauracion-20261018"
    ClonadorApfs().clonar(primera.envio.ruta, vuelta / "repositorio")
    devuelta = Restic(vuelta / "repositorio", clave_prueba)
    servicio.restaurar(devuelta, vuelta / "datos")
    servicio.comprobar(vuelta / "datos")
    assert Registro(raiz).leer()[-1].resultado == "correcto"

    # SC-007: un archivo borrado después se recupera desde la instantánea anterior.
    objeto.unlink()
    servicio_despues = ServicioCopia(
        raiz,
        registro=Registro(raiz),
        ahora=lambda: datetime(2026, 10, 18, 22, 0, tzinfo=UTC),
        repositorio=repositorio,
        estado_dvc=DvcFalso(),  # type: ignore[arg-type]
        envios=Envios(raiz, ClonadorApfs()),
    )
    servicio_despues.actualizar(ronda="r2")
    servicio_despues.restaurar(repositorio, vuelta / "archivo", instantanea=primera.instantanea, ruta=relativa)
    assert (vuelta / "archivo" / relativa).read_bytes() == original


def test_comprobar_sin_estado_de_dvc_es_error_de_uso(raiz: Path) -> None:
    solo_registro = ServicioCopia(raiz, registro=Registro(raiz), ahora=lambda: AHORA)
    with pytest.raises(ErrorUso, match="DVC"):
        solo_registro.comprobar(raiz / "copia" / "restauracion-x" / "datos")


# ── SP-295 · US3: rondas seguras ──────────────────────────────────────────────────────────────


class RepositorioQueOlvida(RepositorioFalso):
    def __init__(self, raiz: Path) -> None:
        super().__init__(raiz)
        self.pasos: list[str] = []

    def copiar(self, rutas, **opciones):  # type: ignore[no-untyped-def]
        self.pasos.append("copiar")
        return super().copiar(rutas, **opciones)

    def olvidar(self, *, dentro_de: str, ultimas: int) -> None:
        self.pasos.append(f"olvidar {dentro_de} {ultimas}")

    def verificar(self, *, subconjunto: str | None = None, completo: bool = False) -> None:
        self.pasos.append(f"verificar {subconjunto}")


def test_con_corridas_en_curso_no_copia(raiz: Path) -> None:
    (raiz / "mlflow" / "mlflow.db").unlink()
    base_mlflow(raiz / "mlflow" / "mlflow.db", corridas=1, en_curso=1)
    repositorio = RepositorioFalso(raiz)

    with pytest.raises(RechazoDominio, match="hay corridas en curso"):
        _servicio(raiz, repositorio).actualizar()
    assert repositorio.copias == []
    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado) == ("actualizacion", "fallo")


def test_olvida_lo_antiguo_entre_copiar_y_verificar(raiz: Path) -> None:
    repositorio = RepositorioQueOlvida(raiz)
    _servicio(raiz, repositorio).actualizar()
    assert repositorio.pasos == ["copiar", "olvidar 30d 1", "verificar 5%"]


def test_una_verificacion_fallida_se_anota_y_no_prepara_envio(raiz: Path) -> None:
    class RepositorioDudoso(RepositorioFalso):
        def verificar(self, *, subconjunto: str | None = None, completo: bool = False) -> None:
            raise RechazoDominio("restic check falló: índice incoherente")

    with pytest.raises(RechazoDominio, match="índice incoherente"):
        _servicio(raiz, RepositorioDudoso(raiz)).actualizar()

    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado) == ("verificacion", "fallo")
    assert not (raiz / "copia" / "para-ipad").exists()


def test_retencion_real_local(raiz: Path, clave_prueba: str, restic_disponible: str) -> None:
    repositorio = Restic(raiz / "copia" / "repositorio", clave_prueba)
    repositorio.iniciar()
    antigua = repositorio.copiar(("dvc-store",), base=raiz, etiquetas=("signai",), cuando="2026-08-01 10:00:00")

    resultado = _servicio(raiz, repositorio).actualizar(ronda="r1")

    ids = [instantanea.id for instantanea in repositorio.instantaneas()]
    assert antigua.instantanea not in ids
    assert ids == [resultado.instantanea]


# ── SP-295 · US4: estado sin clave ────────────────────────────────────────────────────────────


class RepositorioIntocable(RepositorioFalso):
    """`estado` no debe abrir el repositorio: abrirlo exigiría la clave (R-10)."""

    def instantaneas(self):  # type: ignore[no-untyped-def]
        raise AssertionError("estado no debe abrir el repositorio")


def test_estado_sin_ninguna_copia(raiz: Path) -> None:
    estado = _servicio(raiz, RepositorioIntocable(raiz)).estado()
    assert estado.ultima_actualizacion is None
    assert not estado.en_ipad
    assert estado.conjuntos == {"datos-versionados": "pendiente", "seguimiento": "pendiente"}
    assert not estado.al_dia


def test_estado_copiada_pero_sin_enviar(raiz: Path) -> None:
    servicio = _servicio(raiz, RepositorioFalso(raiz))
    resultado = servicio.actualizar()

    estado = _servicio(raiz, RepositorioIntocable(raiz)).estado()
    assert estado.ultima_actualizacion == AHORA.isoformat()
    assert estado.instantanea == resultado.instantanea
    assert estado.carpeta == resultado.envio.carpeta
    assert not estado.en_ipad
    assert estado.conjuntos == {"datos-versionados": "al-dia", "seguimiento": "al-dia"}
    assert not estado.al_dia


def test_estado_al_dia_tras_confirmar(raiz: Path) -> None:
    servicio = _servicio(raiz, RepositorioFalso(raiz))
    resultado = servicio.actualizar()
    servicio.confirmar(resultado.envio.carpeta, resultado.envio.archivos)

    estado = _servicio(raiz, RepositorioIntocable(raiz)).estado()
    assert estado.en_ipad
    assert estado.al_dia


def test_estado_detecta_cambios_corridas_y_dvc(raiz: Path) -> None:
    servicio = _servicio(raiz, RepositorioFalso(raiz))
    resultado = servicio.actualizar()
    servicio.confirmar(resultado.envio.carpeta, resultado.envio.archivos)
    (raiz / "mlflow" / "mlflow.db").unlink()
    base_mlflow(raiz / "mlflow" / "mlflow.db", corridas=2, en_curso=1)

    estado = _servicio(raiz, RepositorioIntocable(raiz), DvcFalso(sin_subir=True)).estado()
    assert estado.conjuntos == {"datos-versionados": "al-dia", "seguimiento": "pendiente"}
    assert estado.corridas_en_curso == 1
    assert estado.dvc_sin_subir
    assert estado.en_ipad
    assert not estado.al_dia


def test_un_envio_fallido_no_cuenta_como_en_el_ipad(raiz: Path) -> None:
    servicio = _servicio(raiz, RepositorioFalso(raiz))
    resultado = servicio.actualizar()
    with pytest.raises(RechazoDominio):
        servicio.confirmar(resultado.envio.carpeta, 999)

    assert not _servicio(raiz, RepositorioIntocable(raiz)).estado().en_ipad


# ── Convergencia: los fallos con rutas de clips se anotan saneados (T072) ──────────────────────


def test_un_fallo_con_rutas_de_clips_se_anota_sin_ellas(raiz: Path) -> None:
    class DvcConRutas(DvcFalso):
        def sin_subir(self) -> bool:
            raise RechazoDominio("dvc status falló: datos/clips/1-Yo/1-Yo__IMG_1008__rep01.mp4 ilegible")

    with pytest.raises(RechazoDominio, match="rep01.mp4"):
        _servicio(raiz, RepositorioFalso(raiz), DvcConRutas()).actualizar()

    entrada = Registro(raiz).leer()[-1]
    assert entrada.resultado == "fallo"
    assert "[ruta omitida]" in (entrada.motivo or "")
    assert ".mp4" not in (entrada.motivo or "")


def test_confirmar_con_un_nombre_que_parece_un_video_se_anota_saneado(raiz: Path) -> None:
    servicio = _servicio(raiz, RepositorioFalso(raiz))
    servicio.actualizar()

    with pytest.raises(RechazoDominio):
        servicio.confirmar("IMG_1008.MOV", 3)

    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado) == ("envio", "fallo")
    assert "IMG_1008.MOV" not in json.dumps(entrada.detalle) + (entrada.motivo or "")


# ── Convergencia: restaurar y comprobar anotan sus fallos (T075) ───────────────────────────────


def test_un_fallo_al_restaurar_se_anota(raiz: Path, tmp_path: Path) -> None:
    class RepositorioQueNoRestaura(RepositorioFalso):
        def restaurar(self, instantanea: str, destino: Path, *, ruta: str | None = None) -> None:
            raise RechazoDominio("restic restore falló: sin espacio en el disco")

    repositorio = RepositorioQueNoRestaura(raiz)
    with pytest.raises(RechazoDominio, match="sin espacio"):
        _servicio(raiz, repositorio).restaurar(repositorio, tmp_path / "destino")

    verificacion, restauracion = Registro(raiz).leer()[-2:]
    assert (verificacion.tipo, verificacion.resultado) == ("verificacion", "correcto")
    assert (restauracion.tipo, restauracion.resultado) == ("restauracion", "fallo")
    assert "sin espacio" in (restauracion.motivo or "")


def test_un_error_de_dvc_al_comprobar_se_anota_saneado(raiz: Path) -> None:
    class DvcRoto(DvcFalso):
        def ausentes_en(self, almacen: Path) -> int:
            raise RechazoDominio("dvc status falló: datos/clips/1-Yo/1-Yo__IMG_1008__rep01.mp4")

    restaurado = _restaurado_identico(raiz)
    with pytest.raises(RechazoDominio, match="dvc status"):
        _servicio(raiz, RepositorioFalso(raiz), DvcRoto()).comprobar(restaurado)

    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado) == ("restauracion", "fallo")
    assert "[ruta omitida]" in (entrada.motivo or "")
    assert ".mp4" not in (entrada.motivo or "")


def test_una_base_ilegible_al_comprobar_se_anota(raiz: Path) -> None:
    restaurado = _restaurado_identico(raiz)
    (restaurado / "mlflow" / "mlflow.db").write_bytes(b"no es una base")

    with pytest.raises(RechazoDominio, match="MLflow"):
        _servicio(raiz, RepositorioFalso(raiz)).comprobar(restaurado)

    entrada = Registro(raiz).leer()[-1]
    assert (entrada.tipo, entrada.resultado) == ("restauracion", "fallo")
    assert entrada.motivo
