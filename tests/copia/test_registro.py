"""El registro de copias: `copia/registro.jsonl` (data-model, «Entrada del registro de copias»)."""

import json
from pathlib import Path

import pytest

from signai_ml.copia.errores import ErrorUso, RechazoDominio
from signai_ml.copia.registro import Entrada, Registro

FECHA = "2026-10-18T21:00:00+00:00"


def _correcta(**cambios: object) -> Entrada:
    datos: dict[str, object] = {"fecha": FECHA, "tipo": "actualizacion", "resultado": "correcto"}
    return Entrada(**(datos | cambios))  # type: ignore[arg-type]


def test_anade_una_linea_json_por_entrada(tmp_path: Path) -> None:
    registro = Registro(tmp_path)
    registro.anadir(_correcta(instantanea="abc", conjuntos=("seguimiento",), bytes_anadidos=10, duracion_s=3))
    registro.anadir(_correcta(tipo="envio", detalle={"carpeta": "signai-copia-20261018-2100", "archivos": 5}))

    lineas = (tmp_path / "copia" / "registro.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 2
    primera = json.loads(lineas[0])
    assert primera["instantanea"] == "abc"
    assert primera["conjuntos"] == ["seguimiento"]
    assert primera["motivo"] is None


def test_lee_las_ultimas_entradas(tmp_path: Path) -> None:
    registro = Registro(tmp_path)
    for indice in range(5):
        registro.anadir(_correcta(bytes_anadidos=indice))

    ultimas = registro.leer(ultimas=2)
    assert [entrada.bytes_anadidos for entrada in ultimas] == [3, 4]
    assert len(registro.leer()) == 5


def test_sin_registro_no_hay_entradas(tmp_path: Path) -> None:
    assert Registro(tmp_path).leer() == []


def test_los_tipos_son_los_del_modelo_de_datos() -> None:
    for tipo in ("actualizacion", "envio", "verificacion", "restauracion", "crudas"):
        _correcta(tipo=tipo)
    with pytest.raises(ErrorUso, match="tipo"):
        _correcta(tipo="otro")


def test_resultado_correcto_o_fallo() -> None:
    with pytest.raises(ErrorUso, match="resultado"):
        _correcta(resultado="regular")


def test_un_fallo_exige_motivo_y_un_acierto_no_lo_lleva() -> None:
    _correcta(resultado="fallo", motivo="hay corridas en curso")
    with pytest.raises(ErrorUso, match="motivo"):
        _correcta(resultado="fallo")
    with pytest.raises(ErrorUso, match="motivo"):
        _correcta(motivo="sobra")


@pytest.mark.parametrize("campo", ["bytes_anadidos", "duracion_s"])
def test_contadores_enteros_no_negativos(campo: str) -> None:
    with pytest.raises(ErrorUso, match=campo):
        _correcta(**{campo: -1})


@pytest.mark.parametrize(
    "valor",
    [
        "/Users/x/Raw-dataset-LSEC/clase",
        "Lista-Prioritaria/1-Yo",
        "IMG_1008.MOV",
        "clip.mp4",
    ],
)
def test_rechaza_datos_de_los_senantes_en_cualquier_campo(valor: str) -> None:
    with pytest.raises(RechazoDominio):
        _correcta(detalle={"anidado": {"ruta": valor}})
    with pytest.raises(RechazoDominio):
        _correcta(ronda=valor)


def test_rechaza_la_ruta_de_las_grabaciones_crudas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNAI_CRUDAS", "/Volumes/Grabaciones")
    with pytest.raises(RechazoDominio):
        _correcta(detalle={"donde": "/Volumes/Grabaciones/1-Yo"})


def test_sanear_quita_rutas_de_grabaciones_y_videos(monkeypatch: pytest.MonkeyPatch) -> None:
    from signai_ml.copia.registro import sanear

    monkeypatch.setenv("SIGNAI_CRUDAS", "/Volumes/Grabaciones")
    texto = sanear("falló datos/clips/1-Yo/x.mp4 y /Volumes/Grabaciones/1-Yo, en Raw-dataset-LSEC/a.MOV")
    assert texto == "falló [ruta omitida] y [ruta omitida], en [ruta omitida]"
    _correcta(resultado="fallo", motivo=texto)
