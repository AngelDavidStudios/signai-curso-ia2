"""Declaración de rondas en TOML (contracts/ronda.md; research R-05)."""

import hashlib
from pathlib import Path

import pytest

from signai_ml.reconocedor.errores import RondaNoValida
from signai_ml.reconocedor.ronda import buscar_configuracion, leer_barrido, leer_ronda, ruta_barrido

RONDA_1 = """
barrido = "b01"
ronda = 1

[base.forma]
capas = 4
dimension = 98
cabezas = 7
intermedia = 256

[base.entrenamiento]
tasa = 1e-3
dropout = 0.05
epocas = 200

[[configuracion]]
nombre = "base"

[[configuracion]]
nombre = "dropout-020"
entrenamiento = { dropout = 0.20 }

[[configuracion]]
nombre = "aumento-espacial"
aumento.desplazamiento = { probabilidad = 0.5, intensidad = 0.05 }
"""

RONDA_2 = """
barrido = "b01"
ronda = 2

[base.entrenamiento]
epocas = 200

[[configuracion]]
nombre = "capas-6"
forma = { capas = 6 }
"""


def _escribir(directorio: Path, nombre: str, texto: str) -> Path:
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / nombre
    ruta.write_text(texto, encoding="utf-8")
    return ruta


def test_fusion_profunda_de_la_base(tmp_path: Path) -> None:
    ronda = leer_ronda(_escribir(tmp_path, "ronda-01.toml", RONDA_1))
    assert (ronda.barrido, ronda.numero) == ("b01", 1)
    base, dropout, aumento = ronda.configuraciones
    assert base.entrenamiento.dropout == 0.05 and base.entrenamiento.tasa == 1e-3
    assert dropout.entrenamiento.dropout == 0.20 and dropout.entrenamiento.tasa == 1e-3
    assert dropout.forma == base.forma
    assert aumento.aumento[0][0] == "desplazamiento"


def test_huella_es_el_sha256_del_fichero(tmp_path: Path) -> None:
    ruta = _escribir(tmp_path, "ronda-01.toml", RONDA_1)
    assert leer_ronda(ruta).huella == hashlib.sha256(ruta.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("texto", "mensaje"),
    [
        (RONDA_1.replace("dropout = 0.20", "dropout = 2.0"), "entrenamiento.dropout"),
        (RONDA_1 + '\n[[configuracion]]\nnombre = "base"\n', "repetido"),
        (RONDA_1 + '\n[[configuracion]]\nnombre = "otra"\n', "mismos valores"),
        (RONDA_1.replace('barrido = "b01"', ""), "barrido"),
        (RONDA_1.replace("ronda = 1", "ronda = 0"), "ronda"),
        (RONDA_1 + '\nextra = 1\n', "extra"),
        ('barrido = "b01"\nronda = 1\n', "configuración"),
        ("esto no es toml = = ", "TOML"),
    ],
)
def test_ronda_invalida(tmp_path: Path, texto: str, mensaje: str) -> None:
    with pytest.raises(RondaNoValida, match=mensaje):
        leer_ronda(_escribir(tmp_path, "ronda-01.toml", texto))


def test_leer_barrido_en_orden(tmp_path: Path) -> None:
    directorio = tmp_path / "b01"
    _escribir(directorio, "ronda-02.toml", RONDA_2)
    _escribir(directorio, "ronda-01.toml", RONDA_1)
    rondas = leer_barrido(directorio)
    assert [r.numero for r in rondas] == [1, 2]
    ronda, cfg = buscar_configuracion(rondas, "capas-6")
    assert ronda.numero == 2 and cfg.forma.capas == 6


def test_leer_barrido_rechaza_huecos_y_repetidos(tmp_path: Path) -> None:
    directorio = tmp_path / "b01"
    _escribir(directorio, "ronda-02.toml", RONDA_2)
    with pytest.raises(RondaNoValida, match="consecutiv"):
        leer_barrido(directorio)
    _escribir(directorio, "ronda-01.toml", RONDA_1)
    _escribir(directorio, "ronda-03.toml", RONDA_2.replace("ronda = 2", "ronda = 3"))
    with pytest.raises(RondaNoValida, match="repetido"):
        leer_barrido(directorio)


def test_leer_barrido_rechaza_valores_repetidos_entre_rondas(tmp_path: Path) -> None:
    directorio = tmp_path / "b01"
    _escribir(directorio, "ronda-01.toml", RONDA_1)
    _escribir(directorio, "ronda-02.toml", RONDA_2.replace("forma = { capas = 6 }", ""))
    with pytest.raises(RondaNoValida, match="mismos valores"):
        leer_barrido(directorio)


def test_leer_barrido_rechaza_otro_barrido(tmp_path: Path) -> None:
    _escribir(tmp_path / "b02", "ronda-01.toml", RONDA_1)
    with pytest.raises(RondaNoValida, match="b02"):
        leer_barrido(tmp_path / "b02")


def test_barrido_sin_rondas(tmp_path: Path) -> None:
    with pytest.raises(RondaNoValida, match="no tiene rondas"):
        leer_barrido(tmp_path / "b09")


def test_configuracion_no_declarada(tmp_path: Path) -> None:
    rondas = leer_barrido(_escribir(tmp_path / "b01", "ronda-01.toml", RONDA_1).parent)
    with pytest.raises(RondaNoValida, match="no-existe"):
        buscar_configuracion(rondas, "no-existe")


def test_ruta_barrido(tmp_path: Path) -> None:
    assert ruta_barrido(tmp_path, "b01") == tmp_path / "config" / "reconocedor" / "b01"
