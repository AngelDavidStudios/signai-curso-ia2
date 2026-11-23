"""Configuración de una corrida (data-model «Configuración»; spec FR-006, FR-009)."""

import pytest

from signai_ml.reconocedor.configuracion import Configuracion, Forma, configuracion_desde_dict
from signai_ml.reconocedor.errores import RondaNoValida

BASE = {
    "forma": {"capas": 4, "dimension": 98, "cabezas": 7, "intermedia": 256},
    "entrenamiento": {
        "optimizador": "adamw", "tasa": 1e-3, "programacion": "coseno", "regularizacion": 0.05,
        "dropout": 0.05, "recorte_gradiente": 1.0, "lote": 32, "epocas": 200, "suavizado_etiquetas": 0.0,
    },
}


def _con(**cambios) -> dict:
    datos = {"forma": dict(BASE["forma"]), "entrenamiento": dict(BASE["entrenamiento"])}
    for seccion, valores in cambios.items():
        datos.setdefault(seccion, {}).update(valores)
    return datos


def test_lee_la_base() -> None:
    cfg = configuracion_desde_dict("base", BASE)
    assert cfg.nombre == "base"
    assert cfg.forma == Forma(4, 98, 7, 256)
    assert cfg.entrenamiento.tasa == 1e-3
    assert cfg.aumento == ()


def test_valores_por_defecto() -> None:
    cfg = configuracion_desde_dict("x", {})
    assert cfg.forma == Forma(4, 98, 7, 256)
    assert cfg.entrenamiento.epocas == 200 and cfg.entrenamiento.suavizado_etiquetas == 0.0


@pytest.mark.parametrize("nombre", ["Base", "con espacio", "a_b", ""])
def test_nombre_invalido(nombre: str) -> None:
    with pytest.raises(RondaNoValida, match="nombre"):
        configuracion_desde_dict(nombre, BASE)


@pytest.mark.parametrize(
    ("seccion", "clave", "valor"),
    [
        ("forma", "capas", 0),
        ("forma", "dimension", 1),
        ("forma", "cabezas", 8),
        ("forma", "intermedia", 0),
        ("entrenamiento", "optimizador", "sgd"),
        ("entrenamiento", "tasa", 0),
        ("entrenamiento", "programacion", "lineal"),
        ("entrenamiento", "regularizacion", -0.1),
        ("entrenamiento", "dropout", 1.0),
        ("entrenamiento", "recorte_gradiente", -1),
        ("entrenamiento", "lote", 0),
        ("entrenamiento", "epocas", 0),
        ("entrenamiento", "suavizado_etiquetas", 1.0),
        ("entrenamiento", "lote", 3.5),
        ("forma", "capas", True),
    ],
)
def test_valor_fuera_de_rango(seccion: str, clave: str, valor: object) -> None:
    with pytest.raises(RondaNoValida, match=f"{seccion}.{clave}"):
        configuracion_desde_dict("x", _con(**{seccion: {clave: valor}}))


def test_clave_desconocida() -> None:
    with pytest.raises(RondaNoValida, match="entrenamiento.momento"):
        configuracion_desde_dict("x", _con(entrenamiento={"momento": 0.9}))
    with pytest.raises(RondaNoValida, match="sobra"):
        configuracion_desde_dict("x", {**BASE, "sobra": 1})


def test_aumento_valido_y_ordenado() -> None:
    cfg = configuracion_desde_dict(
        "a",
        {**BASE, "aumento": {"escala": {"probabilidad": 0.5, "intensidad": 0.1},
                             "desplazamiento": {"probabilidad": 1.0, "intensidad": 0.05}}},
    )
    assert [tipo for tipo, _ in cfg.aumento] == ["desplazamiento", "escala"]


@pytest.mark.parametrize(
    ("aumento", "ruta"),
    [
        ({"espejo": {"probabilidad": 0.5, "intensidad": 0.1}}, "aumento.espejo"),
        ({"rotacion": {"probabilidad": 0.5, "intensidad": 0.0}}, "aumento.rotacion.intensidad"),
        ({"rotacion": {"probabilidad": 0.5, "intensidad": 0.5}}, "aumento.rotacion.intensidad"),
        ({"escala": {"probabilidad": 1.5, "intensidad": 0.1}}, "aumento.escala.probabilidad"),
        ({"escala": {"probabilidad": 0.5, "intensidad": 0.0}}, "aumento.escala.intensidad"),
        ({"recorte_temporal": {"probabilidad": 0.5, "intensidad": 1.0}}, "aumento.recorte_temporal.intensidad"),
        ({"ocultacion": {"probabilidad": 0.5}}, "aumento.ocultacion.intensidad"),
        ({"escala": {"probabilidad": 0.5, "intensidad": 0.1, "otra": 1}}, "aumento.escala.otra"),
    ],
)
def test_aumento_invalido(aumento: dict, ruta: str) -> None:
    with pytest.raises(RondaNoValida, match=ruta):
        configuracion_desde_dict("a", {**BASE, "aumento": aumento})


def test_aplanar() -> None:
    cfg = configuracion_desde_dict("a", {**BASE, "aumento": {"escala": {"probabilidad": 0.5, "intensidad": 0.1}}})
    plano = cfg.aplanar()
    assert plano["forma.capas"] == 4
    assert plano["entrenamiento.tasa"] == 1e-3
    assert plano["aumento.escala.probabilidad"] == 0.5
    assert "nombre" not in plano


def test_mismos_valores_con_otro_nombre_se_detectan() -> None:
    a = configuracion_desde_dict("a", BASE)
    b = configuracion_desde_dict("b", BASE)
    c = configuracion_desde_dict("c", _con(entrenamiento={"dropout": 0.2}))
    assert a.valores() == b.valores()
    assert a.valores() != c.valores()
    assert isinstance(a, Configuracion)


def test_rotacion_valida_entre_los_espaciales() -> None:
    cfg = configuracion_desde_dict(
        "r",
        {**BASE, "aumento": {"velocidad": {"probabilidad": 0.5, "intensidad": 0.2},
                             "rotacion": {"probabilidad": 0.5, "intensidad": 0.15},
                             "escala": {"probabilidad": 0.5, "intensidad": 0.1}}},
    )
    assert [tipo for tipo, _ in cfg.aumento] == ["escala", "rotacion", "velocidad"]
