"""Metadatos y validación del artefacto, como los comprueba la app (contracts/artefacto.md A-2, A-3; SC-006).

`validar` replica `ModelArtifactDescription.validated()` de `signai-ios`: si la app rechazaría el
artefacto al cargarlo, la exportación lo rechaza antes de guardarlo.
"""

import json
import shutil
from pathlib import Path

import coremltools as ct
import pytest

from signai_ml.aceptacion.artefacto import (
    METADATOS,
    cifra,
    escribir_metadatos,
    ficha_artefacto,
    huella_paquete,
    leer_metadatos,
    unir,
    validar,
)
from signai_ml.aceptacion.conversion import ConvertidorCoreml
from signai_ml.aceptacion.errores import ArtefactoNoValido

GLOSAS = [f"glosa{n}" for n in range(1, 11)]


@pytest.fixture(scope="module")
def paquete_base(tmp_path_factory, modelo_diminuto, clases_diminuto) -> Path:
    return ConvertidorCoreml().convertir(modelo_diminuto, clases_diminuto, tmp_path_factory.mktemp("base")).paquete


@pytest.fixture
def paquete(paquete_base: Path, tmp_path: Path) -> Path:
    copia = tmp_path / paquete_base.name
    shutil.copytree(paquete_base, copia)
    return copia


def _metadatos(clases: list[str], glosas: list[str] = GLOSAS) -> dict[str, str]:
    return {
        "clases": ",".join(clases),
        "glosas": ",".join(glosas),
        "cifra": cifra(1.0, 1),
        "modelo.version": "1",
        "modelo.sha256": "a" * 64,
        "aceptacion.run_id": "r" * 32,
        "particion.huella": "e" * 64,
    }


def test_la_cifra_dice_que_mide() -> None:
    assert cifra(0.97222, 1) == "0.972 · prueba aleatoria 1.800 · consulta 1"
    assert cifra(1.0, 2) == "1.000 · prueba aleatoria 1.800 · consulta 2"


def test_las_claves_del_contrato() -> None:
    assert METADATOS == (
        "clases", "glosas", "cifra", "modelo.version", "modelo.sha256", "aceptacion.run_id", "particion.huella",
    )


def test_escribir_y_leer_metadatos(paquete: Path, clases_diminuto) -> None:
    escribir_metadatos(paquete, _metadatos(clases_diminuto))

    leidos = leer_metadatos(paquete)
    assert {k: leidos[k] for k in METADATOS} == _metadatos(clases_diminuto)
    descripcion = ct.models.MLModel(str(paquete), skip_model_load=True).get_spec().description.metadata
    assert "KPoseFormer" in descripcion.shortDescription
    assert "1.000" in descripcion.shortDescription
    assert descripcion.author == ""


def test_escribir_exige_todas_las_claves(paquete: Path, clases_diminuto) -> None:
    incompletos = _metadatos(clases_diminuto)
    del incompletos["cifra"]

    with pytest.raises(ArtefactoNoValido, match="cifra"):
        escribir_metadatos(paquete, incompletos)


def test_escribir_no_toca_los_pesos(paquete: Path, clases_diminuto) -> None:
    pesos = sorted(p for p in paquete.rglob("*") if p.is_file() and "weights" in p.parts)
    antes = [p.read_bytes() for p in pesos]

    escribir_metadatos(paquete, _metadatos(clases_diminuto))

    assert [p.read_bytes() for p in pesos] == antes


def test_un_artefacto_correcto_es_valido(paquete: Path, clases_diminuto) -> None:
    escribir_metadatos(paquete, _metadatos(clases_diminuto))

    validar(paquete, clases_diminuto, GLOSAS)


def test_clases_en_otro_orden(paquete: Path, clases_diminuto) -> None:
    escribir_metadatos(paquete, _metadatos(list(reversed(clases_diminuto))))

    with pytest.raises(ArtefactoNoValido, match="clases"):
        validar(paquete, clases_diminuto, GLOSAS)


@pytest.mark.parametrize("glosas", [GLOSAS[:9], GLOSAS[::-1]], ids=["menos", "otro-orden"])
def test_glosas_que_no_corresponden(paquete: Path, clases_diminuto, glosas) -> None:
    escribir_metadatos(paquete, _metadatos(clases_diminuto, glosas))

    with pytest.raises(ArtefactoNoValido, match="glosas"):
        validar(paquete, clases_diminuto, GLOSAS)


def test_una_glosa_vacia(paquete: Path, clases_diminuto) -> None:
    vacia = ["", *GLOSAS[1:]]
    escribir_metadatos(paquete, _metadatos(clases_diminuto, vacia))

    with pytest.raises(ArtefactoNoValido, match="vacía"):
        validar(paquete, clases_diminuto, vacia)


def test_una_glosa_con_coma_no_se_puede_unir() -> None:
    assert unir(GLOSAS, "glosas") == ",".join(GLOSAS)
    with pytest.raises(ArtefactoNoValido, match="coma"):
        unir(["a,b", *GLOSAS[1:]], "glosas")


def test_otro_numero_de_salidas(paquete: Path) -> None:
    once = [f"{n}-Sint{n}" for n in range(1, 12)]
    escribir_metadatos(paquete, _metadatos(once, [f"g{n}" for n in range(1, 12)]))

    with pytest.raises(ArtefactoNoValido, match="salidas"):
        validar(paquete, once, [f"g{n}" for n in range(1, 12)])


def test_otra_forma_de_entrada(paquete: Path, clases_diminuto) -> None:
    escribir_metadatos(paquete, _metadatos(clases_diminuto))
    spec = ct.models.MLModel(str(paquete), skip_model_load=True).get_spec()
    spec.description.input[0].type.multiArrayType.shape[1] = 74
    _guardar_spec(paquete, spec)

    with pytest.raises(ArtefactoNoValido, match="entrada"):
        validar(paquete, clases_diminuto, GLOSAS)


def _guardar_spec(paquete: Path, spec) -> None:
    (paquete / "Data" / "com.apple.CoreML" / "model.mlmodel").write_bytes(spec.SerializeToString())


def test_tipo_que_no_es_float16(paquete: Path, clases_diminuto) -> None:
    escribir_metadatos(paquete, _metadatos(clases_diminuto))
    spec = ct.models.MLModel(str(paquete), skip_model_load=True).get_spec()
    spec.description.output[0].type.multiArrayType.dataType = 65568  # FLOAT32
    _guardar_spec(paquete, spec)

    with pytest.raises(ArtefactoNoValido, match="FLOAT16"):
        validar(paquete, clases_diminuto, GLOSAS)


def test_la_huella_del_paquete_es_estable(paquete: Path, paquete_base: Path, tmp_path: Path) -> None:
    otra = tmp_path / "otra" / paquete_base.name
    shutil.copytree(paquete_base, otra)

    assert huella_paquete(paquete) == huella_paquete(otra)
    assert len(huella_paquete(paquete)) == 64
    (otra / "Manifest.json").write_text("{}", encoding="utf-8")
    assert huella_paquete(paquete) != huella_paquete(otra)


def test_la_ficha_del_artefacto(paquete: Path, clases_diminuto) -> None:
    escribir_metadatos(paquete, _metadatos(clases_diminuto))

    ficha = ficha_artefacto(paquete, ["aviso de prueba"])

    assert json.loads(json.dumps(ficha)) == ficha
    assert ficha["paquete"] == paquete.name
    assert ficha["sha256"] == huella_paquete(paquete)
    assert ficha["bytes"] == sum(p.stat().st_size for p in paquete.rglob("*") if p.is_file())
    assert ficha["formato"] == "mlprogram"
    assert ficha["precision"] == "float16"
    assert ficha["objetivo"] == "iOS17"
    assert ficha["metadatos"]["cifra"] == cifra(1.0, 1)
    assert {"coremltools", "torch"} <= set(ficha["bibliotecas"])
    assert ficha["avisos"] == ["aviso de prueba"]


def test_con_los_metadatos_escritos_sigue_prediciendo(paquete: Path, clases_diminuto, ventanas_prueba) -> None:
    import numpy as np

    escribir_metadatos(paquete, _metadatos(clases_diminuto))

    x, _ = ventanas_prueba
    salida = ct.models.MLModel(str(paquete)).predict({"ventana": x[:1].astype(np.float16)})["logits"]
    assert salida.shape == (1, 10)
    assert np.isfinite(salida).all()
