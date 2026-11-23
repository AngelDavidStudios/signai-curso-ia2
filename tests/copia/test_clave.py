"""La clave de la copia: generada, mostrada una vez y tecleada desde el papel (research R-07)."""

import pytest

from signai_ml.copia.clave import ALFABETO, LONGITUD, formatear, generar, normalizar, validar
from signai_ml.copia.errores import RechazoDominio


def test_alfabeto_sin_caracteres_ambiguos() -> None:
    assert ALFABETO == "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
    assert not set("0O1IL") & set(ALFABETO)


def test_genera_28_caracteres_del_alfabeto() -> None:
    clave = generar()
    assert LONGITUD == 28
    assert len(clave) == 28
    assert set(clave) <= set(ALFABETO)


def test_dos_claves_son_distintas() -> None:
    assert generar() != generar()


def test_se_muestra_en_siete_grupos_de_cuatro() -> None:
    clave = "ABCD" * 7
    assert formatear(clave) == "-".join(["ABCD"] * 7)


def test_normaliza_lo_tecleado_desde_el_papel() -> None:
    assert normalizar(" abcd-efgh \n") == "ABCDEFGH"
    assert normalizar("2345 6789") == "23456789"


def test_valida_y_devuelve_la_clave_normalizada() -> None:
    clave = generar()
    assert validar(formatear(clave).lower()) == clave


@pytest.mark.parametrize(
    "tecleada",
    ["ABCD-EFGH", "ABCD" * 7 + "A", "0BCD" * 7, "ABCL" * 7, "ABC!" * 7],
)
def test_rechaza_una_clave_mal_formada_sin_repetirla(tecleada: str) -> None:
    with pytest.raises(RechazoDominio) as error:
        validar(tecleada)
    assert normalizar(tecleada) not in str(error.value)
