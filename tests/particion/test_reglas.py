"""Reparto de las grabaciones de una clase (TT-22, research R-01; ADR-0011, enmienda del 2026-11-20)."""

import pytest

from signai_ml.particion import GRABACIONES_MINIMAS, Lado
from signai_ml.particion.errores import IndiceNoValido
from signai_ml.particion.reglas import lados_por_posicion

P, A, B = Lado.PRUEBA, Lado.PLIEGUE_A, Lado.PLIEGUE_B


def test_hacen_falta_tres_grabaciones() -> None:
    assert GRABACIONES_MINIMAS == 3


@pytest.mark.parametrize(
    ("n", "esperado"),
    [(3, [P, A, B]), (4, [P, A, B, A]), (5, [P, A, B, A, B]), (6, [P, A, B, A, B, A])],
)
def test_la_primera_a_prueba_y_el_resto_alternando_pliegues(n: int, esperado: list[Lado]) -> None:
    assert lados_por_posicion(n, "1-Sint") == esperado


def test_siempre_una_sola_grabacion_en_prueba() -> None:
    for n in range(3, 12):
        assert lados_por_posicion(n, "1-Sint").count(P) == 1


@pytest.mark.parametrize("n", [0, 1, 2])
def test_con_menos_de_tres_se_rechaza_nombrando_clase_y_grabaciones(n: int) -> None:
    with pytest.raises(IndiceNoValido, match=rf"7-Sint.*{n} grabaci"):
        lados_por_posicion(n, "7-Sint")
