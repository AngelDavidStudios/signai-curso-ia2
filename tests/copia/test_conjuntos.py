"""Qué se copia: dos conjuntos protegidos y el registro como metadato (data-model; research R-02)."""

from pathlib import Path

from signai_ml.copia.conjuntos import ConjuntoProtegido, conjuntos, exclusiones, rutas_de_copia


def test_hay_exactamente_dos_conjuntos() -> None:
    assert conjuntos() == (
        ConjuntoProtegido("datos-versionados", "dvc-store"),
        ConjuntoProtegido("seguimiento", "mlflow"),
    )


def test_el_registro_entra_como_metadato_si_existe(raiz: Path) -> None:
    assert rutas_de_copia(raiz) == ("dvc-store", "mlflow")

    (raiz / "copia").mkdir()
    (raiz / "copia" / "registro.jsonl").write_text("{}\n", encoding="utf-8")
    assert rutas_de_copia(raiz) == ("dvc-store", "mlflow", "copia/registro.jsonl")


def test_excluye_el_material_desechable_y_el_resto_de_la_copia(raiz: Path) -> None:
    excluidas = exclusiones(raiz)
    for relativa in ("revision", "cierre", "studio", "copia/repositorio", "copia/para-ipad", "copia/restauracion-*"):
        assert str(raiz / relativa) in excluidas


def test_nunca_apunta_a_la_cache_de_dvc_ni_a_las_grabaciones(raiz: Path) -> None:
    rutas = " ".join((*rutas_de_copia(raiz), *exclusiones(raiz)))
    assert ".dvc/cache" not in rutas
    assert "Raw-dataset" not in rutas
