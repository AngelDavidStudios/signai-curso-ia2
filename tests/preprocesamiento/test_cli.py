"""CLI `signai-preproceso`: lote → aprobar-lote → procesar → verificar (contracts/cli.md C-1…C-5)."""

import json
from pathlib import Path

import numpy as np
import pytest

from signai_ml.preproceso.almacen import Versiones
from signai_ml.preproceso.cli import Contexto, main

from .dobles import MACOS, ExtractorDoble, crear_corpus

V = Versiones("signai-ml/preproceso/1", "signai-ml/keypoints/1", "1.0.0", "1.0.0", MACOS)
CLIPS = [
    (f"{c}__{g}__rep{r:02d}", 20 + r)
    for c, g in [("1-Yo", "IMG_1"), ("1-Yo", "IMG_2"), ("2-Tu", "IMG_3")]
    for r in range(1, 6)
]


@pytest.fixture
def ctx(tmp_path: Path) -> Contexto:
    crear_corpus(tmp_path, CLIPS)
    videos: list[str] = []

    def video_falso(clip: Path, keypoints: np.ndarray, destino: Path) -> None:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(b"mp4")
        videos.append(clip.stem)

    c = Contexto(
        raiz=tmp_path, desechable=tmp_path / "desechable", extractor=ExtractorDoble(dict(CLIPS)),
        versiones=V, generar_video=video_falso,
    )
    c.videos = videos  # type: ignore[attr-defined]
    return c


def test_flujo_completo(ctx: Contexto, capsys: pytest.CaptureFixture[str]) -> None:
    datos = ctx.raiz / "datos"

    assert main(["lote", "--procesos", "2"], ctx) == 0
    salida = capsys.readouterr().out
    assert "Proyección" in salida and "revision" in salida
    lote = json.loads((datos / "validacion" / "lote.json").read_text())
    assert len(lote["clips"]) == 3  # una por grabación
    assert (datos / "validacion" / "ventanas" / "ventanas.npy").is_file()
    assert (datos / "validacion" / "informe.md").is_file()
    assert not (datos / "keypoints").exists() and not (datos / "ventanas").exists()
    assert len(ctx.videos) == len(lote["revision"]) > 0  # type: ignore[attr-defined]

    assert main(["procesar"], ctx) == 1
    assert "aprobar-lote" in capsys.readouterr().out

    assert main(["aprobar-lote"], ctx) == 0
    assert json.loads((datos / "validacion" / "aprobacion.json").read_text())["decision"] == "aprobado"

    assert main(["procesar", "--procesos", "3"], ctx) == 0
    salida = capsys.readouterr().out
    assert "uv run dvc add datos/keypoints datos/ventanas datos/validacion && uv run dvc push" in salida
    assert np.load(datos / "ventanas" / "ventanas.npy").shape == (15, 75, 96)

    assert main(["verificar"], ctx) == 0
    assert "correcto" in capsys.readouterr().out


def test_rechazar_exige_nota_y_bloquea(ctx: Contexto, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["lote"], ctx) == 0
    assert main(["aprobar-lote", "--rechazar"], ctx) == 2
    assert main(["aprobar-lote", "--rechazar", "--nota", "lateralidad invertida"], ctx) == 0
    assert main(["procesar"], ctx) == 1
    assert "rechazado" in capsys.readouterr().out


def test_aprobar_sin_lote_procesado_es_error(ctx: Contexto, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["aprobar-lote"], ctx) == 1
    assert "lote" in capsys.readouterr().out


def test_volver_a_procesar_el_lote_con_otra_cadena_invalida_la_aprobacion(
    ctx: Contexto, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataclasses import replace

    assert main(["lote"], ctx) == 0
    assert main(["aprobar-lote"], ctx) == 0
    ctx.versiones = replace(V, version_cadena="signai-ml/preproceso/2")
    assert main(["lote"], ctx) == 0
    assert "invalidada" in capsys.readouterr().out
    assert not (ctx.raiz / "datos" / "validacion" / "aprobacion.json").exists()


def test_verificar_sin_resultado_falla(ctx: Contexto, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verificar"], ctx) == 1
    assert "ventanas.npy" in capsys.readouterr().out


def test_uso_incorrecto_devuelve_dos(ctx: Contexto) -> None:
    assert main(["no-existe"], ctx) == 2
    assert main([], ctx) == 2
