"""Humo de evaluación: la misma semilla reproduce los mismos tensores (constitución VI; R-10)."""

import torch


def _ventana(semilla: int) -> torch.Tensor:
    torch.manual_seed(semilla)
    return torch.rand(75, 96)


def test_misma_semilla_mismos_tensores() -> None:
    assert torch.equal(_ventana(42), _ventana(42))


def test_semillas_distintas_tensores_distintos() -> None:
    assert not torch.equal(_ventana(42), _ventana(43))
