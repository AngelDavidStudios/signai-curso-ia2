"""Números en español para la salida de la CLI: 1.800, 97,22 %, 0,972."""


def miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def porcentaje(valor: float) -> str:
    return f"{valor * 100:.2f}".replace(".", ",") + " %"


def decimal(valor: float, cifras: int = 3) -> str:
    return f"{valor:.{cifras}f}".replace(".", ",")
