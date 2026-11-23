"""La clave de la copia (research R-07; spec FR-004, FR-004a).

Solo existe en papel: se genera aquí, se muestra una vez y después siempre se teclea. Por eso el
alfabeto evita los caracteres que se confunden al transcribir (`0 O 1 I L`) y lo tecleado se
normaliza antes de usarlo. 28 caracteres de un alfabeto de 31 dan ≈139 bits.
"""

import secrets

from .errores import RechazoDominio

ALFABETO = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
LONGITUD = 28
GRUPO = 4


def generar() -> str:
    return "".join(secrets.choice(ALFABETO) for _ in range(LONGITUD))


def formatear(clave: str) -> str:
    """En grupos de cuatro, para escribirla y teclearla sin perderse."""
    return "-".join(clave[inicio : inicio + GRUPO] for inicio in range(0, len(clave), GRUPO))


def normalizar(texto: str) -> str:
    return "".join(caracter for caracter in texto if not caracter.isspace() and caracter != "-").upper()


def validar(texto: str) -> str:
    """Devuelve la clave normalizada. El mensaje de error nunca la repite."""
    clave = normalizar(texto)
    if len(clave) != LONGITUD or not set(clave) <= set(ALFABETO):
        raise RechazoDominio(
            f"la clave no tiene el formato esperado: {LONGITUD} caracteres en grupos de {GRUPO}, "
            "sin 0, O, 1, I ni L. Revisa lo tecleado contra el papel"
        )
    return clave
