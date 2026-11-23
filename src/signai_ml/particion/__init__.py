"""Partición de evaluación del corpus (TT-13 · SP-214; versión 2: TT-22 · SP-372).

Reparte una sola vez las ventanas de TT-12 por **grabación entera**: dentro de cada clase, sus
grabaciones se ordenan por la huella `sha256(semilla|clase|grabacion)` con una semilla fija; la primera
va a prueba y las demás a los pliegues A y B, con los que se elige el modelo por rotación. Ninguna
grabación aporta clips a dos lados. Decisión en `signai-architecture/decisions/0011-particion-80-20-y-umbral.md`
(enmienda del 2026-11-20) y constitución 3.0.0; diseño en `specs/008-reentrenamiento-generalizacion/`.
"""

from enum import StrEnum

VERSION_PARTICION = "signai-ml/particion/2"
VERSION_ANTERIOR = "signai-ml/particion/1"
SEMILLA_PARTICION = 372
GRABACIONES_MINIMAS = 3
CLASES_ESPERADAS = 100


class Lado(StrEnum):
    PRUEBA = "prueba"
    PLIEGUE_A = "pliegue_a"
    PLIEGUE_B = "pliegue_b"
