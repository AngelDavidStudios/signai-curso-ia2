"""Evaluación, aceptación y exportación del reconocedor (TT-15 · SP-216).

Mide el candidato de TT-14 sobre los 1.800 clips de prueba, lo acepta solo si alcanza ≥85 % Top-1
(ADR-0011), registra el modelo aceptado y lo exporta a Core ML en el formato de programa, en 16 bits,
con la paridad demostrada (ADR-0013, ADR-0014). Diseño: `specs/007-evaluacion-aceptacion-exportacion/`.
"""

from pathlib import Path

UMBRAL_TOP1 = (85, 100)  # aceptado si aciertos × 100 ≥ 85 × total (aritmética entera, research R-04)
CLIPS_PRUEBA = 1800
POR_CLASE = 18
TOLERANCIA_PARIDAD = 1e-2

MODELO_REGISTRADO = "reconocedor-kposeformer"
ALIAS = "aceptado"
EXPERIMENTO_ACEPTACION = "aceptacion"
EXPERIMENTO_EXPORTACION = "exportacion"

CANDIDATO = Path("modelos") / "reconocedor"
COREML = Path("modelos") / "coreml"
NOMBRE_PAQUETE = "KPoseFormer.mlpackage"
ENTRADA = "ventana"
SALIDA = "logits"
# Valor por defecto de `exportar --casos`: la fuente del contrato de normalización, nunca `tests/`.
CASOS_REFERENCIA = Path("..") / "signai-architecture" / "contracts" / "normalizacion-casos.json"
