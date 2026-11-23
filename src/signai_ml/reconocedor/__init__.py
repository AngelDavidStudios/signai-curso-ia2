"""Entrenamiento del reconocedor KPoseFormer (TT-14 · SP-215; protocolo por rotación: TT-22 · SP-372).

Entrena desde cero, en el equipo local, el clasificador de las 100 señas que la aplicación evalúa seña a
seña, recortada (ADR-0002, enmienda del 2026-11-20), como la referencia publicada. Recorre un barrido
abierto por rondas: cada configuración se entrena con 3 semillas en las 2 rotaciones de la partición
por grabación (entrena un pliegue y valida el otro), y se elige por su acierto en grabaciones no vistas.
El candidato es el modelo final, entrenado con los dos pliegues hasta las épocas elegidas, sin validar.
Todo se registra en MLflow. Decisiones en `signai-architecture/decisions/0013-…`, `0010-…` y `0011-…`
(enmienda del 2026-11-20); diseño en `specs/006-entrenamiento-reconocedor/` y
`specs/008-reentrenamiento-generalizacion/`.
"""

VERSION_RECONOCEDOR = "signai-ml/reconocedor/1"

SEMILLAS = (215, 216, 217)
CLASES = 100
FOTOGRAMAS = 75
RASGOS = 96

UMBRAL_CONFIANZA = 0.95
MEJORA_MINIMA = 0.005
RONDAS_SIN_MEJORA = 2
TOLERANCIA_REPETICION = 0.01

# Límite del dispositivo para el candidato: enmienda de ADR-0013 del 2026-11-14 (research R-12).
LIMITE_PARAMETROS = 2_000_000
