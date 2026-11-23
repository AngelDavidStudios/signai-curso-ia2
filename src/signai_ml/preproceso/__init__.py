"""Cadena de preprocesamiento del corpus (TT-12 · SP-213): clip → keypoints → ventana (75, 96).

Tres etapas independientes (constitución, principio II): decimación de 60 a 30 fps, extracción de
los 48 keypoints con Apple Vision (esquema de keypoints 1.0.0) y normalización según la
especificación única compartida con la app (`signai-architecture/contracts/normalizacion.md`).
Diseño en `specs/004-preprocesamiento-corpus/`.

Las versiones entran en la vigencia de cada resultado, en el manifiesto y en la aprobación del lote
(research R-16): si cambia cualquiera, el resultado deja de ser vigente.
"""

VERSION_CADENA = "signai-ml/preproceso/1"
VERSION_EXTRACTOR = "signai-ml/keypoints/1"
ESQUEMA = "1.0.0"
NORMALIZACION = "2.0.0"
