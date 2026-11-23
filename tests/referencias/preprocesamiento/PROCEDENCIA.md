# Procedencia de las referencias de preprocesamiento (TT-12)

- `normalizacion-casos.json`: copia exacta de
  `signai-architecture/contracts/normalizacion-casos.json`, los casos de referencia del contrato de
  la normalización (N-8). Entradas **sintéticas**, sin datos de señantes; las salidas las calcula la
  implementación de referencia (`signai_ml.preproceso.normalizacion`) en `float64` y se guardan
  redondeadas a `float32`.
- `generar_casos.py`: el generador. Semilla fija; volver a ejecutarlo con la misma implementación da
  el mismo fichero.

`tests/preprocesamiento/test_paridad_normalizador.py` comprueba que la huella de la copia es la que
declara el contrato. Si el contrato cambia de versión, se regenera aquí y se publica allí, en ese
orden, y se actualiza la huella en la prueba.
