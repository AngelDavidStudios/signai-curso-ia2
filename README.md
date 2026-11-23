# SignAI · Reconocimiento de la Lengua de Señas Ecuatoriana

Código del proyecto semestral de **Inteligencia Artificial II** (UDLA, ISWZ3402): un KPoseFormer que
reconoce 100 señas de la LSEC a partir de los puntos clave que extrae Apple Vision. El modelo se
ejecuta *on-device* en un iPhone.

Este repositorio contiene lo que el artículo describe en Métodos y Resultados: el preprocesamiento,
la partición por grabación, el entrenamiento con selección por rotación, la evaluación de aceptación
y la exportación a Core ML. También incluye las métricas exportadas de todos los experimentos.

> **Datos.** El corpus muestra a personas identificables, así que no se publican ni los vídeos ni
> los puntos clave. Con las métricas de `resultados/` se reproducen todas las tablas y figuras.
> Entrenar de nuevo exige el corpus.

## Estructura

| Carpeta | Contenido |
| --- | --- |
| `src/signai_ml/preproceso/` | Decimación de 60 a 30 fps, extracción de 48 puntos clave con Apple Vision, normalización y remuestreo a la ventana `(75, 96)` |
| `src/signai_ml/particion/` | Partición por grabación entera: una grabación de cada seña a prueba y las otras dos a los pliegues A y B |
| `src/signai_ml/reconocedor/` | KPoseFormer, aumento de datos, bucle de entrenamiento, métricas, barrido por rondas y selección por rotación |
| `src/signai_ml/aceptacion/` | Evaluación única sobre la prueba, regla de aceptación (≥ 85 % Top-1), registro en MLflow y exportación a Core ML con paridad |
| `src/signai_ml/copia/` | Copia de seguridad cifrada (restic) que el ciclo automático ejecuta entre rondas |
| `herramientas/keypoints-vision/` | Extractor de puntos clave en Swift: el mismo detector y esquema que la app |
| `config/reconocedor/` | Las rondas de los barridos en TOML, tal como se lanzaron (`b01`: partición por clip; `b02`: partición por grabación) |
| `resultados/` | Métricas exportadas de MLflow y los guiones que reproducen las tablas y la Figura 2 |
| `tests/` | 812 pruebas con datos sintéticos; ninguna lee el corpus |

## Requisitos

- macOS en Apple Silicon (el entrenamiento usa el backend MPS de PyTorch y la extracción usa Apple Vision).
- [`uv`](https://docs.astral.sh/uv/), Swift 6 (Xcode) y ffmpeg ≥ 8.1 (`brew install ffmpeg`).
- Para la copia entre rondas: restic ≥ 0.19 (`brew install restic`).

```sh
git clone https://github.com/angeldev-studios/signai-curso-ia2.git
cd signai-curso-ia2
uv sync                       # Python 3.13, PyTorch 2.7.0, coremltools 9.0, MLflow 3.16 y DVC 3.67 (uv.lock)
uv run pytest                 # 812 pruebas, unos 2 minutos
```

## Reproducir las tablas y la Figura 2

No necesita el corpus:

```sh
uv run python resultados/tablas.py
```

El guion agrupa las corridas de `resultados/corridas.csv` y `resultados/epocas.csv` por configuración
(3 semillas × 2 rotaciones). Toma la curva media del Top-1 de validación, elige la época del máximo
y calcula:

| Salida | Qué reproduce |
| --- | --- |
| `figura2.csv` | La mejor configuración de cada ronda completa del barrido b02 (Figura 2) |
| Tabla «línea base frente a la mejor» | Top-1, F1 macro, acierto y errores al 95 %, diferencia entre rotaciones y *t* pareada |
| Tabla de ablación | El efecto de cada aumento de datos en la ronda 1 |
| `configuraciones.csv` | Las cifras de todas las configuraciones del barrido |

Los CSV se generan desde la base local de MLflow, que el guion abre en modo de solo lectura:

```sh
uv run python resultados/exportar_mlflow.py ~/signai-data/mlflow/mlflow.db
```

## Cadena completa (con el corpus)

Cada paso se ejecuta una vez, en este orden. Los datos van a `datos/` y los modelos a `modelos/`, y se
versionan con DVC. El seguimiento de experimentos vive en `$SIGNAI_DATA_ROOT/mlflow` (por defecto
`~/signai-data`).

```sh
# 1. Extractor de puntos clave (una vez)
swift build -c release --package-path herramientas/keypoints-vision

# 2. Preprocesamiento: lote de validación, aprobación y corpus completo → datos/ventanas/ (N, 75, 96)
uv run signai-preproceso lote
uv run signai-preproceso aprobar-lote
uv run signai-preproceso procesar
uv run signai-preproceso verificar

# 3. Partición por grabación (semilla 372): se genera una sola vez
uv run signai-particion generar
uv run signai-particion verificar

# 4. Barrido por rondas con selección por rotación, hasta el cierre por estancamiento
uv run signai-reconocedor automatico b02
uv run signai-reconocedor resumen b02

# 5. Modelo final (A ∪ B durante E* épocas, sin validación)
uv run signai-reconocedor candidato b02

# 6. Una sola consulta a la prueba, registro y exportación a Core ML con paridad
uv run signai-aceptacion evaluar
uv run signai-aceptacion registrar
uv run signai-aceptacion exportar
```

Cada corrida queda en MLflow con su configuración, semilla, métricas por época, commit y versiones
de las bibliotecas, además de las huellas SHA-256 de las ventanas y de la partición. Las lecturas de
la partición son distintas para cada uso: la selección y el modelo final no tienen acceso a la
prueba.

## Protocolo de evaluación

| Lado | Contenido | Uso |
| --- | --- | --- |
| Prueba | 1 grabación por seña (2.999 clips) | Una sola consulta al final; se acepta si `aciertos × 100 ≥ 85 × total` |
| Pliegue A | 1 grabación por seña (2.999 clips) | Rotación 1: entrena con A y valida con B |
| Pliegue B | 1 grabación por seña (2.998 clips) | Rotación 2: entrena con B y valida con A |

La partición del primer modelo (`b01`) repartía clips sueltos, así que la prueba contenía repeticiones
de las mismas grabaciones que el entrenamiento y daba un 100 % de Top-1 que no medía generalización.
La partición por grabación corrige esa fuga.
