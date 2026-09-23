# Arquitectura

## Visión general

El sistema es un pipeline lineal con dos consumidores finales. No hay orquestador ni
base de datos: la comunicación entre etapas ocurre a través del sistema de archivos
(CSV, artefactos `.pkl` y su manifiesto), y entre los servicios finales vía HTTP.

```
data/real/cardio/cardio_train.csv ──► train_cardio_real.py ──► models/dt1_*.pkl
                                              ▲                 models/dt1_manifest.json
                     src/cardio_features.py ──┤                          │
                                              ▼                          ▼
                                    api/ (FastAPI :8000) ◄── verifica con src/artefactos.py
                                              ▲
                                              │ HTTP: /health, /v1/modelos, /v1/…
                                              │
                                    app/dashboard.py (Streamlit :8501)

Evidencia del leakage (fuera del servicio):
generate_dataset.py ──► data/raw/*.csv ──► train_classical_models.py ──► models/modelo_*.pkl
```

## Componentes

### `src/cardio_features.py` — fuente única de features y dominio

Orden de features por experimento, derivación de `age_years` (`age / 365.25`) y `bmi`
(`weight / (height/100)²`), cotas de limpieza (talla, peso, IMC), rangos de presión
plausible y rango de edad de servicio. Solo biblioteca estándar: las funciones sirven
igual para `pandas.Series` (entrenamiento) y para `float` (API). Lo importan
`src/train_cardio_real.py` y `api/`; `scripts/audit_cardio_leakage.py` reimplementa la
limpieza a propósito para ser una verificación independiente.

### `src/train_cardio_real.py` — entrenamiento de DT-1

Entrena los 5 algoritmos sobre el dataset real en dos experimentos (A′ con y sin
presión arterial; B1 sin presión), con split estratificado 80/20 y `StandardScaler`
ajustado solo sobre train. Persiste `models/dt1_<experimento>__{modelo,scaler}.pkl` y
`models/dt1_manifest.json` (métricas, IC, calibración, sha256, versiones). Se ejecuta
como módulo: `python -m src.train_cardio_real`. Resultados en
[DT1_RESULTS.md](DT1_RESULTS.md).

### `src/artefactos.py` — verificación de artefactos

Compara versiones instaladas contra el manifiesto con dos perfiles (`entorno` para
desarrollo y reentreno; `servicio` para la API: Python por major.minor, scikit-learn,
numpy y joblib exactos), comprueba el sha256 de cada `.pkl`, lo carga tratando los
avisos de versión como error y verifica la identidad del scaler (`mean_`/`scale_`).
Lo usan `scripts/verify_env.py` y la API al arrancar.

### `src/generate_dataset.py` — generación de datos sintéticos (evidencia)

Produce `data/raw/dataset_hipertension_sintetico.csv` con 50.000 filas y 16 columnas
(15 features + target). Las variables se muestrean de distribuciones normales o
binomiales independientes, con recorte (`clip`) a rangos fisiológicamente plausibles.

La etiqueta se deriva de umbrales sobre `PAS`/`PAD` más ajustes deterministas por
factores de riesgo. **Esta derivación es el origen del target leakage documentado en
[LEAKAGE_ANALYSIS.md](LEAKAGE_ANALYSIS.md).**

- Semilla fija (`seed=42`) → generación reproducible.
- Salida: CSV sin índice, separador coma.

### `src/train_classical_models.py` — entrenamiento sintético (evidencia)

Entrena cinco clasificadores sobre el CSV sintético y persiste cada uno más el
`StandardScaler` ajustado. Ningún servicio carga ya estos artefactos: se conservan
como evidencia reproducible del leakage.

Flujo:

1. Carga el CSV y separa `X = df.drop("Diagnostico")`, `y = df["Diagnostico"]`.
2. Ajusta `StandardScaler` sobre `X` completo y lo guarda en `models/scaler.pkl`.
3. `train_test_split` 80/20 con `random_state=42`.
4. Entrena y evalúa cada modelo (accuracy, macro-F1, ROC-AUC one-vs-rest).
5. Selecciona el de mayor macro-F1 y escribe su nombre en `models/mejor_modelo.txt`.

> **Defectos conocidos en este flujo** (pasos 2 y 3): el escalado ocurre antes del
> split y el split no está estratificado. Ver
> [LEAKAGE_ANALYSIS.md](LEAKAGE_ANALYSIS.md) §4.

Artefactos producidos en `models/`:

| Archivo | Contenido |
|---------|-----------|
| `scaler.pkl` | `StandardScaler` ajustado |
| `modelo_logreg.pkl` · `modelo_svm.pkl` · `modelo_tree.pkl` · `modelo_rf.pkl` · `modelo_xgb.pkl` | Clasificadores entrenados |
| `mejor_modelo.txt` | Nombre lógico del ganador (p. ej. `XGBoost`) |

### `api/` — servicio de inferencia (FastAPI)

Expone A′ (`/v1/riesgo-cardiovascular`) y B1 (`/v1/hipertension-sin-pa`), cada uno con
su `/lote`, más `/health` y `/v1/modelos`. Contrato completo en [API.md](API.md).

- **Carga en lifespan, una vez:** `api/servicio.py` verifica versiones (perfil
  `servicio`) y sha256 con `src/artefactos.py` y carga modelo y scaler de cada
  experimento. Si algo no coincide, la API no arranca. Rutas por `MODEL_DIR` y
  `MANIFEST_PATH`.
- **Validación de entrada:** `api/schemas.py`, con `extra="forbid"` (B1 rechaza
  `ap_hi`/`ap_lo`) y las cotas de `src/cardio_features.py`. El IMC se calcula en el
  servidor.
- **Contrato de features:** la matriz se construye en el orden de `features_orden`
  del manifiesto. Un test compara bit a bit la matriz de la API con la del
  entrenamiento sobre todas las filas del CSV con presión plausible.

### `app/dashboard.py` — interfaz (Streamlit)

Cliente HTTP puro: no importa scikit-learn ni carga modelos. Lee `API_URL`, espera a
la API con reintentos sobre `/health`, construye formularios y etiquetas solo desde
`/v1/modelos` y usa los endpoints `/lote` para el CSV (una llamada por bloque de hasta
1.000 filas, errores por fila).

## Convenciones de datos

Orden de features de los modelos servidos (`src/cardio_features.py`, registrado en
`features_orden` del manifiesto):

```python
FEATURES_B1 = [
    "age_years",
    "gender",
    "height",
    "weight",
    "bmi",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]
FEATURES_A_CON_PA = FEATURES_B1 + ["ap_hi", "ap_lo"]
```

Orden del dataset sintético (evidencia del leakage):

```python
[
    "Edad",
    "Sexo",
    "Peso",
    "Talla",
    "IMC",
    "PAS",
    "PAD",
    "PAM",
    "Frec_Card",
    "Colesterol",
    "Glucosa",
    "Tabaquismo",
    "Ejercicio",
    "Estres",
    "Herencia_HTA",
]
```

Variables derivadas del dataset sintético, calculadas en el generador:

$$\text{IMC} = \frac{\text{Peso}}{\text{Talla}^2} \qquad
  \text{PAM} = \frac{\text{PAS} + 2\cdot\text{PAD}}{3}$$

## Pipeline heredado (scripts de la raíz)

`data_pipeline.py`, `train_models.py`, `model_utils.py`, `streamlit_app.py` y
`chatbot_cli.py` conforman una **implementación anterior e incompatible** que
permanece en el repositorio pero no está integrada.

Incompatibilidades con el pipeline sintético de `src/`:

| Aspecto | Pipeline sintético (`src/`) | Pipeline heredado (raíz) |
|---------|--------------------------|--------------------------|
| Columna de target | `Diagnostico` | `HTA_Nivel` |
| Nombre de la variable de estrés | `Estres` | `Estrés` (con tilde) |
| Ruta del dataset | `data/raw/dataset_hipertension_sintetico.csv` | `data/hypertension_synthetic.csv` |
| Formato de artefacto | `.pkl` sueltos + `scaler.pkl` | `best_model.joblib` (bundle) |
| Modelos | 5 clásicos | 4 clásicos + red neuronal Keras |

Además, `train_models.py` **no ejecuta con scikit-learn ≥ 1.4**: usa el parámetro
`base_estimator` de `CalibratedClassifierCV`, renombrado a `estimator` en la 1.2 y
eliminado en la 1.4. También pasa `multi_class` a `LogisticRegression`, deprecado
desde la 1.5.

La resolución (unificar o eliminar) está priorizada en [ROADMAP.md](ROADMAP.md).

## Decisiones de diseño

**Por qué el dashboard consume HTTP en vez de importar el modelo.** Separar
inferencia de presentación permite escalar, versionar y monitorear el modelo de
forma independiente, y hace que el mismo servicio sirva a otros clientes. El costo
es la latencia por petición, que el endpoint `/lote` acota en el modo masivo.

**Por qué los artefactos no se versionan.** Los `.pkl` son binarios grandes que
git no puede diferenciar; versionarlos infla el historial de forma irreversible.
El repositorio versiona el *código que los produce* y `models/dt1_manifest.json`, con
el sha256 de cada artefacto; la reproducibilidad se apoya en la semilla fija y en el
entorno fijado por `requirements.lock.txt`. El costo es que un clon limpio requiere
`make train-real` (~51 min) antes de arrancar el servicio.

**Por qué se persiste el scaler junto a los modelos.** El escalado forma parte de la
función de inferencia, no del entrenamiento. Reajustarlo en producción produciría
predicciones silenciosamente incorrectas. Empaquetarlo todo en un `Pipeline` de
scikit-learn sería más robusto — está en el ROADMAP.
