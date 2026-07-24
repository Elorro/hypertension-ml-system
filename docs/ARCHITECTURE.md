# Arquitectura

## Visión general

El sistema es un pipeline lineal de cuatro etapas con dos consumidores finales.
No hay orquestador ni base de datos: la comunicación entre etapas ocurre a través
del sistema de archivos (CSV y artefactos `.pkl`), y entre los servicios finales
vía HTTP.

```
generate_dataset.py ──► data/raw/*.csv ──► train_classical_models.py ──► models/*.pkl
                                                                              │
                                                        ┌─────────────────────┘
                                                        ▼
                                            api/main.py (FastAPI :8000)
                                                        ▲
                                                        │ HTTP POST /predecir
                                                        │
                                            app/dashboard.py (Streamlit :8501)
```

## Componentes

### `src/generate_dataset.py` — generación de datos

Produce `data/raw/dataset_hipertension_sintetico.csv` con 50.000 filas y 16 columnas
(15 features + target). Las variables se muestrean de distribuciones normales o
binomiales independientes, con recorte (`clip`) a rangos fisiológicamente plausibles.

La etiqueta se deriva de umbrales sobre `PAS`/`PAD` más ajustes deterministas por
factores de riesgo. **Esta derivación es el origen del target leakage documentado en
[LEAKAGE_ANALYSIS.md](LEAKAGE_ANALYSIS.md).**

- Semilla fija (`seed=42`) → generación reproducible.
- Salida: CSV sin índice, separador coma.

### `src/train_classical_models.py` — entrenamiento y selección

Entrena cinco clasificadores sobre el CSV sintético y persiste cada uno más el
`StandardScaler` ajustado.

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
| `scaler.pkl` | `StandardScaler` ajustado — **obligatorio** en inferencia |
| `modelo_logreg.pkl` · `modelo_svm.pkl` · `modelo_tree.pkl` · `modelo_rf.pkl` · `modelo_xgb.pkl` | Clasificadores entrenados |
| `mejor_modelo.txt` | Nombre lógico del ganador (p. ej. `XGBoost`) |

### `api/main.py` — servicio de inferencia (FastAPI)

Expone el modelo ganador como servicio HTTP.

- **Carga en tiempo de import:** lee `scaler.pkl`, `mejor_modelo.txt` y el `.pkl`
  correspondiente al arrancar el módulo. Si falta cualquiera, el proceso falla al
  iniciar. Es deliberado (fallo temprano y ruidoso) pero implica que
  `src/train_classical_models.py` debe haberse ejecutado antes.
- **Mapeo nombre → archivo:** el diccionario `nombre_a_archivo` traduce el contenido
  de `mejor_modelo.txt` a la ruta del artefacto. Añadir un modelo nuevo al
  entrenamiento exige actualizar también este diccionario.
- **Validación de entrada:** el modelo Pydantic `Paciente` valida tipos, no rangos.
  Un `PAS` de 900 se acepta y produce una predicción sin sentido.
- **Contrato de features:** el orden del array construido en `/predecir` debe
  coincidir **exactamente** con el orden de columnas del CSV de entrenamiento. Es un
  acoplamiento implícito y frágil, sostenido solo por un comentario en el código.

### `app/dashboard.py` — interfaz (Streamlit)

Cliente del servicio HTTP. No carga modelos: todo pasa por `POST /predecir`.

- **Pestaña individual:** formulario de 13 campos; calcula `IMC` y `PAM` derivados y
  envía el payload completo de 15 features.
- **Pestaña masiva:** carga un CSV, valida columnas obligatorias, deriva `IMC`/`PAM`
  si faltan, y **emite una petición HTTP por fila**. Para archivos grandes esto es
  lento; un endpoint batch está en el [ROADMAP](ROADMAP.md).

## Convenciones de datos

El orden canónico de features, compartido por entrenamiento e inferencia:

```python
["Edad", "Sexo", "Peso", "Talla", "IMC", "PAS", "PAD", "PAM",
 "Frec_Card", "Colesterol", "Glucosa", "Tabaquismo", "Ejercicio",
 "Estres", "Herencia_HTA"]
```

Variables derivadas, calculadas idénticamente en dashboard y generador:

$$\text{IMC} = \frac{\text{Peso}}{\text{Talla}^2} \qquad
  \text{PAM} = \frac{\text{PAS} + 2\cdot\text{PAD}}{3}$$

## Pipeline heredado (scripts de la raíz)

`data_pipeline.py`, `train_models.py`, `model_utils.py`, `streamlit_app.py` y
`chatbot_cli.py` conforman una **implementación anterior e incompatible** que
permanece en el repositorio pero no está integrada.

Incompatibilidades con el pipeline activo:

| Aspecto | Pipeline activo (`src/`) | Pipeline heredado (raíz) |
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
es la latencia por petición, notoria en el modo masivo.

**Por qué los artefactos no se versionan.** Los `.pkl` son binarios grandes que
git no puede diferenciar; versionarlos infla el historial de forma irreversible.
El repositorio versiona el *código que los produce*, y la reproducibilidad se
garantiza con la semilla fija. El costo es que un clon limpio requiere `make setup`
antes de arrancar el servicio.

**Por qué se persiste el scaler junto a los modelos.** El escalado forma parte de la
función de inferencia, no del entrenamiento. Reajustarlo en producción produciría
predicciones silenciosamente incorrectas. Empaquetarlo todo en un `Pipeline` de
scikit-learn sería más robusto — está en el ROADMAP.
