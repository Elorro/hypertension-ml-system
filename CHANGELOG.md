# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y el versionado sigue [SemVer](https://semver.org/lang/es/).

---

## [No publicado]

### Añadido

- **Entrenamiento sobre datos reales** — `src/train_cardio_real.py` entrena los
  5 algoritmos sobre `data/real/cardio/cardio_train.csv` (68.678 filas tras
  limpieza) en dos experimentos: A′ (riesgo cardiovascular, con ablación de
  `ap_hi`/`ap_lo`) y B1 (hipertensión con la presión arterial excluida de las
  features). Persiste `models/dt1_manifest.json` con métricas, curvas de
  calibración, IC bootstrap, hashes de artefactos y versiones del entorno.
- **`scripts/audit_cardio_leakage.py`** — criterio de aceptación de DT-1: busca
  reglas deterministas (umbral univariado y árbol CART propio) sobre el dataset
  real, con control positivo. Sin dependencias de ML: solo numpy y pandas.
- **`docs/DT1_RESULTS.md`** — resultados completos y sus límites.
- Objetivos `make train-real` y `make audit-real`.
- **Entorno reproducible** (`f508c85`) — `requirements.lock.txt` con el grafo completo
  y numpy/pandas/scikit-learn/xgboost fijados a las versiones del manifiesto de DT-1;
  `scripts/verify_env.py` y `make verify-env` como compuerta (versiones, sha256 de los
  `.pkl`, carga y predicción).
- **`src/cardio_features.py`** — fuente única de features, derivación y cotas de DT-1,
  compartida por entrenamiento y API. **`src/artefactos.py`** — verificación de
  versiones (perfiles `entorno` y `servicio`) y sha256 de los artefactos (`86a5c29`).
- **API sobre los modelos reales** (`6e59035`): `POST /v1/riesgo-cardiovascular` (A′,
  probabilidad + clase con umbral 0,5 explícito), `POST /v1/hipertension-sin-pa` (B1,
  experimental, solo probabilidad), sus variantes `/lote` (hasta 1.000 filas, errores por
  índice), `GET /health` y `GET /v1/modelos`. Entrada en unidades humanas validada
  contra el dominio de entrenamiento; B1 rechaza la presión arterial con 422. Carga en
  lifespan con verificación de versiones y sha256: si no coinciden, no arranca.
- `title` y `x-etiquetas` por campo en el esquema de entrada (`73c84ea`).
- **Dashboard** como cliente HTTP puro de la API nueva (`a9b89ee`): espera a la API con
  reintentos, formularios y límites desde `/v1/modelos`, lote por CSV con errores por fila.
- **Suite de tests** (`86a5c29`, `6e59035`, `a9b89ee`): contrato de API y dashboard en
  CI con un doble del modelo; integración con los `.pkl` y el CSV reales, que se salta
  con motivo visible si faltan.
- `requirements-serve.txt` y `requirements-dashboard.txt` (instalaciones mínimas);
  objetivos `make serve-api` y `make serve-dashboard`.
- **DT-4 · Protocolo preregistrado** (`0d8d4c9`, `docs/DT4_PROTOCOL.md`), commiteado
  antes de entrenar: selección por log-loss en validación cruzada de 5 folds sobre el
  train, regla de 1 EE con orden de simplicidad, una evaluación en test por experimento
  y dos ablaciones de A′ preregistradas.
- **`src/train_cardio_dt4.py`** (`ad518f2`) y objetivo `make train-dt4`; manifiesto de la
  corrida `models/dt4_manifest.json` (`41572ed`), con resultados por fold, regla
  aplicada, `pip freeze` y joblib. Resultados en `docs/DT4_RESULTS.md`.
- `make verify-env-dt1` y `python -m scripts.verify_env --manifest …` (`a7bf9ef`).
- `/v1/modelos` informa cómo se seleccionó cada modelo (`seleccion`) (`a7bf9ef`).

### Cambiado

- La versión del proyecto pasa a 2.0.0 (`pyproject.toml`, publicada en `/health`), sin
  release.
- `python -m src.train_cardio_real` y `python -m scripts.verify_env` sustituyen a la
  ejecución directa de esos archivos.
- Python 3.14 como versión de referencia (`f508c85`).
- **Modelos servidos: los de DT-4** (`a7bf9ef`): A′ RandomForest(max_depth=12,
  min_samples_leaf=20) y B1 RandomForest(max_depth=8, min_samples_leaf=100). La API y
  `verify_env` usan `models/dt4_manifest.json` por defecto; el de DT-1 queda como
  registro. RSS de la API 352 → 202 MiB; `.pkl` servidos 130 → 32 MB.
- La nota de métricas y la advertencia de B1 servidas ya no hablan de «ganador» elegido
  sobre el test; el dashboard muestra el AUC de B1 con tres decimales (`a7bf9ef`).
- Todo el repositorio formateado con `ruff format` sin cambio de comportamiento
  (`0e75317`); el job de lint de CI pasa a ser bloqueante (`d6d1b0f`).
- El dataset de Kaggle deja de redistribuirse: estuvo versionado, se detectó que su
  licencia es desconocida y se purgó de toda la historia con `git filter-repo`; se
  descarga de la fuente. `.gitignore` ignora `data/real/` completo (`4b3b5d4`).

### Eliminado

- **Cambio incompatible:** `POST /predecir` y su contrato sintético de 4 clases y
  15 features (`6e59035`). `api/` ya no carga `models/modelo_*.pkl`; el pipeline
  sintético se conserva solo como evidencia del leakage.

### Corregido

- **DT-1 · Target leakage** — el pipeline de entrenamiento deja de depender del
  dataset sintético cuyo target era función determinista de las features. La mejor
  regla determinista sobre las nuevas features supera al baseline en +2,28 pp
  (límite de aceptación: 5 pp).
- En el pipeline real, el `StandardScaler` se ajusta solo sobre train (DT-2), el
  split es estratificado (DT-3 parcial) y el SVM usa `CalibratedClassifierCV` en
  lugar de `SVC(probability=True)`, deprecado (DT-17 parcial).
- **DT-7 · Validación de rangos** y **DT-9 · Endpoint batch**, en la API nueva
  (`6e59035`). **DT-11 · Versiones fijadas** (`f508c85`).
- **DT-3 · Validación cruzada** y **DT-4 · Selección separada de la evaluación**
  (`0d8d4c9`, `ad518f2`, `41572ed`, `a7bf9ef`). Limitación declarada: el test ya se
  había observado en DT-1.
- La ablación de A′ de DT-1 (Δ AUC 0,1103) estaba inflada por comparar contra una SVM
  elegida por macro-F1; la ablación controlada de DT-4 da 0,0991 [0,0919 · 0,1061].
- Documentación: la regla `if` deja de describirse con un conteo de líneas que no
  cuadraba con ningún bloque real, y el 91,1 % del dataset sintético se distingue de
  la auditoría sobre datos reales (`c3814b2`).

### Por hacer

Ver [docs/ROADMAP.md](docs/ROADMAP.md). Pendientes: la parte de DT-5 que falta
(decidir si el pipeline heredado de la raíz se migra o se elimina), la cobertura de
DT-6 en CI, el análisis de equidad (DT-15) y la decisión de despliegue.

---

## [1.0.0] — 2026-01-05

Primera versión funcional del sistema end-to-end.

### Añadido

- **Generación de datos** — `src/generate_dataset.py` produce 50.000 registros
  sintéticos con 15 features clínicas y de estilo de vida, semilla fija (42).
- **Entrenamiento comparativo** — `src/train_classical_models.py` entrena Regresión
  Logística, SVM-RBF, Árbol de Decisión, Random Forest y XGBoost, y selecciona el
  mejor por macro-F1.
- **Servicio de inferencia** — `api/main.py` (FastAPI) expone `GET /` y
  `POST /predecir` con validación Pydantic y documentación OpenAPI automática.
- **Dashboard** — `app/dashboard.py` (Streamlit) con inferencia individual por
  formulario e inferencia masiva por carga de CSV.
- **EDA sobre datos reales** — `notebooks/EDA_cardiovascular_real.ipynb` analiza el
  dataset Cardiovascular Disease de Kaggle (70.000 pacientes).
- **Avisos médicos** en el servicio y el dashboard.

### Documentación

- README con arquitectura, instalación, uso y limitaciones declaradas.
- `docs/ARCHITECTURE.md` — módulos, contratos y decisiones de diseño.
- `docs/LEAKAGE_ANALYSIS.md` — auditoría del target leakage con evidencia reproducible.
- `docs/MODEL_CARD.md` — model card con uso previsto, límites y consideraciones éticas.
- `docs/DATA.md` — diccionario de variables, procedencia y licencias.
- `docs/API.md` — referencia de endpoints.
- `docs/ROADMAP.md` — deuda técnica priorizada en cuatro fases.
- `CONTRIBUTING.md`, `LICENSE` (MIT), `CITATION.cff`.
- `scripts/verify_leakage.py` — auditoría ejecutable del defecto metodológico.
- `Makefile` con los flujos de trabajo habituales.

### Corregido

- **`src/train_classical_models.py` no ejecutaba con scikit-learn ≥ 1.7**: usaba
  `multi_class="multinomial"`, parámetro eliminado en esa versión. Detectado por CI
  con scikit-learn 1.9. Con el solver por defecto (lbfgs) el ajuste ya es
  multinomial, así que el comportamiento no cambia.
- `except:` desnudo en el cálculo de ROC-AUC sustituido por `except ValueError` con
  aviso explícito (DT-8).
- Imports sin usar y desordenados en `api/`, `app/` y `src/`; variable de bucle sin
  usar en `app/dashboard.py`. `ruff check` pasa limpio sobre el pipeline activo.
- `requirements.txt`: eliminadas `lightgbm` y `catboost` (no se importan en ninguna
  parte del código) y añadida `requests`, que el dashboard usa y faltaba declarar.
- `.gitignore` ampliado para cubrir caches de test, notebooks, secretos y artefactos.

### Limitaciones conocidas

- **Target leakage** — la etiqueta es una función determinista de las features. Las
  métricas no miden capacidad predictiva. Ver `docs/LEAKAGE_ANALYSIS.md`.
- **Escalado antes del split** — el `StandardScaler` se ajusta sobre el dataset
  completo (DT-2).
- **Split no estratificado y sin validación cruzada** (DT-3, DT-4).
- **Dos pipelines paralelos** — los scripts de la raíz son una implementación previa
  e incompatible; `train_models.py` además no ejecuta con scikit-learn ≥ 1.4 (DT-5).
- **Sin tests automatizados** (DT-6).
- **Validación de tipos pero no de rangos** en el servicio (DT-7).

[No publicado]: https://github.com/Elorro/hypertension-ml-system/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Elorro/hypertension-ml-system/releases/tag/v1.0.0
