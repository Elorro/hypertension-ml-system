# Riesgo cardiovascular y estimación de hipertensión sin medir la presión: ML sobre datos reales con auditoría de target leakage

Sistema end-to-end de machine learning sobre el dataset real *Cardiovascular Disease*
(Kaggle): entrenamiento comparativo de cinco algoritmos, servicio REST que expone
**probabilidades** de riesgo cardiovascular y de hipertensión estimada sin medir la
presión, y dashboard interactivo. Conserva, como evidencia reproducible, el pipeline
sintético original y su target leakage.

![Python](https://img.shields.io/badge/python-3.14-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-prototipo%20educativo-orange)

---

## ⚠️ Léase primero: qué es y qué NO es este proyecto

Este repositorio es una **demostración de ingeniería de sistemas de ML**, no un
predictor clínico válido. La distinción no es un tecnicismo:

> **El modelo original no predecía hipertensión. Reaprendía una regla que ya conocíamos.**

El pipeline original entrena sobre un dataset sintético cuya etiqueta se construye
aplicando un umbral determinista sobre la presión sistólica (`PAS`) y diastólica
(`PAD`) — las mismas variables que después se entregan al modelo como entrada. Es un
caso de libro de **target leakage**: el modelo puede alcanzar métricas casi perfectas
simplemente reproduciendo el `if` que generó la etiqueta. Ese pipeline sigue en el
repositorio como evidencia, fuera del camino de servicio.

Está verificado numéricamente y el análisis es reproducible:

```bash
python scripts/verify_leakage.py
```

Detalle completo, evidencia y consecuencias en **[docs/LEAKAGE_ANALYSIS.md](docs/LEAKAGE_ANALYSIS.md)**.

**Estado actual del defecto: corregido en el entrenamiento y en el servicio.** Desde
el 2026-09-17 existe un segundo pipeline,
[`src/train_cardio_real.py`](src/train_cardio_real.py), que entrena sobre el dataset
real de Kaggle (68.678 pacientes) excluyendo la presión arterial de las features. Ahí
la pregunta es genuina y la respuesta, modesta: **AUC 0,6955 [0,6866 · 0,7049]** para
estimar el estado hipertensivo sin medirlo, con una accuracy apenas +3,35 pp sobre el
baseline de clase mayoritaria (modelo seleccionado en DT-4). Verificable con:

```bash
python scripts/audit_cardio_leakage.py    # criterio de aceptación de DT-1
```

Resultados vigentes en **[docs/DT4_RESULTS.md](docs/DT4_RESULTS.md)** (selección por
validación cruzada con protocolo preregistrado); los de la primera corrida, históricos,
en [docs/DT1_RESULTS.md](docs/DT1_RESULTS.md). El servicio FastAPI y el dashboard sirven
**solo** modelos reales (A′ con presión arterial y B1 sin ella), los de DT-4 desde
`a7bf9ef`; el contrato sintético `/predecir` se eliminó en `6e59035`.

**Lo que sí demuestra este repositorio, y demuestra bien:**

- Pipeline reproducible de datos → entrenamiento → selección de modelo → artefactos.
- Selección de modelo preregistrada (`docs/DT4_PROTOCOL.md`, commiteado antes de
  entrenar): 25 candidatos por experimento, log-loss en validación cruzada de 5 folds
  sobre el train, regla de un error estándar con orden de simplicidad declarado, y el
  test evaluado una sola vez sobre el seleccionado.
- Servicio de inferencia con FastAPI, esquema validado con Pydantic y OpenAPI automático:
  validación del dominio de entrenamiento, contrato anti-leakage (B1 rechaza la
  presión con 422) y verificación de versiones y sha256 de los artefactos al arrancar.
- Dashboard Streamlit con inferencia individual y masiva vía CSV, cliente HTTP puro.
- EDA sobre un dataset clínico real (70.000 pacientes, Kaggle).
- Auditoría de leakage con **control positivo**, y una ablación controlada con
  bootstrap pareado que cuantifica cuánto vale medir la presión arterial:
  Δ AUC = 0,0991 [0,0919 · 0,1061] (DT-4; el 0,1103 de DT-1 estaba inflado por comparar
  contra una SVM elegida por macro-F1).

**Aviso médico:** herramienta con fines educativos. No constituye diagnóstico,
consejo médico ni sustituye la valoración de un profesional de la salud.

---

## Arquitectura

```
 data/real/cardio/cardio_train.csv          src/cardio_features.py
               │                          (features, derivación, cotas)
               ▼                                 │          │
 src/train_cardio_real.py ◄──────────────────────┘          │
               │                                            │
               ▼                                            ▼
 models/dt1_*.pkl + dt1_manifest.json ──► api/main.py (FastAPI) ◄── HTTP ── app/dashboard.py
                                           verifica versiones y         (Streamlit, cliente
                                           sha256 al arrancar            HTTP puro)

 Evidencia del leakage, fuera del servicio:
 src/generate_dataset.py ──► src/train_classical_models.py ──► models/modelo_*.pkl
```

Descripción detallada de módulos, contratos y flujo de datos en
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

> **Nota sobre duplicación:** los scripts de la raíz son una versión anterior e
> incompatible del pipeline (`src/` + `api/` + `app/` es el activo). Resolverla —migrar
> o eliminar— sigue pendiente en [docs/ROADMAP.md](docs/ROADMAP.md) (DT-5).

---

## Instalación

Requiere Python 3.14 (referencia: 3.14.6). Los `.pkl` de DT-1 son objetos serializados
de scikit-learn, así que el entorno se instala desde el lock, con versiones exactas.

```bash
git clone https://github.com/Elorro/hypertension-ml-system.git
cd hypertension-ml-system

python3.14 -m venv .venv
source .venv/bin/activate        # zsh/bash · Windows: .venv\Scripts\activate

pip install -r requirements.lock.txt   # entorno completo (entrenamiento, tests, servicio)
```

Instalaciones mínimas: `requirements-serve.txt` (solo la API, sin pandas ni xgboost) y
`requirements-dashboard.txt` (solo el dashboard, sin scikit-learn).

## Uso rápido

Los artefactos entrenados (`models/*.pkl`) y el dataset real **no se versionan**. Con
`data/real/cardio/cardio_train.csv` descargado (ver [docs/DATA.md](docs/DATA.md)):

```bash
make train-real       # entrena DT-1 y genera models/dt1_*.pkl (~51 min)
make verify-env       # versiones + sha256 + carga y predicción de los .pkl
make serve-api        # servicio en http://127.0.0.1:8000  (docs en /docs)
make serve-dashboard  # dashboard en http://localhost:8501 (API_URL, por defecto :8000)
```

Sin `make`:

```bash
python -m src.train_cardio_real
uvicorn api.main:app --host 127.0.0.1 --port 8000
API_URL=http://127.0.0.1:8000 streamlit run app/dashboard.py
```

> El servicio carga los modelos una vez al arrancar, tras verificar versiones y sha256
> contra `models/dt1_manifest.json`. Si falta un `.pkl` o no coincide, **no arranca**.

### Ejemplo de inferencia

```bash
curl -X POST http://127.0.0.1:8000/v1/riesgo-cardiovascular \
  -H "Content-Type: application/json" \
  -d '{"age_years":52,"gender":1,"height":165,"weight":72,"cholesterol":1,"gluc":1,
       "smoke":0,"alco":0,"active":1,"ap_hi":130,"ap_lo":85}'
# → {"probabilidad":0.5173…,"prevalencia_base":0.4948…,"modelo":{…},"aviso":"…","clase":1,"umbral":0.5}
```

Referencia completa de endpoints y esquemas en **[docs/API.md](docs/API.md)**.

---

## Modelos servidos

| Endpoint | Modelo | Target | AUC test [IC 95 %] | Salida |
|----------|--------|--------|--------------------|--------|
| `POST /v1/riesgo-cardiovascular` | A′ `riesgo_cv_con_pa`: RF(depth 12, hoja 20) | `cardio`, con presión | 0,8019 [0,7938 · 0,8088] | probabilidad + clase (umbral 0,5) |
| `POST /v1/hipertension-sin-pa` | B1 `hta_b1`, experimental: RF(depth 8, hoja 100) | `hta`, **sin** presión | 0,6955 [0,6866 · 0,7049] | solo probabilidad |

Seleccionados en DT-4 por log-loss en validación cruzada con la regla de 1 EE; en los
dos, el mínimo de log-loss era un XGBoost poco profundo con el RF dentro de 1 EE. El
test se evaluó una sola vez, pero ya se había observado en DT-1 (limitación declarada).
Recursos medidos en local: RSS de la API con los dos modelos 202 MiB (352 con los de
DT-1), `.pkl` 32 MB (130). Ficha completa en [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Pipeline sintético: modelos y criterio de selección (evidencia del leakage)

| Modelo               | Implementación            | Artefacto           |
|----------------------|---------------------------|---------------------|
| Regresión Logística  | `LogisticRegression`      | `modelo_logreg.pkl` |
| SVM (kernel RBF)     | `SVC(probability=True)`   | `modelo_svm.pkl`    |
| Árbol de Decisión    | `DecisionTreeClassifier`  | `modelo_tree.pkl`   |
| Random Forest        | `RandomForestClassifier`  | `modelo_rf.pkl`     |
| Gradient Boosting    | `XGBClassifier`           | `modelo_xgb.pkl`    |

La selección se hace por **macro-F1** sobre el conjunto de test y el ganador se
persiste en `models/mejor_modelo.txt`. Ningún servicio carga ya estos artefactos.

### Resultados, y cómo leerlos

| Modelo | Accuracy | Macro-F1 |
|--------|----------|----------|
| Regresión Logística | 47,70 % | 0,4711 |
| SVM RBF | 68,82 % | 0,6961 |
| Árbol de Decisión (`max_depth=8`) | 94,83 % | 0,9497 |
| Random Forest (300 árboles) | 94,84 % | 0,9498 |
| **XGBoost (400 árboles)** | **95,32 %** | **0,9547** |

Ese 95,3 % **no se lee contra el 29,6 % del azar, sino contra el 91,1 % de las
etiquetas del dataset sintético que recupera una regla `if` determinista, sin
entrenar** (`scripts/verify_leakage.py::regla_completa`): la misma que generó esas
etiquetas. Es el defecto que DT-1 cerró; sobre el dataset real, sin la presión en las
features, la mejor regla determinista que encuentra `scripts/audit_cardio_leakage.py`
supera al baseline de clase mayoritaria en solo +2,28 pp, mientras que con
`ap_hi`/`ap_lo` (control positivo) reconstruye el target al 99,27 %. Y la tabla
contiene dos señales del defecto:

- **Un árbol de profundidad 8 llega a 94,83 %**, a medio punto de 400 árboles
  boosteados. Cuando el gradient boosting no se despega de un modelo trivial, no
  queda estructura estadística que extraer.
- **Los modelos lineales y de kernel se hunden a 48–69 %** mientras los de árboles
  rondan el 95 %. El target es una partición por umbrales: los árboles la
  representan nativamente, los lineales no pueden. La geometría del problema es la
  de un `if`.

Análisis completo en [docs/LEAKAGE_ANALYSIS.md](docs/LEAKAGE_ANALYSIS.md) y
[docs/MODEL_CARD_SINTETICO.md](docs/MODEL_CARD_SINTETICO.md) (histórica).

---

## Clases objetivo del dataset sintético

| Clase | Etiqueta               | Criterio (guías AHA, simplificado) |
|-------|------------------------|------------------------------------|
| 0     | Normal                 | PAS < 120 y PAD < 80               |
| 1     | Prehipertensión        | PAS 120–129 o PAD 80–84            |
| 2     | Hipertensión Grado 1   | PAS 130–139 o PAD 85–89            |
| 3     | Hipertensión Grado 2   | PAS ≥ 140 o PAD ≥ 90               |

Diccionario de variables, procedencia y licencias de los datos en
**[docs/DATA.md](docs/DATA.md)**.

---

## Estructura del repositorio

```
hypertension-ml-system/
├── api/
│   ├── main.py                   # Servicio FastAPI: endpoints y carga en lifespan
│   ├── schemas.py                # Esquemas Pydantic (entrada en unidades humanas)
│   └── servicio.py               # Registro de modelos y matriz de features
├── app/dashboard.py              # Dashboard Streamlit (cliente HTTP del servicio)
├── src/
│   ├── cardio_features.py        # Fuente única: features, derivación y cotas de DT-1
│   ├── artefactos.py             # Verificación de versiones y sha256 de los .pkl
│   ├── train_cardio_real.py      # DT-1: entrenamiento sobre datos reales
│   ├── generate_dataset.py       # Generador sintético (evidencia del leakage)
│   └── train_classical_models.py # Entrenamiento sintético (evidencia; no se sirve)
├── scripts/
│   ├── verify_env.py             # Compuerta del entorno (make verify-env)
│   ├── verify_leakage.py         # Auditoría del leakage del dataset sintético
│   └── audit_cardio_leakage.py   # Criterio de aceptación de DT-1 (datos reales)
├── tests/                        # Contrato (en CI) e integración (se saltan sin .pkl/CSV)
├── notebooks/
│   └── EDA_cardiovascular_real.ipynb  # EDA sobre datos reales de Kaggle
├── data/
│   ├── raw/                      # Dataset sintético generado (no versionado)
│   └── real/cardio/              # Dataset Kaggle (no versionado; se descarga de la fuente)
├── models/                       # Artefactos entrenados (.pkl no versionados)
│   └── dt1_manifest.json         # Evidencia de la corrida DT-1 (sí versionado)
├── docs/                         # Documentación técnica
└── [scripts de la raíz]          # Pipeline heredado — ver ROADMAP
```

---

## Documentación

| Documento | Contenido |
|-----------|-----------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Módulos, contratos y flujo de datos |
| [LEAKAGE_ANALYSIS.md](docs/LEAKAGE_ANALYSIS.md) | Auditoría del target leakage con evidencia |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Model card de A′ y B1: uso previsto, límites, riesgos |
| [MODEL_CARD_SINTETICO.md](docs/MODEL_CARD_SINTETICO.md) | Model card histórica del modelo sintético |
| [DATA.md](docs/DATA.md) | Diccionario de datos, procedencia y licencias |
| [API.md](docs/API.md) | Referencia de endpoints |
| [DT4_PROTOCOL.md](docs/DT4_PROTOCOL.md) | Protocolo preregistrado de selección y evaluación |
| [DT4_RESULTS.md](docs/DT4_RESULTS.md) | Resultados vigentes: selección por CV y una evaluación en test |
| [DT1_RESULTS.md](docs/DT1_RESULTS.md) | Resultados históricos de la primera corrida sobre el dataset real |
| [ROADMAP.md](docs/ROADMAP.md) | Deuda técnica priorizada y trabajo futuro |
| [BITACORA.md](BITACORA.md) | Registro cronológico de decisiones |

---

## Estado del proyecto

**Fase actual: prototipo funcional sirviendo los modelos de DT-4; compuerta de
validación estadística (DT-1 a DT-4) cerrada.**

- ✅ **DT-1 cerrado** (2026-09-17). El entrenamiento migró al dataset real; ninguna
  regla determinista sobre las features supera al baseline por más de 2,28 pp.
- ✅ **DT-3 y DT-4 cerrados**: selección por log-loss en validación cruzada de 5 folds
  sobre el train y una sola evaluación en test, con protocolo preregistrado (`0d8d4c9`),
  código `ad518f2`, manifiesto `41572ed` y servicio `a7bf9ef`. Limitación: el test ya se
  había observado en DT-1.
- ✅ **Servicio migrado** (`6e59035`, `a7bf9ef`): la API y el dashboard sirven solo A′ y
  B1 de DT-4. Sigue abierta la otra mitad de DT-5: el pipeline heredado de la raíz.
- 🟡 **DT-6 parcial**: 108 tests (los de contrato corren en CI; los de integración se
  saltan sin `.pkl` ni CSV). Cobertura de `src/` + `api/`: 72 % en local, 65 % en CI,
  donde se aplica el criterio del 70 %.
- No hay despliegue público: el servicio corre en local.

Ver [docs/ROADMAP.md](docs/ROADMAP.md) y [BITACORA.md](BITACORA.md).

## Licencia

MIT — ver [LICENSE](LICENSE). El dataset de Kaggle **no se incluye**: su licencia es
desconocida, así que se purgó de la historia del repositorio y se descarga de la fuente.
Ver [docs/DATA.md](docs/DATA.md).

## Autor

Luis Araque — [luisalfredoaraque@gmail.com](mailto:luisalfredoaraque@gmail.com)
