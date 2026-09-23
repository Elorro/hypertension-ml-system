# Sistema de Clasificación de Hipertensión Arterial

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
la pregunta es genuina y la respuesta, modesta: **AUC 0,6941 [0,6849 · 0,7035]** para
estimar el estado hipertensivo sin medirlo, con una accuracy apenas +3,25 pp sobre el
baseline de clase mayoritaria. Verificable con:

```bash
python scripts/audit_cardio_leakage.py    # criterio de aceptación de DT-1
```

Resultados completos en **[docs/DT1_RESULTS.md](docs/DT1_RESULTS.md)**. Desde
`6e59035` el servicio FastAPI y el dashboard sirven **solo** esos modelos reales
(A′ con presión arterial y B1 sin ella); el contrato sintético `/predecir` se eliminó.

**Lo que sí demuestra este repositorio, y demuestra bien:**

- Pipeline reproducible de datos → entrenamiento → selección de modelo → artefactos.
- Comparación sistemática de 5 algoritmos con criterio de selección explícito (macro-F1).
- Servicio de inferencia con FastAPI, esquema validado con Pydantic y OpenAPI automático:
  validación del dominio de entrenamiento, contrato anti-leakage (B1 rechaza la
  presión con 422) y verificación de versiones y sha256 de los artefactos al arrancar.
- Dashboard Streamlit con inferencia individual y masiva vía CSV, cliente HTTP puro.
- EDA sobre un dataset clínico real (70.000 pacientes, Kaggle).
- Auditoría de leakage con **control positivo**, y una ablación con bootstrap pareado
  que cuantifica cuánto vale medir la presión arterial: Δ AUC = 0,1103
  [0,1027 · 0,1181].

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
| `POST /v1/riesgo-cardiovascular` | A′ `riesgo_cv_con_pa` | `cardio`, con presión | 0,8017 [0,7937 · 0,8088] | probabilidad + clase (umbral 0,5) |
| `POST /v1/hipertension-sin-pa` | B1 `hta_b1`, experimental | `hta`, **sin** presión | 0,6941 [0,6849 · 0,7035] | solo probabilidad |

Ambos ganadores son Random Forest, pero en B1 los cinco algoritmos empatan dentro del
ruido. Las métricas están sesgadas al alza porque el mismo test eligió al ganador
(DT-4). Ficha completa en [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

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
| [DT1_RESULTS.md](docs/DT1_RESULTS.md) | Resultados sobre el dataset real y sus límites |
| [ROADMAP.md](docs/ROADMAP.md) | Deuda técnica priorizada y trabajo futuro |
| [BITACORA.md](BITACORA.md) | Registro cronológico de decisiones |

---

## Estado del proyecto

**Fase actual: prototipo funcional sirviendo los modelos reales; DT-4 abierto.**

- ✅ **DT-1 cerrado** (2026-09-17). El entrenamiento migró al dataset real; ninguna
  regla determinista sobre las features supera al baseline por más de 2,28 pp.
- ❌ **DT-4 abierto**, y es el defecto estadístico más grave que queda: el ganador se
  elige por macro-F1 sobre el mismo test que reporta sus métricas, así que las cifras
  publicadas están sesgadas al alza.
- ❌ **DT-3 parcial**: split estratificado sí, validación cruzada no. Los cinco
  algoritmos quedan dentro del ruido entre sí.
- ✅ **Servicio migrado** (`6e59035`): la API y el dashboard sirven solo A′ y B1 de
  DT-1. Sigue abierta la otra mitad de DT-5: el pipeline heredado de la raíz.
- 🟡 **DT-6 parcial**: 90 tests (los de contrato corren en CI; los de integración se
  saltan sin `.pkl` ni CSV), pero la cobertura de `src/` + `api/` es 65 %, bajo el 70 %
  del criterio.
- No hay despliegue público: el servicio corre en local.

Ver [docs/ROADMAP.md](docs/ROADMAP.md) y [BITACORA.md](BITACORA.md).

## Licencia

MIT — ver [LICENSE](LICENSE). El dataset de Kaggle **no se incluye**: su licencia es
desconocida, así que se purgó de la historia del repositorio y se descarga de la fuente.
Ver [docs/DATA.md](docs/DATA.md).

## Autor

Luis Araque — [luisalfredoaraque@gmail.com](mailto:luisalfredoaraque@gmail.com)
