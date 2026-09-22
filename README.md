# Sistema de Clasificación de Hipertensión Arterial

Sistema end-to-end de machine learning para clasificación de niveles de hipertensión
arterial (HTA): generación de datos, entrenamiento comparativo de cinco algoritmos,
servicio REST de inferencia y dashboard interactivo.

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-prototipo%20educativo-orange)

---

## ⚠️ Léase primero: qué es y qué NO es este proyecto

Este repositorio es una **demostración de ingeniería de sistemas de ML**, no un
predictor clínico válido. La distinción no es un tecnicismo:

> **El modelo no predice hipertensión. Reaprende una regla que ya conocemos.**

El dataset es sintético y su etiqueta se construye aplicando un umbral determinista
sobre la presión sistólica (`PAS`) y diastólica (`PAD`) — las mismas variables que
después se entregan al modelo como entrada. Es un caso de libro de **target leakage**:
el modelo puede alcanzar métricas casi perfectas simplemente reproduciendo el `if`
que generó la etiqueta.

Está verificado numéricamente y el análisis es reproducible:

```bash
python scripts/verify_leakage.py
```

Detalle completo, evidencia y consecuencias en **[docs/LEAKAGE_ANALYSIS.md](docs/LEAKAGE_ANALYSIS.md)**.

**Estado actual del defecto: corregido en el pipeline de entrenamiento, aún no en el
servicio.** Desde el 2026-09-17 existe un segundo pipeline,
[`src/train_cardio_real.py`](src/train_cardio_real.py), que entrena sobre el dataset
real de Kaggle (68.678 pacientes) excluyendo la presión arterial de las features. Ahí
la pregunta es genuina y la respuesta, modesta: **AUC 0,6941 [0,6849 · 0,7035]** para
estimar el estado hipertensivo sin medirlo, con una accuracy apenas +3,25 pp sobre el
baseline de clase mayoritaria. Verificable con:

```bash
python scripts/audit_cardio_leakage.py    # criterio de aceptación de DT-1
```

Resultados completos en **[docs/DT1_RESULTS.md](docs/DT1_RESULTS.md)**. El servicio
FastAPI y el dashboard **siguen sirviendo el modelo sintético**: lo que se expone en
`/predecir` continúa siendo el `if` tautológico hasta que se cierre DT-5.

**Lo que sí demuestra este repositorio, y demuestra bien:**

- Pipeline reproducible de datos → entrenamiento → selección de modelo → artefactos.
- Comparación sistemática de 5 algoritmos con criterio de selección explícito (macro-F1).
- Servicio de inferencia con FastAPI, esquema validado con Pydantic y OpenAPI automático.
- Dashboard Streamlit con inferencia individual y masiva vía CSV.
- EDA sobre un dataset clínico real (70.000 pacientes, Kaggle).
- Auditoría de leakage con **control positivo**, y una ablación con bootstrap pareado
  que cuantifica cuánto vale medir la presión arterial: Δ AUC = 0,1103
  [0,1027 · 0,1181].

**Aviso médico:** herramienta con fines educativos. No constituye diagnóstico,
consejo médico ni sustituye la valoración de un profesional de la salud.

---

## Arquitectura

```
                 ┌──────────────────────┐
                 │  generate_dataset.py │  genera CSV sintético (50k filas)
                 └──────────┬───────────┘
                            ▼
                 ┌───────────────────────────┐
                 │ train_classical_models.py │  entrena 5 modelos, elige por macro-F1
                 └──────────┬────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │      models/*.pkl    │  artefactos + scaler + mejor_modelo.txt
                 └──────────┬───────────┘
                            ▼
        ┌───────────────────┴────────────────────┐
        ▼                                        ▼
┌───────────────┐                      ┌────────────────────┐
│  api/main.py  │◄─────HTTP POST───────│  app/dashboard.py  │
│   (FastAPI)   │      /predecir       │    (Streamlit)     │
└───────────────┘                      └────────────────────┘
```

Descripción detallada de módulos, contratos y flujo de datos en
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

> **Nota sobre duplicación:** el repositorio contiene actualmente **dos pipelines
> paralelos e incompatibles entre sí** (`src/` + `api/` + `app/` es el activo;
> los scripts de la raíz son una versión anterior). Está documentado y priorizado
> en [docs/ROADMAP.md](docs/ROADMAP.md).

---

## Instalación

Requiere Python 3.12+.

```bash
git clone https://github.com/Elorro/hypertension-ml-system.git
cd hypertension-ml-system

python -m venv .venv
source .venv/bin/activate        # zsh/bash · Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Uso rápido

Los artefactos entrenados (`models/*.pkl`) **no se versionan**, así que hay que
generarlos antes de levantar el servicio:

```bash
make setup      # dataset sintético + entrenamiento de los 5 modelos
make api        # servicio en http://127.0.0.1:8000  (docs en /docs)
make dashboard  # dashboard en http://localhost:8501
```

Sin `make`:

```bash
python src/generate_dataset.py
python src/train_classical_models.py
uvicorn api.main:app --reload
streamlit run app/dashboard.py
```

> El servicio carga el modelo **en tiempo de import**. Si `models/` está vacío,
> `uvicorn` falla al arrancar con `FileNotFoundError`. Ejecuta `make setup` primero.

### Ejemplo de inferencia

```bash
curl -X POST http://127.0.0.1:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "Edad": 52, "Sexo": 1, "Peso": 88.0, "Talla": 1.74, "IMC": 29.1,
    "PAS": 145.0, "PAD": 92.0, "PAM": 109.7, "Frec_Card": 78.0,
    "Colesterol": 215.0, "Glucosa": 112.0, "Tabaquismo": 1,
    "Ejercicio": 2, "Estres": 7, "Herencia_HTA": 1
  }'
```

Referencia completa de endpoints y esquemas en **[docs/API.md](docs/API.md)**.

---

## Modelos y criterio de selección

| Modelo               | Implementación            | Artefacto           |
|----------------------|---------------------------|---------------------|
| Regresión Logística  | `LogisticRegression`      | `modelo_logreg.pkl` |
| SVM (kernel RBF)     | `SVC(probability=True)`   | `modelo_svm.pkl`    |
| Árbol de Decisión    | `DecisionTreeClassifier`  | `modelo_tree.pkl`   |
| Random Forest        | `RandomForestClassifier`  | `modelo_rf.pkl`     |
| Gradient Boosting    | `XGBClassifier`           | `modelo_xgb.pkl`    |

La selección se hace por **macro-F1** sobre el conjunto de test y el ganador se
persiste en `models/mejor_modelo.txt`, que el servicio lee al arrancar.

### Resultados, y cómo leerlos

| Modelo | Accuracy | Macro-F1 |
|--------|----------|----------|
| Regresión Logística | 47,70 % | 0,4711 |
| SVM RBF | 68,82 % | 0,6961 |
| Árbol de Decisión (`max_depth=8`) | 94,83 % | 0,9497 |
| Random Forest (300 árboles) | 94,84 % | 0,9498 |
| **XGBoost (400 árboles)** | **95,32 %** | **0,9547** |

Ese 95,3 % **no se lee contra el 29,6 % del azar, sino contra el 91,1 % que alcanza
una regla `if` de doce líneas sin entrenamiento** — la misma que generó las
etiquetas. Y la tabla contiene dos señales del defecto:

- **Un árbol de profundidad 8 llega a 94,83 %**, a medio punto de 400 árboles
  boosteados. Cuando el gradient boosting no se despega de un modelo trivial, no
  queda estructura estadística que extraer.
- **Los modelos lineales y de kernel se hunden a 48–69 %** mientras los de árboles
  rondan el 95 %. El target es una partición por umbrales: los árboles la
  representan nativamente, los lineales no pueden. La geometría del problema es la
  de un `if`.

Análisis completo en [docs/LEAKAGE_ANALYSIS.md](docs/LEAKAGE_ANALYSIS.md) y
[docs/MODEL_CARD.md](docs/MODEL_CARD.md).

---

## Clases objetivo

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
├── api/main.py                   # Servicio FastAPI de inferencia
├── app/dashboard.py              # Dashboard Streamlit (consume el servicio)
├── src/
│   ├── generate_dataset.py       # Generador de datos sintéticos
│   ├── train_classical_models.py # Entrenamiento sintético (alimenta al servicio)
│   └── train_cardio_real.py      # DT-1: entrenamiento sobre datos reales
├── scripts/
│   ├── verify_leakage.py         # Auditoría del leakage del dataset sintético
│   └── audit_cardio_leakage.py   # Criterio de aceptación de DT-1 (datos reales)
├── notebooks/
│   └── EDA_cardiovascular_real.ipynb  # EDA sobre datos reales de Kaggle
├── data/
│   ├── raw/                      # Dataset sintético generado (no versionado)
│   └── real/cardio/              # Dataset Kaggle (70k pacientes)
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
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Model card: uso previsto, límites, riesgos |
| [DATA.md](docs/DATA.md) | Diccionario de datos, procedencia y licencias |
| [API.md](docs/API.md) | Referencia de endpoints |
| [DT1_RESULTS.md](docs/DT1_RESULTS.md) | Resultados sobre el dataset real y sus límites |
| [ROADMAP.md](docs/ROADMAP.md) | Deuda técnica priorizada y trabajo futuro |
| [BITACORA.md](BITACORA.md) | Registro cronológico de decisiones |

---

## Estado del proyecto

**Fase actual: prototipo funcional; compuerta de validación superada en el
entrenamiento, no en el servicio.**

- ✅ **DT-1 cerrado** (2026-09-17). El entrenamiento migró al dataset real; ninguna
  regla determinista sobre las features supera al baseline por más de 2,28 pp.
- ❌ **DT-4 abierto**, y es el defecto estadístico más grave que queda: el ganador se
  elige por macro-F1 sobre el mismo test que reporta sus métricas, así que las cifras
  publicadas están sesgadas al alza.
- ❌ **DT-3 parcial**: split estratificado sí, validación cruzada no. Los cinco
  algoritmos quedan dentro del ruido entre sí.
- ❌ **DT-5 abierto**: `api/main.py` sigue cargando `models/modelo_*.pkl`, el modelo
  entrenado sobre el dataset sintético.

Ver [docs/ROADMAP.md](docs/ROADMAP.md) y [BITACORA.md](BITACORA.md).

## Licencia

MIT — ver [LICENSE](LICENSE). El dataset de Kaggle incluido en `data/real/` se
distribuye bajo sus propios términos; ver [docs/DATA.md](docs/DATA.md).

## Autor

Luis Araque — [luisalfredoaraque@gmail.com](mailto:luisalfredoaraque@gmail.com)
