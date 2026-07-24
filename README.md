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

**Lo que sí demuestra este repositorio, y demuestra bien:**

- Pipeline reproducible de datos → entrenamiento → selección de modelo → artefactos.
- Comparación sistemática de 5 algoritmos con criterio de selección explícito (macro-F1).
- Servicio de inferencia con FastAPI, esquema validado con Pydantic y OpenAPI automático.
- Dashboard Streamlit con inferencia individual y masiva vía CSV.
- EDA sobre un dataset clínico real (70.000 pacientes, Kaggle).

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

**Por qué este README no publica una tabla de accuracy:** dado el leakage descrito
arriba, cualquier métrica alta mide la capacidad de memorizar un umbral, no de
predecir. El único número informativo es la comparación contra el **baseline de la
regla clínica** (clasificar solo con `PAS`/`PAD`), y por construcción ningún modelo
puede superarlo de forma significativa. Ver
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
│   └── train_classical_models.py # Entrenamiento y selección de modelo
├── scripts/verify_leakage.py     # Auditoría reproducible del target leakage
├── notebooks/
│   └── EDA_cardiovascular_real.ipynb  # EDA sobre datos reales de Kaggle
├── data/
│   ├── raw/                      # Dataset sintético generado (no versionado)
│   └── real/cardio/              # Dataset Kaggle (70k pacientes)
├── models/                       # Artefactos entrenados (no versionados)
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
| [ROADMAP.md](docs/ROADMAP.md) | Deuda técnica priorizada y trabajo futuro |
| [BITACORA.md](BITACORA.md) | Registro cronológico de decisiones |

---

## Estado del proyecto

**Fase actual: prototipo funcional con validación estadística pendiente.**

El sistema corre end-to-end, pero no ha pasado la compuerta de validación: el
dataset sintético no permite conclusiones predictivas. El siguiente hito es
reentrenar sobre el dataset real de Kaggle excluyendo `PAS`/`PAD` del conjunto de
features. Ver [docs/ROADMAP.md](docs/ROADMAP.md).

## Licencia

MIT — ver [LICENSE](LICENSE). El dataset de Kaggle incluido en `data/real/` se
distribuye bajo sus propios términos; ver [docs/DATA.md](docs/DATA.md).

## Autor

Luis Araque — [luisalfredoaraque@gmail.com](mailto:luisalfredoaraque@gmail.com)
