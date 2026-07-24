# Changelog

Todos los cambios notables de este proyecto se documentan en este archivo.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y el versionado sigue [SemVer](https://semver.org/lang/es/).

---

## [No publicado]

### Por hacer

Ver [docs/ROADMAP.md](docs/ROADMAP.md). Prioridad inmediata: corregir el target
leakage (DT-1) migrando el entrenamiento al dataset real de Kaggle.

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
