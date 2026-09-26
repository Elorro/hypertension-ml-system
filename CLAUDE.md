# CLAUDE.md — hypertension_ml

Contexto para agentes de IA que trabajen en este repositorio.
Complementa las reglas globales de `~/.claude/CLAUDE.md`.

---

## Qué es este proyecto

Sistema end-to-end sobre el dataset real *Cardiovascular Disease* (Kaggle):
entrenamiento comparativo de 5 modelos (DT-1) → servicio FastAPI que devuelve
probabilidades de A′ (riesgo cardiovascular, con presión) y B1 (hipertensión sin
presión, experimental) → dashboard Streamlit. El pipeline sintético original (4 niveles
de HTA) se conserva solo como evidencia del target leakage.

**Naturaleza real del proyecto:** demostración de ingeniería de ML, no herramienta
clínica. Ver la sección de estado antes de proponer cualquier trabajo.

## Fase actual

**Fase 3 (prototipo). El servicio sirve los modelos de DT-4 desde `a7bf9ef`.**

| Fase | Estado |
|------|--------|
| 1. Diseño | ✅ Completa |
| 2. Validación matemática | ✅ DT-1 a DT-4 cerrados (DT-4 con protocolo preregistrado) |
| 3. Prototipo | ✅ Funcional (API y dashboard sobre A′ y B1 de DT-4) |
| 4. Testing | ◐ Suite de contrato e integración (DT-6 parcial: cobertura en CI 65 % < 70 %) |

Cualquier cambio de modelo o de hiperparámetros exige un **protocolo preregistrado
nuevo** (como `docs/DT4_PROTOCOL.md`), commiteado antes de entrenar; nunca seleccionar
mirando el test. El test de DT-4 ya se había observado en DT-1 (limitación declarada).
No hay despliegue público; plataforma y manejo de los `.pkl` en deploy están sin
decidir (servidos: 32 MB; RSS de la API: 202 MiB).

## El defecto que dominaba todo lo demás — cerrado en entrenamiento (2026-09-17) y en el servicio (2026-09-23)

El target del dataset sintético es una **función determinista de las features**.
`Diagnostico` se calcula con umbrales sobre `PAS`/`PAD`, y esas mismas columnas se
entregan al modelo como entrada.

Verificación: `python scripts/verify_leakage.py`

```
Baseline clase mayoritaria    → 29,61 %
Regla PAS/PAD pura            → 61,87 %
Regla completa del generador  → 91,09 %
```

Una regla `if` determinista, sin entrenar (`scripts/verify_leakage.py::regla_completa`),
recupera el 91,1 % de las etiquetas del dataset sintético. El 8,9 % restante es ruido
de redondeo del CSV, no señal. **Cualquier métrica de accuracy sobre el sintético es
tautológica.** Análisis completo en `docs/LEAKAGE_ANALYSIS.md`.

Implicación operativa: no reportar accuracy/F1 sin acompañarlo del baseline de la
regla clínica. No presentar el sistema como predictor de hipertensión.

**Qué cambió con DT-1.** `src/train_cardio_real.py` entrena sobre
`data/real/cardio/cardio_train.csv` (68.678 filas tras limpieza) con la presión
arterial excluida de las features. Criterio de aceptación verificado con
`python scripts/audit_cardio_leakage.py`:

```
B1 — hta SIN presión (real)          65,67 % → 67,95 %    +2,28 pp   PASA
CONTROL POSITIVO — con ap_hi/ap_lo   65,67 % → 99,27 %   +33,60 pp   FALLA (correcto)
```

Resultado vigente (DT-4, `docs/DT4_RESULTS.md`): AUC 0,6955 [0,6866 · 0,7049] para
estimar hipertensión sin medir la presión, con accuracy apenas +3,35 pp sobre el
baseline. El Random Forest servido es el más simple dentro de 1 EE del mínimo de
log-loss (un XGBoost poco profundo) — no presentarlo como «ganó Random Forest». La
regresión logística empata en AUC pero queda fuera de 1 EE en log-loss. La mejora
frente a DT-1 está en el sobreajuste y el tamaño, no en el AUC. Ablación vigente:
Δ AUC = 0,0991 [0,0919 · 0,1061]; el 0,1103 de DT-1 es histórico e inflado.

**Qué cambió en el servicio (`6e59035`, `a9b89ee`, `a7bf9ef`).** `api/` y `app/` sirven
solo los artefactos `dt4_*` (A′ y B1); `POST /predecir` y el contrato sintético se
eliminaron. La API verifica versiones y sha256 contra `models/dt4_manifest.json` al
arrancar y no arranca si algo no coincide. `dt1_*` y su manifiesto quedan como
registro. En API y docs se dice «probabilidad (ECE medido en test:
…)», nunca «calibrada»; B1 no devuelve clase.

## Estructura y qué está vivo

**Pipeline activo:**

```
src/cardio_features.py          → fuente única de features, derivación y cotas (DT-1)
src/train_cardio_real.py        → models/dt1_* (registro de DT-1; no modificar)
src/train_cardio_dt4.py         → models/dt4_*.pkl + dt4_manifest.json  (make train-dt4; protocolo docs/DT4_PROTOCOL.md)
src/artefactos.py               → verificación de versiones y sha256 (API y verify_env)
api/{main,schemas,servicio}.py  → FastAPI :8000 (A′ y B1)
app/dashboard.py                → Streamlit :8501 (cliente HTTP puro, API_URL)
```

**Evidencia del leakage — conservar, fuera del servicio:**

```
src/generate_dataset.py         → data/raw/dataset_hipertension_sintetico.csv
src/train_classical_models.py   → models/modelo_*.pkl + scaler.pkl + mejor_modelo.txt
scripts/verify_leakage.py
```

**Pipeline heredado — no tocar sin decisión previa:**

`data_pipeline.py`, `train_models.py`, `model_utils.py`, `streamlit_app.py`,
`chatbot_cli.py` en la raíz. Esquema incompatible (`HTA_Nivel` vs `Diagnostico`,
`Estrés` con tilde vs `Estres` sin tilde). Además `train_models.py` **no ejecuta con
scikit-learn ≥ 1.4** (`base_estimator` fue eliminado de `CalibratedClassifierCV`).

Antes de modificar cualquier archivo de la raíz, confirmar con Luis si se migra
(DT-5 opción b) o se elimina (opción a).

## Convenciones no negociables

- **Semillas fijas.** `random_state=42` / `seed=42` en toda operación aleatoria.
- **Toda auditoría de leakage lleva control positivo.** Un «pasa» sin demostrar que
  la misma maquinaria detecta el leakage cuando existe no prueba nada.
- **`scripts/audit_cardio_leakage.py` no importa sklearn.** Debe correr aunque el
  entorno de ML no esté instalado; mantenerlo en numpy + pandas.
- **`scripts/audit_cardio_leakage.py` reimplementa la limpieza a propósito.** No debe
  importar `src/cardio_features.py`: es una verificación independiente.
- **`Estres` sin tilde** en el pipeline sintético de `src/`. Con tilde solo en el heredado.
- **Orden de features en un solo sitio:** `src/cardio_features.py`. Cambiarlo invalida
  los `.pkl` de DT-1 y de DT-4; los tests de equivalencia lo detectan. Ver `CONTRIBUTING.md`.
- **Nunca versionar** `.pkl`, CSV generados ni el `.venv`.
- **Avisos médicos** presentes en servicio, dashboard y docs. No retirarlos.
- Documentación y docstrings en español; nombres de código en el idioma que ya usa
  cada archivo.

## Comandos

```bash
make train-dt4        # DT-4: CV + 1 EE + una evaluación en test (~7 min, nohup, logs/dt4_run.log)
make train-real       # DT-1 (registro): ~51 min, genera dt1_*.pkl
make verify-env       # compuerta: versiones + sha256 + carga y predicción de los dt4_*.pkl
make verify-env-dt1   # lo mismo para el registro de DT-1
make serve-api        # uvicorn :8000 (alias: make api)
make serve-dashboard  # streamlit :8501 (alias: make dashboard)
make setup            # pipeline sintético (evidencia; no alimenta al servicio)
make audit            # auditoría de leakage del dataset sintético
make audit-real       # criterio de aceptación de DT-1 (solo numpy+pandas)
make test             # pytest: contrato siempre; integración se salta sin .pkl/CSV
make lint             # ruff check + ruff format --check
```

La API carga los modelos en el lifespan: sin `models/dt4_*.pkl` o con un sha256
distinto al del manifiesto, no arranca.

## Trabajo pendiente

Priorizado en `docs/ROADMAP.md` con IDs `DT-N`. DT-1 a DT-4 están cerrados. Pendientes
principales: la parte de DT-5 que falta (pipeline heredado de la raíz: migrar u
eliminar), la cobertura de DT-6 en CI, el análisis de equidad (DT-15) y la decisión de
despliegue (plataforma y manejo de los `.pkl`).

## Fuente de verdad

`BITACORA.md` registra el estado real y las decisiones tomadas. Ante cualquier duda
sobre en qué punto está el proyecto, leer su última entrada antes de proponer
trabajo.
