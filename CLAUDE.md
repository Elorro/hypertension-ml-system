# CLAUDE.md — hypertension_ml

Contexto para agentes de IA que trabajen en este repositorio.
Complementa las reglas globales de `~/.claude/CLAUDE.md`.

---

## Qué es este proyecto

Sistema end-to-end de clasificación de hipertensión arterial en 4 niveles.
Pipeline: generación de datos → entrenamiento comparativo de 5 modelos → servicio
FastAPI → dashboard Streamlit.

**Naturaleza real del proyecto:** demostración de ingeniería de ML, no herramienta
clínica. Ver la sección de estado antes de proponer cualquier trabajo.

## Fase actual

**Fase 3 (prototipo). Compuerta de Fase 2 superada en el entrenamiento, no en el
servicio.**

| Fase | Estado |
|------|--------|
| 1. Diseño | ✅ Completa |
| 2. Validación matemática | ⚠️ DT-1 cerrado; DT-3/DT-4 abiertos |
| 3. Prototipo | ✅ Funcional (sirviendo el modelo sintético) |
| 4. Testing | ⬜ No iniciada |

No proponer optimización de hiperparámetros, modelos nuevos ni despliegue hasta
cerrar **DT-4** (el test elige y evalúa al mismo tiempo) y **DT-5** (el servicio
sigue cargando el modelo sintético).

## El defecto que dominaba todo lo demás — cerrado en entrenamiento (2026-09-17)

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

Resultado honesto: AUC 0,6941 [0,6849 · 0,7035] para estimar hipertensión sin medir
la presión, con accuracy apenas +3,25 pp sobre el baseline. **Los cinco algoritmos
empatan dentro del ruido** — no presentar «ganó Random Forest». Detalle en
`docs/DT1_RESULTS.md`.

**Advertencia vigente:** `api/main.py` y el dashboard siguen cargando
`models/modelo_*.pkl`, el modelo sintético. Lo que el sistema *expone* sigue siendo
el `if` tautológico. Los artefactos de DT-1 llevan prefijo `dt1_`.

## Estructura y qué está vivo

**Pipeline activo:**

```
src/generate_dataset.py         → data/raw/dataset_hipertension_sintetico.csv
src/train_classical_models.py   → models/*.pkl + scaler.pkl + mejor_modelo.txt
src/train_cardio_real.py        → models/dt1_*.pkl + dt1_manifest.json  (DT-1)
api/main.py                     → FastAPI :8000
app/dashboard.py                → Streamlit :8501 (cliente HTTP del servicio)
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
- **`Estres` sin tilde** en el pipeline activo. Con tilde solo en el heredado.
- **Orden de features acoplado** entre 4 archivos + docs. Cambiarlo exige tocarlos
  todos a la vez; ver `CONTRIBUTING.md`.
- **Nunca versionar** `.pkl`, CSV generados ni el `.venv`.
- **Avisos médicos** presentes en servicio, dashboard y docs. No retirarlos.
- Documentación y docstrings en español; nombres de código en el idioma que ya usa
  cada archivo.

## Comandos

```bash
make setup       # dataset + entrenamiento (prerrequisito del servicio)
make api         # uvicorn :8000
make dashboard   # streamlit :8501
make train-real  # DT-1: entrenamiento sobre datos reales (~51 min)
make audit       # auditoría de leakage del dataset sintético
make audit-real  # criterio de aceptación de DT-1 (solo numpy+pandas)
make test        # pytest (suite aún no existe — DT-6)
make lint        # ruff
```

El servicio carga el modelo en tiempo de import: sin `models/` poblado, `uvicorn`
falla al arrancar.

## Trabajo pendiente

Priorizado en `docs/ROADMAP.md` con IDs `DT-N`. El orden importa: DT-1 a DT-4 son
bloqueantes y nada posterior tiene sentido sin ellos.

Siguiente hito concreto (DT-4): split en tres (60/20/20) sobre
`src/train_cardio_real.py`, seleccionando en validación y tocando el test una sola
vez. Mientras no esté, las métricas publicadas están sesgadas al alza por selección.

## Fuente de verdad

`BITACORA.md` registra el estado real y las decisiones tomadas. Ante cualquier duda
sobre en qué punto está el proyecto, leer su última entrada antes de proponer
trabajo.
