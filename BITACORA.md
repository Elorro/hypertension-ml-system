# Bitácora — hypertension_ml

Registro cronológico de decisiones y estado real del proyecto.
Fuente de verdad: ante discrepancia con cualquier otro documento, manda la última
entrada de este archivo.

Formato: entradas descendentes (lo más reciente arriba).

---

## 2026-07-24 — Auditoría metodológica y documentación del repositorio

**Fase:** 3 (prototipo) · **Compuerta de Fase 2 (validación matemática): NO superada**

### Qué se hizo

Revisión completa del repositorio con vistas a publicarlo. Auditoría estadística del
pipeline de entrenamiento y redacción de la documentación técnica.

### Hallazgo principal: target leakage confirmado

El target `Diagnostico` del dataset sintético es una **función determinista de las
features de entrada**. `src/generate_dataset.py` lo construye con umbrales sobre
`PAS`/`PAD` más ajustes por factores de riesgo, y `src/train_classical_models.py`
entrega todas esas columnas al modelo (`X = df.drop("Diagnostico", axis=1)`).

Evidencia (`scripts/verify_leakage.py`, sobre las 50.000 filas con seed=42):

| Referencia | Coincidencia con el target |
|------------|---------------------------|
| Baseline clase mayoritaria | 29,61 % |
| Regla `if` sobre PAS/PAD, sin entrenar | 61,87 % |
| Regla completa del generador, sin entrenar | **91,09 %** |

La brecha del ~9 % es **ruido de redondeo**, no señal: el generador clasifica sobre
valores en punto flotante y guarda el CSV redondeado (`pas.round(0)`,
`imc.round(1)`), lo que desplaza filas en las fronteras de decisión. Sobre valores
sin redondear la reconstrucción sería exacta.

**Conclusión:** los modelos no aprenden nada clínico, reaprenden las guías AHA
codificadas en el generador. Ninguna métrica del proyecto mide capacidad predictiva.

### Hallazgos secundarios

1. **Escalado antes del split** — `scaler.fit_transform(X)` sobre el dataset
   completo, `train_test_split` después. Leakage de estadísticos.
2. **Split sin `stratify`** — proporciones de clase distintas entre train y test.
3. **Sin validación cruzada** — selección del mejor modelo sobre un único split; las
   diferencias de F1 pueden estar dentro del ruido de partición.
4. **Selección y evaluación sobre el mismo conjunto** — F1 del ganador sesgado al alza.
5. **Dos pipelines paralelos incompatibles** — los scripts de la raíz usan
   `HTA_Nivel` y `Estrés` (con tilde); `src/` usa `Diagnostico` y `Estres`.
6. **`train_models.py` está muerto** — usa `base_estimator` de
   `CalibratedClassifierCV`, eliminado en scikit-learn 1.4. No ejecuta en entornos
   modernos.
7. **`requirements.txt` incorrecto** — declaraba `lightgbm` y `catboost`, que no se
   importan en ningún archivo, y omitía `requests`, que `app/dashboard.py` sí usa.
8. **`.venv/` corrupto** — generado en otra máquina (`/home/ele/...`), con binarios
   no ejecutables. Irrelevante para el repositorio (está ignorado), pero hay que
   recrearlo localmente.

### Decisiones tomadas

- **Documentar el leakage de forma prominente en lugar de ocultarlo.** Un revisor
  competente lo detecta en minutos; declararlo con evidencia reproducible y un plan
  de corrección es más creíble que un README que presuma métricas del 95 %.
  El repositorio se posiciona como demostración de ingeniería de ML, no como
  predictor clínico.
- **No publicar tabla de accuracy en el README.** Sería engañosa. En su lugar se
  documenta cuál es el baseline correcto y por qué ningún modelo puede superarlo.
- **Conservar el pipeline heredado por ahora**, documentado como tal, hasta decidir
  entre migrar el modelo Keras y el CLI (DT-5b) o eliminarlo (DT-5a).
  Recomendación registrada: migrar, porque la comparación clásicos vs. red neuronal
  es el aporte diferencial del proyecto.
- **Licencia MIT**, con aviso médico adicional en el propio `LICENSE`.
- **Camino de salida elegido para DT-1:** migrar el entrenamiento al dataset real de
  Kaggle ya presente en `data/real/`, definiendo el target desde `ap_hi`/`ap_lo` y
  excluyendo esas columnas de las features.

### Artefactos producidos

`README.md` reescrito · `docs/{ARCHITECTURE,LEAKAGE_ANALYSIS,MODEL_CARD,DATA,API,ROADMAP}.md`
· `scripts/verify_leakage.py` · `CONTRIBUTING.md` · `CHANGELOG.md` · `LICENSE` ·
`CITATION.cff` · `CLAUDE.md` · `Makefile` · `.gitignore` y `requirements.txt`
corregidos · `requirements-dev.txt` · CI de GitHub Actions.

### Estado al cierre

Sistema funcional end-to-end, documentación completa, defecto central declarado y
con plan de corrección priorizado en `docs/ROADMAP.md`.

### Siguiente paso

**DT-1** — reentrenar sobre `data/real/cardio/cardio_train.csv` (70.000 pacientes)
sin `ap_hi`/`ap_lo` en las features. Pregunta de investigación: ¿se puede estimar el
estado hipertensivo sin medir la presión arterial? El baseline honesto vuelve a ser
la clase mayoritaria y el techo pasa a ser desconocido — que es justamente lo que
hace al problema interesante.

Pendiente de decisión de Luis: verificar los términos de redistribución del dataset
de Kaggle antes de publicar el repositorio, y decidir si se retira
`data/real/cardiovascular-disease-dataset.zip` del control de versiones (redundante
con el CSV ya descomprimido).

---

## 2026-01-05 — Sistema ML completado

**Fase:** 3 (prototipo)

Commit `72cae30`. Pipeline completo funcionando:

- Generador de dataset sintético (50.000 filas, 15 features, 4 clases).
- Entrenamiento de 5 modelos clásicos con selección automática por macro-F1.
- Servicio FastAPI con esquema Pydantic y documentación OpenAPI.
- Dashboard Streamlit con inferencia individual y masiva por CSV.
- EDA sobre dataset real de Kaggle (70.000 pacientes) en notebook.

Modelo ganador de la corrida: **XGBoost** (`models/mejor_modelo.txt`).

Sin validación estadística formal en este punto.

---

## 2026-01-05 — Commit inicial

**Fase:** 1 (diseño)

Commit `43c6c6e`. Estructura base del proyecto.
