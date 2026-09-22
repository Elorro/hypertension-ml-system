# Bitácora — hypertension_ml

Registro cronológico de decisiones y estado real del proyecto.
Fuente de verdad: ante discrepancia con cualquier otro documento, manda la última
entrada de este archivo.

Formato: entradas descendentes (lo más reciente arriba).

---

## 2026-09-21 — DT-1 cerrado: el proyecto pasa a ser predictivo

**Fase:** 2 (validación matemática) — **compuerta superada**

### Qué se cerró

El defecto que dominaba el repositorio desde enero está resuelto. El entrenamiento ya
no corre sobre el dataset sintético cuyo target era una función determinista de las
features, sino sobre `data/real/cardio/cardio_train.csv` (70.000 → 68.678 filas tras
limpieza declarada a priori).

La corrida de referencia es del **2026-09-17** (`src/train_cardio_real.py`, 3.071 s,
`seed=42`); esta entrada registra su cierre formal: auditoría de aceptación,
documentación y actualización del ROADMAP.

### Resultados

**A′ — ablación de la presión arterial (target `cardio`).** Con `ap_hi`/`ap_lo`:
AUC 0,8017. Sin ellas: 0,6914. Bootstrap pareado sobre las mismas filas:
Δ AUC = **0,1103, IC 95 % [0,1027 · 0,1181]**. Es el resultado cuantitativo más
sólido del proyecto.

**B1 — hipertensión sin medir la presión (target `hta`, 10 features).**
AUC 0,6941 [0,6849 · 0,7035]. Hay señal real, pero la accuracy (68,92 %) supera al
baseline de clase mayoritaria (65,67 %) en solo **+3,25 pp**: la información está en
el ranking de riesgo, no en la decisión binaria con umbral 0,5.

### Tres lecturas que conviene no suavizar

1. **Los cinco algoritmos empatan.** Random Forest 0,6941 vs. regresión logística
   0,6924, con IC de ancho ±0,009. Declarar un ganador sería sobreleer el resultado.
   Lo defendible: el techo sin presión arterial es AUC ≈ 0,69 y una regresión
   logística lo alcanza.
2. **Los hiperparámetros heredados sobreajustan.** RF: AUC 0,803 en train → 0,694 en
   test. XGBoost: 0,830 → 0,685. La regresión logística no se mueve (0,695 → 0,692).
   Venían del pipeline sintético, donde la flexibilidad servía para memorizar un `if`.
3. **DT-4 sigue abierto y es ahora el peor defecto vivo.** El ganador se elige por
   macro-F1 sobre el mismo test que reporta sus métricas.

### Criterio de aceptación

`scripts/audit_cardio_leakage.py` (nuevo, solo numpy + pandas) busca reglas
deterministas sin entrenar modelos: mejor umbral univariado y árbol CART propio de
profundidad ≤ 3.

```
                                      baseline   mejor regla   ganancia
B1 — hta SIN presión (real)            65,67 %      67,95 %     +2,28 pp   PASA
CONTROL POSITIVO — con ap_hi/ap_lo     65,67 %      99,27 %    +33,60 pp   FALLA (correcto)
```

El control positivo es lo que hace válida la auditoría: la misma maquinaria que
reporta +2,28 pp reconstruye el target al 99,27 % en cuanto se le devuelven las dos
columnas de presión. La auditoría además reimplementa la limpieza de forma
independiente y verifica que llega a las mismas 68.678 filas.

### Hallazgo de entorno

El entorno local se degradó desde la corrida del 17-sep: ya no tiene `scikit-learn`,
`xgboost` ni `joblib`, `pandas` pasó de 2.3.3 a 3.0.3 y `.venv/bin/python3` perdió el
bit de ejecución. Por eso la auditoría se escribió sin dependencias de ML: corre hoy
tal cual. Reproducir las cifras exactas del manifiesto exige reinstalar las versiones
que este registra — es el argumento empírico a favor de DT-11 (fijar versiones).

### Decisiones tomadas

- **El ganador de B1 no se presenta como «Random Forest gana».** Se reporta el empate
  y el techo de AUC.
- **La auditoría lleva control positivo obligatorio.** Un «pasa» sin él no prueba nada
  sobre el poder del buscador de reglas.
- **Los artefactos `dt1_*` no sustituyen todavía a los del servicio.** `api/main.py`
  sigue sirviendo el modelo sintético; migrarlo es parte de DT-5, y hasta entonces lo
  que el sistema expone en producción sigue siendo el `if` tautológico.
- **El umbral `≥` de `hta` no se re-eligió.** La corrida con `>` es reporte de
  robustez (AUC 0,6657) y no guardó artefactos.

### Artefactos producidos

`scripts/audit_cardio_leakage.py` · `docs/DT1_RESULTS.md` · `models/dt1_manifest.json`
(53 KB de evidencia: métricas, curvas de calibración, hashes, tiempos) ·
`models/dt1_*__modelo.pkl` + `__scaler.pkl` · `docs/ROADMAP.md`, `README.md`,
`PORTFOLIO.md`, `CHANGELOG.md` y `CLAUDE.md` actualizados · `Makefile` con
`train-real` y `audit-real`.

### Siguiente paso

**DT-4** — split en tres (60/20/20): seleccionar sobre validación, tocar el test una
sola vez. Es barato de implementar sobre `train_cardio_real.py` y es lo único que
falta para que las cifras publicadas dejen de estar sesgadas al alza. Después, DT-3
(validación cruzada) y DT-5 (migrar el servicio a los artefactos `dt1_*`, que es lo
que hace que el sistema deje de servir el modelo tautológico).

---

## 2026-07-24 (tarde) — CI en rojo: pipeline roto con scikit-learn moderno

**Fase:** 3 (prototipo)

### Qué pasó

El primer push activó el CI recién añadido y el job `pipeline` falló en el paso
"Entrenar modelos". El token de `gh` está vencido (403 al pedir los logs), así que
se reprodujo localmente en un venv limpio.

**Causa raíz:** `src/train_classical_models.py` pasaba `multi_class="multinomial"` a
`LogisticRegression`. El parámetro fue deprecado en scikit-learn 1.5 y **eliminado
en la 1.7**; el runner instaló la 1.9.0. Es el mismo tipo de defecto ya documentado
para `train_models.py` en la raíz, pero este estaba en el pipeline activo.

Corregido eliminando el parámetro: con el solver por defecto (lbfgs) y target
multiclase, el ajuste ya es multinomial. Comportamiento idéntico.

### Evidencia empírica del leakage

Entrenamiento completo ejecutado tras el fix (scikit-learn 1.9, split 80/20):

| Modelo | Accuracy | Macro-F1 |
|--------|----------|----------|
| Regresión Logística | 47,70 % | 0,4711 |
| SVM RBF | 68,82 % | 0,6961 |
| Árbol de Decisión (`max_depth=8`) | 94,83 % | 0,9497 |
| Random Forest (300 árboles) | 94,84 % | 0,9498 |
| XGBoost (400 árboles) | 95,32 % | 0,9547 |

Los números confirman el diagnóstico de la mañana por dos vías independientes:

1. **Brecha entre familias.** Árboles ~95 % vs. lineales/kernel 48–69 %. El target es
   una partición por umbrales; los árboles la representan nativamente y los lineales
   no pueden. La geometría del problema es la de un `if`.
2. **Un árbol de profundidad 8 empata con 400 boosteados.** 94,83 % vs. 95,32 %.
   Cuando el gradient boosting no se despega de un modelo trivial, no queda
   estructura que extraer.

**Corrección a la entrada anterior:** se afirmó que ningún modelo superaría el
91,09 % de la reconstrucción manual. XGBoost lo superó por 4,2 pp. Motivo: la
reconstrucción arrastra el ruido de redondeo del CSV, mientras que el modelo aprende
sobre los valores redondeados que efectivamente observa. El 91,09 % es el techo de
la reconstrucción, no del problema; el techo efectivo ronda el 95–96 %.

### Otros cambios

- Cerrado **DT-8**: `except:` desnudo → `except ValueError` con aviso explícito.
- Limpieza de lint: imports sin usar y desordenados en `api/`, `app/`, `src/`, y
  variable de bucle sin usar en `dashboard.py`. `ruff check` pasa limpio (10 → 0).
- Nuevo **DT-17**: `SVC(probability=True)` quedó deprecado en scikit-learn 1.9 y se
  elimina en la 1.11. Como `requirements.txt` admite `<2.0`, el pipeline se romperá
  solo. No se aplicó el cambio porque altera el tipo del artefacto `modelo_svm.pkl`.
- Medido el coste del SVM-RBF: O(n²), ~4 min sobre 40.000 filas de entrenamiento
  (0,60 s con n=2.000 → 9,07 s con n=8.000). Domina el tiempo total del pipeline.

### Estado al cierre

Pipeline completo verificado end-to-end en local con scikit-learn 1.9. Lint limpio.
Pendiente de push y de confirmar el CI en verde.

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
