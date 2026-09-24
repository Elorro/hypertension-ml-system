# DT-4 — Protocolo preregistrado: separar selección de evaluación

Este documento fija, **antes de entrenar**, cómo se elegirá y evaluará el modelo de
cada experimento en DT-4 (cierra también DT-3). Se commitea antes de la corrida; su
sha256 y el commit quedarán registrados en `models/dt4_manifest.json`. Cualquier
desviación durante la implementación o la corrida se registrará en el manifiesto
(`desviaciones`) y en el documento de resultados; **este protocolo no se corregirá a
posteriori**.

No contiene resultados.

---

## 1. Problema que resuelve

En DT-1 el ganador de cada experimento se eligió por macro-F1 con umbral 0,5 **sobre
el mismo test** que después reportó sus métricas (DT-4), con una sola partición y sin
variabilidad entre folds (DT-3). Además, el servicio entrega probabilidades: elegir
con una métrica de decisión binaria es incoherente con lo que se sirve.

DT-4:

- elegirá modelo e hiperparámetros por **validación cruzada sobre el train**;
- con una **regla de puntuación propia** (log-loss) como métrica de selección;
- y tocará el test **una sola vez por experimento**, con el modelo ya elegido.

## 2. Datos y partición

- **Dataset:** `data/real/cardio/cardio_train.csv`, el mismo de DT-1. Su sha256 se
  verificará contra el registrado en `models/dt1_manifest.json` antes de entrenar; si
  no coincide, la corrida se abortará.
- **Limpieza:** la de DT-1 sin cambios (`load_and_clean` de `src/train_cardio_real.py`,
  cuyas salidas están congeladas por hash en `tests/test_equivalencia_features.py`).
- **Partición train/test:** la misma 80/20 estratificada de DT-1 (`make_split`,
  `random_state=42`): estratificada por `cardio` para A′ y por `hta` para B1. Los
  índices de train y test serán idénticos a los de DT-1; la corrida lo verificará
  contra los hashes congelados.

**Limitación declarada.** Ese 20 % de test ya se observó en DT-1 (métricas de cinco
algoritmos por experimento). Algunas decisiones de este protocolo —atacar el
sobreajuste de los árboles, excluir la SVM— están motivadas por lo que DT-1 reportó
sobre ese test. La evaluación de DT-4 no será, por tanto, la de un test virgen: el
sesgo por grados de libertad del investigador no desaparece del todo, aunque la
selección en sí ya no mire el test. Se mantiene la misma partición para que la
comparación con DT-1 sea sobre las mismas filas.

## 3. Experimentos

| Clave | Target | Features | Uso |
|-------|--------|----------|-----|
| `riesgo_cv_con_pa` (A′ con PA) | `cardio` | `FEATURES_A_CON_PA` (12) | se sirve |
| `riesgo_cv_sin_pa` (A′ sin PA) | `cardio` | `FEATURES_A_SIN_PA` (10) | solo ablación |
| `hta_b1` (B1) | `hta = (ap_hi ≥ 140) ∨ (ap_lo ≥ 90)` | `FEATURES_B1` (10) | se sirve |

Las listas de features vienen de `src/cardio_features.py`. B1 volverá a pasar por
`assert_no_pressure_lineage`.

## 4. Validación cruzada

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sobre las filas de
  train de cada experimento, estratificado por su target.
- **Los mismos folds para todos los candidatos** de un experimento (comparación
  pareada). A′ con PA y A′ sin PA comparten partición y folds.
- **Escalado dentro de la CV.** En cada fold se ajustará un
  `Pipeline(StandardScaler, modelo)` **solo con las filas de entrenamiento del fold**:
  el scaler de un fold no ve nunca las filas de validación de ese fold. En la CV no
  existirá ningún scaler ajustado sobre todo el train; ese scaler se ajustará
  únicamente en el reajuste final del seleccionado (§10). Un test unitario lo
  verificará: que la media y la escala del scaler de cada fold coinciden con las de
  sus filas de entrenamiento y no con las del train completo.
- Por fold y candidato se registrarán: log-loss, Brier, AUC ROC y ECE (10 bins
  uniformes) sobre la parte de validación; log-loss sobre la parte de entrenamiento
  (diagnóstico de sobreajuste); tiempo de ajuste.

## 5. Candidatos

La SVM-RBF queda **excluida a priori**: en DT-1 consumió la mayor parte del tiempo
de cómputo (≈ 89 % de la corrida, sumando sus cuatro ajustes), no se distinguió de
los demás algoritmos y colapsó al azar en la variante de umbral estricto de B1.

Semilla `random_state=42` en todo estimador que la admita. La grilla incluye, en cada
familia, los hiperparámetros de DT-1 (marcados con \*), para que la comparación con
DT-1 quede dentro de la propia CV.

| Familia | Fijos | Grilla | Candidatos |
|---------|-------|--------|------------|
| Regresión logística (`LogisticRegression`) | `max_iter=2000`, solver por defecto (lbfgs) | `C` ∈ {0,01; 0,1; 1\*; 10} | 4 |
| Árbol de decisión (`DecisionTreeClassifier`) | — | `max_depth` ∈ {4; 6; 8\*} × `min_samples_leaf` ∈ {1\*; 50; 200} | 9 |
| Random Forest (`RandomForestClassifier`) | `n_estimators=300`, `n_jobs=-1` | `max_depth` ∈ {8; 12\*} × `min_samples_leaf` ∈ {1\*; 20; 100} | 6 |
| XGBoost (`XGBClassifier`) | `n_estimators=400`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `eval_metric="logloss"`, `n_jobs=-1` | `max_depth` ∈ {3; 5; 8\*} × `min_child_weight` ∈ {1\*; 20} | 6 |

Total: **25 candidatos por experimento**, 375 ajustes de CV más los reajustes finales.

## 6. Métrica de selección

**Log-loss media sobre los 5 folds de validación** (`sklearn.metrics.log_loss`,
`labels=[0, 1]`). Es una regla de puntuación propia: premia probabilidades bien
ordenadas *y* bien calibradas, que es lo que el servicio entrega. Brier, AUC y ECE se
reportarán, pero **no participarán en la selección**.

## 7. Regla de un error estándar

Por experimento:

1. `mejor` = candidato con la menor log-loss media en CV.
2. `EE` = desviación estándar muestral (`ddof=1`) de las 5 log-loss de `mejor` / √5.
3. `elegibles` = candidatos con log-loss media ≤ media(`mejor`) + `EE`.
4. `seleccionado` = el primer elegible en el **orden de simplicidad** (§8).

El EE es **aproximado**: los 5 folds comparten la mayor parte de sus datos de
entrenamiento y no son independientes, así que √5 subestima la varianza real de la
media. La regla se usa como criterio de parsimonia declarado, no como prueba de
hipótesis.

## 8. Orden de simplicidad

Orden total, declarado de antemano, de más simple a más complejo:

1. **Por familia:** regresión logística < árbol de decisión < Random Forest < XGBoost.
   Criterio: capacidad efectiva y auditabilidad del modelo servido. La regresión
   logística es un vector de coeficientes; el árbol, una sola estructura legible; el
   Random Forest, un promedio de árboles independientes; XGBoost, una suma secuencial de
   árboles con más hiperparámetros y una dependencia de servicio adicional (`xgboost`).
2. **Dentro de cada familia**, más regularización = más simple:
   - regresión logística: `C` ascendente;
   - árbol y Random Forest: `max_depth` ascendente y, a igual profundidad,
     `min_samples_leaf` descendente;
   - XGBoost: `max_depth` ascendente y, a igual profundidad, `min_child_weight`
     descendente.

## 9. Criterios de exclusión

- Un candidato cuyo ajuste lance una excepción, o cuya log-loss no sea finita, en
  **cualquier** fold, queda excluido de la selección. Se reporta con el error.
- Los avisos (p. ej. `ConvergenceWarning`) no excluyen: se registran por candidato y
  fold.
- Ningún candidato se excluirá por su rendimiento ni por mirar el test.

## 10. Ajuste final

El candidato seleccionado de cada experimento se reajustará **una vez** sobre todo el
train (80 %): `StandardScaler` sobre el train completo y modelo sobre el train
escalado. Es el único punto del protocolo en el que un scaler ve todo el train. Se
persistirán como dos artefactos (`modelo` y `scaler`), igual que en DT-1, para que la
API los cargue con el mismo mecanismo.

## 11. Evaluación en test

Estructuralmente, la función que evalúa en test **recibe el modelo ya seleccionado y
reajustado** y se llamará **una vez por experimento**, después de fijar la selección.
Un test unitario verificará que la CV no accede a las filas de test y que la
evaluación se invoca exactamente una vez por experimento.

Sobre el seleccionado se reportará:

- AUC ROC con IC 95 % bootstrap (1.000 remuestreos, semilla 42), PR-AUC con su baseline
  (prevalencia), log-loss, Brier con su baseline (predecir siempre la prevalencia de
  train), ECE (10 bins uniformes) y curva de calibración;
- accuracy, macro-F1 y matriz de confusión con umbral 0,5, como descripción de la
  decisión binaria (no como criterio);
- AUC y log-loss en train, para el diagnóstico de sobreajuste.

Análisis adicionales, todos sobre las predicciones del seleccionado y sin reentrenar
salvo donde se indica:

- **Ablación A′, en dos medidas preregistradas.** Ambas son evaluaciones en test
  fijadas de antemano, que **no intervienen en la selección**: se calculan después de
  fijar los seleccionados, y ningún resultado de ellas cambia qué se elige ni qué se
  sirve. En las dos, Δ AUC y su IC 95 % por bootstrap pareado sobre las mismas filas de
  test (1.000 remuestreos, semilla 42).
  - **Principal (controlada):** la configuración exacta (familia e hiperparámetros) del
    seleccionado con PA se reajustará sobre el mismo train **sin `ap_hi`/`ap_lo`**
    (`FEATURES_A_SIN_PA`, con su propio `StandardScaler` sobre el train) y se
    evaluará una vez en test. Δ AUC = AUC(seleccionado con PA) − AUC(misma
    configuración sin PA). Aísla el aporte de las dos columnas. No guarda artefactos.
  - **Secundaria (práctica):** seleccionado con PA frente a seleccionado sin PA
    (`riesgo_cv_sin_pa`, elegido por su propia CV). Mide la diferencia entre los dos
    sistemas que se podrían desplegar, que puede mezclar el aporte de la presión con
    un cambio de familia o de hiperparámetros.
  - No se reportarán deltas por algoritmo: exigirían evaluar en test candidatos no
    seleccionados.
- **Sensibilidad a presión implausible:** métricas del seleccionado excluyendo del test
  las filas fuera de PAS [70, 250] o PAD [40, 200].
- **Robustez de umbral de B1:** la **misma configuración** seleccionada para `hta` se
  reajustará sobre el mismo train con la etiqueta estricta
  `hta_gt = (ap_hi > 140) ∨ (ap_lo > 90)` y se evaluará una vez en test con esa
  etiqueta. Es un reporte de robustez, no una segunda selección; no guarda artefactos.

Comparación con DT-1: se reportarán junto a las de DT-1 del mismo experimento, sobre
las mismas filas de test. Se espera que las de DT-4 no las superen: las de DT-1 estaban
sesgadas al alza por selección sobre el test.

## 12. Lo que no se hará

- No se reelegirá ningún modelo ni hiperparámetro después de ver el test.
- No se ajustará el umbral de decisión.
- No se aplicará calibración post-hoc.
- No se modificarán `src/train_cardio_real.py`, `models/dt1_*.pkl` ni
  `models/dt1_manifest.json`: son el registro de DT-1. Al final de la corrida se
  verificarán sus sha256.

## 13. Artefactos y manifiesto

- `models/dt4_riesgo_cv_con_pa__{modelo,scaler}.pkl`,
  `models/dt4_riesgo_cv_sin_pa__{modelo,scaler}.pkl`,
  `models/dt4_hta_b1__{modelo,scaler}.pkl`.
- `models/dt4_manifest.json` (versión 2), con todo lo del manifiesto de DT-1 más:
  - `protocolo`: ruta y sha256 de este documento, y el commit que lo introdujo;
  - `codigo`: commit de HEAD en el momento de la corrida y si el árbol estaba limpio;
  - `entorno`: versiones de Python, scikit-learn, xgboost, pandas, numpy y **joblib**,
    más `pip_freeze` completo;
  - `dataset.sha256` del CSV;
  - `cv`: esquema de folds;
  - `grilla` y `orden_simplicidad` tal como se aplicaron;
  - por experimento: resultados **por fold** de cada candidato, media, sd y EE; la
    regla de 1 EE aplicada (`mejor`, `EE`, umbral, `elegibles`, `seleccionado` y el
    motivo); hiperparámetros del seleccionado; métricas en test; `mean_`/`scale_` del
    scaler; ruta y sha256 de los artefactos;
  - `ablacion` de A′ con las dos medidas por separado (`principal_controlada` y
    `secundaria_practica`): configuración usada, AUC de cada lado, Δ AUC, IC 95 % y
    número de remuestreos;
  - `desviaciones`: lista, vacía si no hubo ninguna;
  - tiempos por ajuste y duración total.

## 14. Reproducibilidad

- La corrida se lanzará con `python -m src.train_cardio_dt4`, en segundo plano, con log
  en `logs/dt4_run.log` (ignorado por git).
- El script **abortará** si el árbol de trabajo de `src/`, `docs/DT4_PROTOCOL.md` o los
  requirements tiene cambios sin commitear, para que el commit registrado describa
  exactamente el código ejecutado.
- Entorno: el de `requirements.lock.txt` (`make verify-env` en verde antes de lanzar).

## 15. Costo estimado

Objetivo: ≤ 60 minutos en la máquina de desarrollo (4 núcleos). La estimación de la
Fase 0, a partir de los tiempos de ajuste por modelo registrados en DT-1, queda muy por
debajo de ese límite. Si la corrida lo excediera, se reportará como desviación; la
grilla no se recortará a mitad de corrida.
