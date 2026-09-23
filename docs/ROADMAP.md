# Roadmap y deuda técnica

Trabajo pendiente, ordenado por severidad. Cada entrada indica el problema, su
efecto real y el criterio de aceptación que la cierra.

**Leyenda de prioridad:** 🔴 bloqueante · 🟡 importante · 🟢 mejora. Estado: ✅ resuelto · ◐ parcial (en el diagrama final)

---

## Fase 1 — Validez estadística (bloqueante)

Ninguna conclusión del proyecto es defendible hasta cerrar esta fase.

### ✅ DT-1 · Eliminar el target leakage — *resuelto (2026-09-17)*

**Problema.** El target del dataset sintético era una función determinista de las
features de entrada. Ver [LEAKAGE_ANALYSIS.md](LEAKAGE_ANALYSIS.md).

**Qué se hizo.** El entrenamiento migró al dataset real
(`data/real/cardio/cardio_train.csv`, 70.000 pacientes → 68.678 tras limpieza) en
[`src/train_cardio_real.py`](../src/train_cardio_real.py), con dos experimentos:

- **A′** — target `cardio`, con ablación de `ap_hi`/`ap_lo`. Medir la presión vale
  Δ AUC = 0,1103, IC 95 % [0,1027 · 0,1181] (bootstrap pareado).
- **B1** — target `hta = (ap_hi ≥ 140) ∨ (ap_lo ≥ 90)` con las dos columnas de
  presión **excluidas de las features**. AUC 0,6941 [0,6849 · 0,7035].

**Criterio de aceptación — cumplido.** `python scripts/audit_cardio_leakage.py`:
la mejor regla determinista sobre las features (árbol CART de profundidad ≤ 3)
alcanza 67,95 % frente a un baseline de clase mayoritaria de 65,67 %: **+2,28 pp**,
por debajo del límite de 5 pp. El control positivo de la misma auditoría —las mismas
reglas con `ap_hi`/`ap_lo` devueltas a las features— llega a 99,27 % (+33,60 pp),
que es lo que demuestra que el buscador de reglas sí detecta leakage cuando existe.

Resultados completos y sus límites: [DT1_RESULTS.md](DT1_RESULTS.md).

**Lo que NO cerraba.** El servicio seguía cargando el modelo sintético; se migró a
los artefactos `dt1_*` en `6e59035` (2026-09-23). Las métricas de arriba siguen
sesgadas al alza por DT-3/DT-4, que continúan abiertos.

### ✅ DT-2 · Corregir el escalado antes del split — *resuelto en el camino de servicio (2026-09-23)*

**Problema.** `src/train_classical_models.py` ejecuta `scaler.fit_transform(X)`
sobre el dataset completo y **después** hace `train_test_split`. La media y la
desviación del conjunto de test filtran al entrenamiento.

**Solución.** Encapsular escalado y modelo en un `sklearn.pipeline.Pipeline`, de
modo que `fit` solo vea el conjunto de entrenamiento. Resuelve además el riesgo de
desincronización entre `scaler.pkl` y el modelo en inferencia.

```python
pipe = Pipeline([("scaler", StandardScaler()), ("model", XGBClassifier(...))])
pipe.fit(X_train, y_train)  # el scaler se ajusta solo con train
joblib.dump(pipe, "models/modelo_xgb.pkl")  # un solo artefacto autocontenido
```

**Criterio de aceptación.** Ningún `fit` ni `fit_transform` se ejecuta sobre datos
que incluyan el conjunto de test.

**Estado parcial (2026-09-17).** `src/train_cardio_real.py` ya ajusta el scaler
exclusivamente sobre train y persiste el `StandardScaler` junto a sus `mean_`/`scale_`
en el manifiesto. `src/train_classical_models.py` sigue con el defecto. DT-2 se cierra
cuando el servicio deje de depender de ese script.

**Cierre (2026-09-23).** Desde `6e59035` el servicio carga solo los artefactos `dt1_*`.
`src/train_classical_models.py` conserva el defecto a propósito: forma parte de la
evidencia reproducible del leakage y no alimenta ningún servicio.

### 🔴 DT-3 · Split estratificado y validación cruzada

**Problema.** El split no usa `stratify=y`, y la selección del mejor modelo se hace
sobre un único split. La diferencia de macro-F1 entre modelos puede estar
íntegramente dentro del ruido de partición.

**Solución.** `StratifiedKFold(n_splits=5)` con `cross_val_score`, reportando
media ± desviación estándar. La selección se hace por la media de validación
cruzada, no por un valor puntual.

**Criterio de aceptación.** El reporte de métricas incluye desviación estándar entre
folds, y el modelo elegido supera al segundo por más de una desviación estándar —
o se documenta explícitamente que la diferencia no es significativa.

**Estado parcial (2026-09-17).** La estratificación ya está (`stratify=` en ambos
experimentos de `train_cardio_real.py`) y hay IC 95 % bootstrap del AUC del ganador.
Falta la validación cruzada: sigue siendo una única partición, así que el IC acota el
ruido de muestreo del test, no el de partición. La segunda mitad del criterio sí está
documentada: en B1 los cinco algoritmos quedan dentro del ruido (Random Forest 0,6941
vs. regresión logística 0,6924, con IC de ancho ±0,009) y así se reporta en
[DT1_RESULTS.md](DT1_RESULTS.md).

### 🔴 DT-4 · Separar selección de evaluación

**Problema.** El mismo conjunto de test elige el modelo ganador y reporta su
desempeño. El F1 publicado está sesgado al alza por selección.

**Solución.** Split en tres: entrenamiento (60 %), validación (20 %) para
seleccionar, y test (20 %) tocado **una sola vez**, al final.

**Criterio de aceptación.** El conjunto de test se evalúa exactamente una vez, con
el modelo ya elegido.

**Sin avance (2026-09-17).** `train_cardio_real.py` hereda el defecto: su
`criterio_seleccion` es `macro_f1 en test`. Es ahora el defecto estadístico más grave
que queda abierto en el pipeline real.

---

## Fase 2 — Consolidación del código

### 🟡 DT-5 · Unificar los dos pipelines — *parcial*

**Problema.** El repositorio contiene dos implementaciones paralelas e
incompatibles. Los scripts de la raíz (`data_pipeline.py`, `train_models.py`,
`model_utils.py`, `streamlit_app.py`, `chatbot_cli.py`) usan otro esquema de
columnas (`HTA_Nivel` vs `Diagnostico`, `Estrés` vs `Estres`), otra ruta de datos y
otro formato de artefacto. Un revisor no puede saber cuál es el sistema real.

Agravante: `train_models.py` **no ejecuta con scikit-learn ≥ 1.4**. Usa
`base_estimator` en `CalibratedClassifierCV` — renombrado a `estimator` en la 1.2,
eliminado en la 1.4 — y pasa `multi_class` a `LogisticRegression`, deprecado desde
la 1.5. El código de la raíz está muerto en cualquier entorno moderno.

**Opciones.**

- **(a) Eliminar la raíz.** Rápido y honesto. Se pierde la red neuronal Keras y el
  chatbot CLI, que son los únicos aportes exclusivos de esa rama.
- **(b) Migrar lo valioso.** Portar el modelo Keras y el CLI al esquema de `src/`,
  luego eliminar el resto. Más trabajo, conserva la comparación clásicos vs. red
  neuronal, que es un buen argumento de portafolio.

**Recomendación: (b).** El valor diferencial del repositorio está en la comparación
de familias de modelos.

**Criterio de aceptación.** Un solo esquema de columnas, un solo generador, un solo
script de entrenamiento. Ningún archivo huérfano en la raíz.

**Estado parcial (2026-09-23).** El camino de servicio ya no depende del pipeline
sintético: `api/` y `app/` sirven solo los modelos de DT-1 (`6e59035`, `a9b89ee`) y
el contrato `POST /predecir` se eliminó. El generador y `train_classical_models.py`
se conservan como evidencia del leakage. Sigue pendiente lo que define esta tarea: la
decisión (a)/(b) sobre los scripts de la raíz, que no se han tocado.

### 🟡 DT-6 · Suite de tests — *parcial*

**Problema.** No había tests. Ninguna garantía de no-regresión.

**Cobertura mínima propuesta.**

- `test_generate_dataset.py` — reproducibilidad de la semilla, shape, rangos, ausencia de nulos.
- `test_api.py` — `TestClient` de FastAPI: `GET /`, `POST /predecir` con payload
  válido, `422` con campo faltante, coherencia entre `diagnostico_numerico` y
  `argmax(probabilidades)`.
- `test_contract.py` — el orden de features del servicio coincide con el del
  entrenamiento. Este test previene el fallo silencioso más peligroso del sistema.

**Criterio de aceptación.** `pytest` verde en CI, cobertura > 70 % sobre `src/` y `api/`.

**Estado parcial (2026-09-23).** Hay suite (`86a5c29`, `6e59035`, `a9b89ee`): 90 tests
en local. Los de contrato de la API y del dashboard corren en CI con un doble del
modelo; los de integración (`requires_artifacts`, `requires_data`) se saltan en CI con
el motivo visible. El contrato de features está cubierto: la matriz de la API es
idéntica bit a bit a la del entrenamiento sobre el CSV. El plan de arriba (con
`/predecir`) quedó superado por el contrato nuevo. Falta el criterio de cobertura:
65 % sobre `src/` + `api/` en local, arrastrado por `generate_dataset.py` y
`train_classical_models.py` sin tests.

### ✅ DT-7 · Validación de rangos en el servicio — *resuelto (2026-09-23)*

**Problema.** El esquema Pydantic validaba tipos, no plausibilidad. `PAS: 900` se
aceptaba y devolvía una predicción sin sentido.

**Solución.** `Field(ge=..., le=...)` en cada campo, más un validador cruzado que
verifique `PAD < PAS` y la coherencia de `IMC` con `Peso`/`Talla`.

**Criterio de aceptación.** Entradas fisiológicamente imposibles devuelven `422` con
mensaje explicativo.

**Cierre (`6e59035`).** La entrada del contrato nuevo se valida contra el dominio de
entrenamiento (`src/cardio_features.py`): `Field(ge, le)` por campo, IMC calculado en
el servidor y validado, `ap_lo < ap_hi` y presión plausible. El 422 dice qué cota se
violó. Cubierto por los tests de contrato.

### ✅ DT-8 · Manejo de errores y `except` desnudo — *resuelto*

`src/train_classical_models.py` usaba `except:` sin tipo al calcular ROC-AUC,
silenciando cualquier excepción. Sustituido por `except ValueError` con aviso
explícito de la causa.

---

## Fase 3 — Robustez operativa

### ✅ DT-9 · Endpoint batch — *resuelto (2026-09-23)*

El dashboard emitía una petición HTTP por fila del CSV. Un archivo de 10.000
pacientes generaba 10.000 llamadas. Añadir `POST /predecir_batch` que acepte una lista
y responda con un array.

**Cierre.** `POST /v1/riesgo-cardiovascular/lote` y `POST /v1/hipertension-sin-pa/lote`
(`6e59035`): hasta 1.000 filas, errores por índice sin tumbar el lote. El dashboard
los usa por bloques (`a9b89ee`).

### 🟢 DT-10 · Logging estructurado

Sin trazabilidad de peticiones. Añadir logging JSON con `request_id`, latencia y
clase predicha — sin persistir datos clínicos de entrada.

### ✅ DT-11 · Fijar versiones de dependencias — *resuelto (2026-09-23)*

`requirements.txt` no fija versiones exactas. Un `pip install` hoy y en seis meses
producen entornos distintos. Generar un lock con `pip freeze > requirements.lock.txt`
tras validar que la suite pasa.

**Cierre (`f508c85`).** `requirements.lock.txt` fija el grafo completo; numpy, pandas,
scikit-learn y xgboost van con `==` a las versiones del manifiesto de DT-1, y
`make verify-env` es la compuerta. Límite declarado: fija versiones, no hashes.

### 🟡 DT-17 · `SVC(probability=True)` deprecado

**Problema.** `src/train_classical_models.py` usa `SVC(kernel="rbf", C=2,
probability=True)`. El parámetro `probability` quedó **deprecado en scikit-learn 1.9
y se elimina en la 1.11**. Como `requirements.txt` admite `scikit-learn<2.0`, el
pipeline se romperá solo cuando salga esa versión.

**Solución.** Sustituir por el reemplazo que la propia librería recomienda:

```python
from sklearn.calibration import CalibratedClassifierCV

svm = CalibratedClassifierCV(SVC(kernel="rbf", C=2), ensemble=False)
```

Ojo: cambia el tipo del objeto persistido en `modelo_svm.pkl`, así que hay que
regenerar el artefacto.

**Contexto de rendimiento.** El SVM-RBF domina el tiempo de entrenamiento: escala
como O(n²) y sobre las 40.000 filas de entrenamiento tarda ~4 minutos, frente a
segundos del resto de modelos. Medición: 0,60 s (n=2.000) → 2,23 s (n=4.000) →
9,07 s (n=8.000).

**Criterio de aceptación.** El entrenamiento completo no emite `FutureWarning` de
scikit-learn.

**Estado parcial (2026-09-17).** `train_cardio_real.py` ya usa
`CalibratedClassifierCV(SVC(...), method="sigmoid", cv=5, ensemble=False)`. Queda
`train_classical_models.py`. Coste medido en el dataset real: el SVM consume 462 s de
los 3.071 s totales en A′-con-PA y 1.273 s en la corrida de robustez — entre el 40 y
el 80 % del tiempo de cada experimento.

### 🟢 DT-12 · Contenerización

`Dockerfile` para el servicio y `docker-compose.yml` que levante servicio +
dashboard juntos.

### 🟢 DT-13 · Persistir métricas versionadas

`src/train_classical_models.py` imprime métricas a stdout y las pierde. Escribir
`reports/metrics.json` con timestamp, versión del dataset y hash del commit, para
poder comparar corridas.

---

## Fase 4 — Extensiones

### 🟢 DT-14 · Interpretabilidad

Valores SHAP sobre el modelo ganador, expuestos en el dashboard. En un dominio
clínico, una predicción sin explicación es inutilizable. Nota: solo tenía sentido
después de DT-1 — sobre el dataset sintético, SHAP se limitaría a confirmar que
`PAS` y `PAD` lo explican todo, que es la definición del target.

### 🟢 DT-15 · Análisis de equidad

Desempeño desagregado por sexo y grupo etario. Obligatorio antes de cualquier uso
que afecte a personas.

### 🟢 DT-16 · Calibración de probabilidades

El dashboard mostraba probabilidades como si fueran confianzas calibradas. Verificar
con curvas de calibración y aplicar `CalibratedClassifierCV` si hace falta.

**Estado parcial (2026-09-23).** La verificación existe para los modelos servidos:
curvas de calibración en el manifiesto y ECE de 0,0118 (A′) y 0,0069 (B1), medidos
sobre el mismo test que eligió al ganador (DT-4). No se aplicó calibración post-hoc.
La API y el dashboard dicen «probabilidad (ECE medido en test: …)», no «calibrada».

---

## Orden de ejecución sugerido

```
DT-1 ✅ ─► DT-2 ✅ ─► DT-3 ──► DT-4      Fase 1: sin esto, nada más importa
                              │
                              ▼
                    DT-5 ◐ ─► DT-6 ◐ ─► DT-7 ✅, DT-8 ✅      Fase 2
                                          │
                                          ▼
                              DT-9 ✅ … DT-13 (DT-11 ✅)     Fase 3
                                          │
                                          ▼
                              DT-14 … DT-16            Fase 4
```
