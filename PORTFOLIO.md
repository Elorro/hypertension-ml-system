# hypertension-ml — ficha de portafolio

**Qué es.** Sistema end-to-end de ML que clasifica la presión arterial de un paciente en
4 niveles (Normal, Prehipertensión, HTA Grado 1, HTA Grado 2) y devuelve la distribución
de probabilidad sobre las cuatro clases, a partir de 15 variables clínicas y de estilo de
vida (edad, IMC, PAS/PAD, colesterol, glucosa, tabaquismo, ejercicio, estrés, herencia).
Demostración de ingeniería de ML; no es herramienta clínica.

**Datos.** Dos pipelines. El que alimenta al servicio entrena sobre un dataset
**sintético** de 50.000 registros (`src/generate_dataset.py`, `seed=42`). El segundo
(`src/train_cardio_real.py`) entrena sobre el dataset real *Cardiovascular Disease* de Kaggle
(70.000 pacientes → 68.678 tras limpieza declarada a priori) y es el que cierra DT-1.

**Pipeline.**
- EDA sobre datos reales → generación del dataset de entrenamiento.
- Comparación de 5 algoritmos (Regresión Logística, SVM-RBF, Árbol de Decisión, Random
  Forest, XGBoost) y selección automática por **macro-F1**; ganó XGBoost.
- Servicio de inferencia REST con **FastAPI** (`POST /predecir`, esquema Pydantic, OpenAPI).
- **Dashboard Streamlit** como cliente HTTP del servicio: paciente individual y lote por CSV.

**Resultado sobre datos reales (DT-1, `docs/DT1_RESULTS.md`).** Dos preguntas genuinas,
mismos 5 algoritmos en cada una, partición estratificada, scaler ajustado solo en train:
- *¿Cuánto vale medir la presión arterial?* Ablación con bootstrap pareado sobre las mismas
  filas: quitarle `ap_hi`/`ap_lo` al modelo de riesgo cardiovascular cuesta
  **Δ AUC = 0,1103, IC 95 % [0,1027 · 0,1181]**.
- *¿Se puede estimar el estado hipertensivo sin medirlo?* Con la presión excluida de las
  features: **AUC 0,6941 [0,6849 · 0,7035]**, ECE 0,0069 — hay señal real, pero la accuracy
  supera al baseline de clase mayoritaria en solo **+3,25 pp**. La información está en el
  ranking de riesgo, no en la decisión binaria.
- Los cinco algoritmos quedan **dentro del ruido entre sí** (RF 0,6941 vs. regresión
  logística 0,6924, IC de ancho ±0,009), y los hiperparámetros heredados sobreajustan
  (RF: 0,803 en train → 0,694 en test). Se reporta el empate, no un ganador.

**Rigor — lo que distingue al proyecto.**
- *Auditoría de target leakage* (`docs/LEAKAGE_ANALYSIS.md`, reproducible con
  `scripts/verify_leakage.py`): la etiqueta es una función determinista de PAS/PAD, que
  también son features. Una regla `if` de doce líneas, sin entrenar, recupera el 91,1 % de
  las etiquetas; XGBoost llega a 95,3 % y un solo árbol de profundidad 8 a 94,8 %. La brecha
  árboles (~95 %) vs. modelos lineales (48–69 %) delata que el target es una partición por
  umbrales. Conclusión: las métricas son tautológicas y el baseline correcto es la regla
  clínica, no la clase mayoritaria (29,6 %). Documenta además defectos secundarios
  (escalado antes del split, split sin estratificar, selección y evaluación sobre el mismo
  test) y un plan de corrección priorizado.
- *Criterio de aceptación con control positivo* (`scripts/audit_cardio_leakage.py`): busca
  reglas deterministas —umbral univariado y árbol CART propio, sin sklearn— sobre el dataset
  real. En la configuración real supera al baseline en +2,28 pp (límite: 5 pp); devolviéndole
  `ap_hi`/`ap_lo` a las features, la misma maquinaria reconstruye el target al 99,27 %
  (+33,60 pp). Ese control es lo que hace que el «pasa» signifique algo.
- *Model card* (`docs/MODEL_CARD.md`, formato Mitchell et al. 2019): uso previsto educativo,
  usos desaconsejados (diagnóstico, triaje, seguros), métricas presentadas junto al baseline
  de la regla, limitaciones de datos (features independientes, balance de clases artificial,
  sin validación externa), ausencia de análisis de equidad y consideraciones de privacidad
  (Ley 1581 de 2012).

**Lo que sigue abierto, y se dice.** El servicio todavía sirve el modelo sintético; la
selección del ganador comparte conjunto de test con la evaluación (sesgo al alza); no hay
validación cruzada, validación externa ni análisis de equidad. Todo priorizado en
`docs/ROADMAP.md`.

**Stack.** Python · scikit-learn · XGBoost · pandas · FastAPI · Pydantic · Streamlit · GitHub Actions.
