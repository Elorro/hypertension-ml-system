# hypertension-ml — ficha de portafolio

**Qué es.** Sistema end-to-end de ML sobre datos clínicos reales que devuelve
**probabilidades**: de enfermedad cardiovascular a partir de 12 variables (edad, sexo,
talla, peso, IMC, colesterol, glucosa, tabaco, alcohol, actividad y presión arterial), y
—como experimento— de hipertensión estimada **sin medir la presión**, con las otras 10.
Demostración de ingeniería de ML; no es herramienta clínica.

**Datos.** Dos pipelines. El que alimenta al servicio (`src/train_cardio_real.py`)
entrena sobre el dataset real *Cardiovascular Disease* de Kaggle (70.000 pacientes →
68.678 tras limpieza declarada a priori) y es el que cierra DT-1. El original, sobre un
dataset **sintético** de 50.000 registros (`src/generate_dataset.py`, `seed=42`), se
conserva como evidencia reproducible del target leakage.

**Pipeline.**
- EDA sobre datos reales → limpieza declarada a priori → entrenamiento.
- Comparación de 5 algoritmos (Regresión Logística, SVM-RBF, Árbol de Decisión, Random
  Forest, XGBoost) con selección por **macro-F1**; en B1 empatan dentro del ruido, y así
  se reporta.
- Servicio de inferencia REST con **FastAPI**: `POST /v1/riesgo-cardiovascular` (A′) y
  `POST /v1/hipertension-sin-pa` (B1, sin clase), con endpoints de lote, validación del
  dominio de entrenamiento, contrato anti-leakage (B1 rechaza la presión con 422) y
  verificación de versiones y sha256 de los artefactos al arrancar.
- **Dashboard Streamlit** como cliente HTTP puro: paciente individual y lote por CSV,
  con formularios construidos desde el contrato de la API.
- 90 tests: contrato de API y dashboard en CI con un doble del modelo; integración con
  los `.pkl` reales, incluida la equivalencia bit a bit entre la matriz de la API y la
  del entrenamiento.

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
  también son features. Una regla `if` determinista, sin entrenar
  (`scripts/verify_leakage.py::regla_completa`), recupera el 91,1 % de las etiquetas
  del dataset sintético; XGBoost llega a 95,3 % y un solo árbol de profundidad 8 a
  94,8 %. La brecha árboles (~95 %) vs. modelos lineales (48–69 %) delata que el target
  es una partición por umbrales. Conclusión: las métricas son tautológicas y el baseline
  correcto es la regla clínica, no la clase mayoritaria (29,6 %). Documenta además
  defectos secundarios (escalado antes del split, split sin estratificar, selección y
  evaluación sobre el mismo test) y un plan de corrección priorizado.
- *Criterio de aceptación con control positivo* (`scripts/audit_cardio_leakage.py`): busca
  reglas deterministas —umbral univariado y árbol CART propio, sin sklearn— sobre el dataset
  real. En la configuración real supera al baseline en +2,28 pp (límite: 5 pp); devolviéndole
  `ap_hi`/`ap_lo` a las features, la misma maquinaria reconstruye el target al 99,27 %
  (+33,60 pp). Ese control es lo que hace que el «pasa» signifique algo.
- *Model cards* (formato Mitchell et al. 2019): `docs/MODEL_CARD.md` para A′ y B1 —uso
  previsto educativo, usos desaconsejados (diagnóstico, triaje, seguros), métricas junto a
  su baseline y a la nota de sesgo por DT-4, codificación de sexo inferida, ausencia de
  análisis de equidad y consideraciones de privacidad (Ley 1581 de 2012)— y
  `docs/MODEL_CARD_SINTETICO.md`, histórica, para el modelo sintético.

**Lo que sigue abierto, y se dice.** La selección del ganador comparte conjunto de test
con la evaluación (sesgo al alza, DT-4); no hay validación cruzada, validación externa ni
análisis de equidad; el pipeline heredado de la raíz sigue sin resolverse; no hay
despliegue público. Todo priorizado en `docs/ROADMAP.md`.

**Stack.** Python · scikit-learn · XGBoost · pandas · FastAPI · Pydantic · Streamlit · GitHub Actions.
