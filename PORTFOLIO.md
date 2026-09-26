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
- Selección de modelo **preregistrada** (DT-4): protocolo commiteado antes de entrenar;
  25 candidatos por experimento (regresión logística, árbol, Random Forest, XGBoost);
  log-loss en validación cruzada de 5 folds sobre el train; regla de un error estándar
  con orden de simplicidad declarado; test evaluado una sola vez. La SVM se excluyó a
  priori (≈ 89 % del cómputo de la corrida anterior, sin ventaja).
- Servicio de inferencia REST con **FastAPI**: `POST /v1/riesgo-cardiovascular` (A′) y
  `POST /v1/hipertension-sin-pa` (B1, sin clase), con endpoints de lote, validación del
  dominio de entrenamiento, contrato anti-leakage (B1 rechaza la presión con 422) y
  verificación de versiones y sha256 de los artefactos al arrancar.
- **Dashboard Streamlit** como cliente HTTP puro: paciente individual y lote por CSV,
  con formularios construidos desde el contrato de la API.
- 108 tests: contrato de API y dashboard en CI con un doble del modelo; integración con
  los `.pkl` reales, incluida la equivalencia bit a bit entre la matriz de la API y la
  del entrenamiento; garantías del protocolo verificadas por mutación (la CV no toca el
  test, cada evaluación en test ocurre una vez y después de su selección, el scaler de
  cada fold solo ve sus filas de entrenamiento).

**Resultado sobre datos reales (DT-4, `docs/DT4_RESULTS.md`).** Dos preguntas genuinas,
partición estratificada, selección por validación cruzada y una sola evaluación en test:
- *¿Cuánto vale medir la presión arterial?* Ablación controlada (la misma configuración
  del modelo con y sin `ap_hi`/`ap_lo`, bootstrap pareado sobre las mismas filas):
  **Δ AUC = 0,0991, IC 95 % [0,0919 · 0,1061]**. La cifra de la primera corrida (DT-1),
  0,1103, estaba inflada: comparaba contra una SVM elegida por macro-F1; la medida
  controlada coincide con el Δ del propio Random Forest que DT-1 ya reportaba (0,1004).
- *¿Se puede estimar el estado hipertensivo sin medirlo?* Con la presión excluida de las
  features: **AUC 0,6955 [0,6866 · 0,7049]**, ECE 0,0079 — hay señal real, pero la
  accuracy supera al baseline de clase mayoritaria en solo **+3,35 pp**. La información
  está en el ranking de riesgo, no en la decisión binaria. El techo en AUC sin presión
  sigue siendo ≈ 0,69-0,70.
- Frente a DT-1, la mejora en B1 no está en el AUC (0,6941 → 0,6955, dentro del ruido)
  sino en el **sobreajuste** (brecha train/test 0,109 → 0,014) y el **tamaño** del
  modelo (70 → 5,5 MB). En log-loss, la regresión logística queda fuera de 1 EE
  (0,59190 frente a un umbral de 0,58895): en AUC empataba, en probabilidad no.

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
  su baseline y a la limitación del test ya observado, codificación de sexo inferida, ausencia de
  análisis de equidad y consideraciones de privacidad (Ley 1581 de 2012)— y
  `docs/MODEL_CARD_SINTETICO.md`, histórica, para el modelo sintético.

**Lo que sigue abierto, y se dice.** El test de DT-4 ya se había observado en DT-1
(limitación declarada en el protocolo); no hay validación externa ni análisis de
equidad; el pipeline heredado de la raíz sigue sin resolverse; no hay despliegue público
(RSS medido de la API: 202 MiB; `.pkl` servidos: 32 MB). Todo priorizado en
`docs/ROADMAP.md`.

**Stack.** Python · scikit-learn · XGBoost · pandas · FastAPI · Pydantic · Streamlit · GitHub Actions.
