# Model Card — Riesgo cardiovascular (A′) e hipertensión sin presión arterial (B1)

Formato basado en *Model Cards for Model Reporting* (Mitchell et al., 2019).

> **Alcance.** Describe los dos modelos que sirve la API desde `a7bf9ef`
> (`models/dt4_riesgo_cv_con_pa__*.pkl` y `models/dt4_hta_b1__*.pkl`), seleccionados en
> DT-4 sobre el dataset real. Toda cifra vigente sale de `models/dt4_manifest.json`; el
> análisis completo está en [DT4_RESULTS.md](DT4_RESULTS.md). Las cifras de DT-1
> ([DT1_RESULTS.md](DT1_RESULTS.md)) aparecen aquí solo como históricas. El modelo
> sintético anterior tiene su propia ficha histórica:
> [MODEL_CARD_SINTETICO.md](MODEL_CARD_SINTETICO.md).

---

## Detalles del modelo

| Campo | A′ `riesgo_cv_con_pa` | B1 `hta_b1` |
|-------|-----------------------|-------------|
| **Rol** | Principal | Experimental |
| **Target** | `cardio`: enfermedad cardiovascular, etiqueta original del dataset | `hta = (ap_hi ≥ 140) ∨ (ap_lo ≥ 90)`, umbral JNC7/ESC fijado a priori |
| **Features** | 12: edad, sexo, talla, peso, IMC, colesterol, glucosa, tabaco, alcohol, actividad, `ap_hi`, `ap_lo` | Las mismas 10 sin `ap_hi`/`ap_lo` (verificado por linaje) |
| **Algoritmo servido** | Random Forest (300 árboles, `max_depth=12`, `min_samples_leaf=20`), 26,3 MB | Random Forest (300 árboles, `max_depth=8`, `min_samples_leaf=100`), 5,5 MB |
| **Salida de la API** | Probabilidad + clase con umbral 0,5 explícito | Solo probabilidad |

Comunes a ambos:

- **Desarrollador:** Luis Araque. **Licencia:** MIT.
- **Corrida de referencia:** 2026-09-25, `src/train_cardio_dt4.py` (`ad518f2`),
  `seed=42`, Python 3.14.6 · scikit-learn 1.9.1 · numpy 2.5.3 · joblib 1.6.0.
- **Preprocesamiento:** `StandardScaler` ajustado solo sobre train (dentro de cada fold
  en la validación cruzada). El IMC se calcula como `weight / (height/100)²`; la edad,
  como `age / 365.25` (años, float).
- **Selección (DT-4, protocolo preregistrado `0d8d4c9`):** 25 candidatos por experimento
  (regresión logística, árbol, Random Forest, XGBoost; SVM excluida a priori), elegidos
  por log-loss media en validación cruzada de 5 folds **sobre el train** con la regla de
  un error estándar y un orden de simplicidad declarado. El test se evaluó **una vez**,
  sobre el seleccionado. En los dos casos el mínimo de log-loss era un XGBoost poco
  profundo, con el Random Forest dentro de 1 EE.
- **Probabilidades:** sin calibración post-hoc; su calibración se mide con ECE
  (10 bins uniformes) en test.

## Uso previsto

**Uso primario:** demostración de ingeniería de ML: entrenamiento auditado sobre datos
reales, servicio con contrato validado, y consumo desde una interfaz.

**Usuarios previstos:** estudiantes y desarrolladores que evalúan la arquitectura o la
metodología.

**Fuera de alcance — usos explícitamente desaconsejados:**

- Diagnóstico, tamizaje o triaje de pacientes reales. B1 en particular **no sustituye
  la medición de la presión arterial**.
- Cualquier decisión clínica, de tratamiento o de seguimiento.
- Uso en contextos de seguros, empleo o cualquier decisión que afecte a personas.
- Entradas fuera del dominio de entrenamiento: la API las rechaza con 422 (edad
  [29, 65] años, talla [120, 220] cm, peso [30, 200] kg, IMC [12, 70] kg/m², PAS
  [70, 250] y PAD [40, 200] mmHg con PAD < PAS).

## Factores

**Grupos evaluados:** ninguno. No hay análisis de desempeño desagregado por sexo ni
por grupo etario (DT-15).

**Codificación de `gender`:** el dataset usa 1/2 sin documentar su significado. Que
2 = hombre es una **inferencia** (talla media 169,9 vs. 161,4 cm, registrada en el
manifiesto), no un dato de la fuente.

**Instrumentación:** se desconoce cómo se midieron la presión y el resto de variables.
El CSV contiene presiones imposibles (PAS de hasta 16.020, PAD negativas); la limpieza
descarta la presión invertida y deja 92 filas implausibles, que no alteran las
métricas (análisis de sensibilidad abajo). Colesterol y glucosa son ordinales (1–3),
no valores de laboratorio. Tabaco, alcohol y actividad son autorreportados.

## Métricas

Test estratificado, n = 13.736; train n = 54.942.

| Métrica | A′ (con presión) | B1 (sin presión) |
|---------|------------------|------------------|
| AUC ROC [IC 95 % bootstrap] | 0,8019 [0,7938 · 0,8088] | 0,6955 [0,6866 · 0,7049] |
| AUC en train | 0,8276 | 0,7097 |
| Log-loss | 0,5414 | 0,5889 |
| PR-AUC (baseline = prevalencia) | 0,7858 (0,4948) | 0,5327 (0,3433) |
| Brier (baseline: predecir la prevalencia) | 0,1807 (0,2500) | 0,2015 (0,2254) |
| ECE, 10 bins uniformes | 0,0141 | 0,0079 |
| Accuracy con umbral 0,5 (baseline clase mayoritaria) | 73,49 % (50,52 %) | 69,02 % (65,67 %) |
| Matriz de confusión, umbral 0,5 (VN · FP · FN · VP) | 5.463 · 1.476 · 2.165 · 4.632 | 8.023 · 998 · 3.257 · 1.458 |

Cómo leerlas:

- **B1 tiene señal, y es modesta.** El IC del AUC está lejos de 0,5, pero la accuracy
  supera al baseline en solo +3,35 pp y con umbral 0,5 deja 3.257 falsos negativos
  frente a 1.458 verdaderos positivos. La información está en el ranking. Por eso la
  API no devuelve clase para B1. El techo en AUC sin presión sigue siendo ≈ 0,69-0,70.
- **La regresión logística no está dentro de 1 EE.** En AUC empataba con el resto
  (DT-1); en log-loss queda fuera en los dos experimentos (B1: 0,59190 frente a un
  umbral de 0,58895). El Random Forest servido no «ganó»: es el más simple de los
  candidatos dentro de 1 EE del mínimo.
- **Frente a DT-1 (histórico), la mejora en B1 es de sobreajuste y tamaño, no de AUC.**
  La brecha train → test del AUC pasa de 0,109 a 0,014 y el modelo de 70 a 5,5 MB; el
  AUC pasa de 0,6941 a 0,6955, dentro del ruido. En A′ la brecha pasa de 0,057 a 0,026 y
  el modelo de ≈ 60 a 26,3 MB; AUC, log-loss y Brier quedan iguales, y **el ECE empeora
  de 0,0118 a 0,0141** (no hay IC del ECE).
- **Cuánto vale medir la presión:** en la ablación controlada de A′ (la misma
  configuración con y sin `ap_hi`/`ap_lo`, mismas filas, bootstrap pareado), quitar la
  presión cuesta **Δ AUC = 0,0991 [0,0919 · 0,1061]**. La cifra histórica de DT-1,
  0,1103, estaba inflada: comparaba contra una SVM elegida por macro-F1. La medida
  controlada coincide con el Δ del propio Random Forest que DT-1 ya reportaba (0,1004).
- **Sensibilidad:** excluyendo de test las filas con presión implausible, el AUC no se
  mueve (A′ 0,8019 sin 24 filas; B1 0,6955 → 0,6958 sin 20).

**Límites del reporte:** la selección no miró el test, pero ese test ya se había
observado en DT-1, y el diseño de DT-4 (atacar el sobreajuste, excluir la SVM) se
informó con lo que DT-1 vio en él: no es un test virgen. El IC bootstrap acota el ruido
de muestreo del test, no el de partición; la variabilidad entre folds está en el
manifiesto.

## Datos de entrenamiento

*Cardiovascular Disease Dataset* (Kaggle), 70.000 filas → 68.678 tras una limpieza
declarada a priori: 24 duplicados exactos, 1.236 filas con presión invertida
(`ap_lo ≥ ap_hi`) y 62 con antropometría imposible. Prevalencias: `cardio` 49,48 %,
`hta` 34,33 %. Detalle en [DATA.md](DATA.md) y [DT1_RESULTS.md](DT1_RESULTS.md) §2.

Criterio de aceptación de DT-1 (`scripts/audit_cardio_leakage.py`): sin la presión,
ninguna regla determinista supera al baseline por más de +2,28 pp; con la presión
(control positivo), la misma maquinaria reconstruye `hta` al 99,27 %.

## Datos de evaluación

El 20 % estratificado del mismo dataset. No hay validación externa ni temporal: un
solo dataset, de una sola procedencia.

## Consideraciones éticas

**Riesgo clínico.** Una probabilidad presentada sin contexto puede inducir confianza
injustificada. Por eso cada respuesta de la API lleva `prevalencia_base` y un `aviso`
de no uso clínico, B1 va rotulado como experimental y sin clase, y el dashboard
muestra la advertencia en cada vista.

**Equidad.** Sin análisis por subgrupo, y con el sexo codificado por inferencia. Es
obligatorio antes de cualquier uso que afecte a personas (DT-15).

**Privacidad.** El servicio no persiste las peticiones: procesa en memoria y responde.
Un despliegue real con datos de pacientes quedaría sujeto a la normativa de datos
sensibles de salud (en Colombia, Ley 1581 de 2012 y sus decretos reglamentarios).

## Advertencias y recomendaciones

1. **No usar en contexto clínico.** Sin excepción.
2. **No citar las métricas sin su baseline** ni sin la nota de que el test ya se había
   observado en DT-1.
3. **Antes de cualquier uso serio:** análisis de equidad (DT-15), validación externa y
   un test no observado previamente.
4. **No hay despliegue público.** El servicio corre en local; RSS medido de la API con
   los dos modelos: 202 MiB.

---

*Para dudas o correcciones sobre esta model card: abrir un issue en el repositorio.*
