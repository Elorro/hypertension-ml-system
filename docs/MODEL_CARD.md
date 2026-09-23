# Model Card — Riesgo cardiovascular (A′) e hipertensión sin presión arterial (B1)

Formato basado en *Model Cards for Model Reporting* (Mitchell et al., 2019).

> **Alcance.** Describe los dos modelos que sirve la API desde `6e59035`
> (`models/dt1_riesgo_cv_con_pa__*.pkl` y `models/dt1_hta_b1__*.pkl`), entrenados en
> DT-1 sobre el dataset real. Toda cifra sale de `models/dt1_manifest.json`; el
> análisis completo está en [DT1_RESULTS.md](DT1_RESULTS.md). El modelo sintético
> anterior tiene su propia ficha histórica: [MODEL_CARD_SINTETICO.md](MODEL_CARD_SINTETICO.md).

---

## Detalles del modelo

| Campo | A′ `riesgo_cv_con_pa` | B1 `hta_b1` |
|-------|-----------------------|-------------|
| **Rol** | Principal | Experimental |
| **Target** | `cardio`: enfermedad cardiovascular, etiqueta original del dataset | `hta = (ap_hi ≥ 140) ∨ (ap_lo ≥ 90)`, umbral JNC7/ESC fijado a priori |
| **Features** | 12: edad, sexo, talla, peso, IMC, colesterol, glucosa, tabaco, alcohol, actividad, `ap_hi`, `ap_lo` | Las mismas 10 sin `ap_hi`/`ap_lo` (verificado por linaje) |
| **Algoritmo servido** | Random Forest (300 árboles, `max_depth=12`) | Random Forest (300 árboles, `max_depth=12`) |
| **Salida de la API** | Probabilidad + clase con umbral 0,5 explícito | Solo probabilidad |

Comunes a ambos:

- **Desarrollador:** Luis Araque. **Licencia:** MIT.
- **Corrida de referencia:** 2026-09-17, `src/train_cardio_real.py`, `seed=42`,
  Python 3.14.6 · scikit-learn 1.9.1 · numpy 2.5.3.
- **Preprocesamiento:** `StandardScaler` ajustado solo sobre train. El IMC se calcula
  como `weight / (height/100)²`; la edad, como `age / 365.25` (años, float).
- **Selección:** 5 algoritmos (regresión logística, SVM-RBF, árbol, Random Forest,
  XGBoost), elegidos por macro-F1 **sobre el mismo test que reporta las métricas**
  (DT-4 abierto).
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
| AUC ROC [IC 95 % bootstrap] | 0,8017 [0,7937 · 0,8088] | 0,6941 [0,6849 · 0,7035] |
| AUC en train | 0,8585 | 0,8027 |
| PR-AUC (baseline = prevalencia) | 0,7874 (0,4948) | 0,5277 (0,3433) |
| Brier (baseline: predecir la prevalencia) | 0,1808 (0,2500) | 0,2020 (0,2254) |
| ECE, 10 bins uniformes | 0,0118 | 0,0069 |
| Accuracy con umbral 0,5 (baseline clase mayoritaria) | 73,33 % (50,52 %) | 68,92 % (65,67 %) |
| Matriz de confusión, umbral 0,5 (VN · FP · FN · VP) | 5.468 · 1.471 · 2.193 · 4.604 | 7.908 · 1.113 · 3.156 · 1.559 |

Cómo leerlas:

- **B1 tiene señal, y es modesta.** El IC del AUC está lejos de 0,5, pero la accuracy
  supera al baseline en solo +3,25 pp y con umbral 0,5 deja 3.156 falsos negativos
  frente a 1.559 verdaderos positivos. La información está en el ranking. Por eso la
  API no devuelve clase para B1.
- **Los algoritmos empatan.** En B1, Random Forest 0,6941 vs. regresión logística
  0,6924, con un IC de ±0,009. En A′, Random Forest 0,8017 vs. XGBoost 0,7986. Que el
  servido sea Random Forest no significa que «ganó»: lo defendible es el techo de AUC
  de cada problema.
- **Sobreajuste de los hiperparámetros heredados.** La brecha train → test del AUC es
  de 0,057 en A′ y de 0,109 en B1.
- **Cuánto vale medir la presión:** en la ablación de A′ (mismas filas, bootstrap
  pareado), quitar `ap_hi`/`ap_lo` cuesta Δ AUC = 0,1103 [0,1027 · 0,1181].
- **Sensibilidad:** excluyendo de test las filas con presión implausible, el AUC no se
  mueve (A′ 0,8017 → 0,8017 sin 24 filas; B1 0,6941 → 0,6944 sin 20).

**Sesgo en el reporte:** el mismo test eligió al ganador entre 5 algoritmos, así que
todas las cifras están sesgadas al alza por selección (DT-4). Es además una sola
partición: el IC bootstrap acota el ruido de muestreo del test, no el de partición
(DT-3).

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
2. **No citar las métricas sin su baseline** ni sin la nota de sesgo por DT-4.
3. **Antes de cualquier uso serio:** split train/validación/test (DT-4), validación
   cruzada (DT-3), análisis de equidad (DT-15) y validación externa.
4. **No hay despliegue público.** El servicio corre en local.

---

*Para dudas o correcciones sobre esta model card: abrir un issue en el repositorio.*
