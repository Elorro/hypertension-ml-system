# DT-4 — Selección por validación cruzada y una sola evaluación en test

Registro de la corrida que cierra DT-4 (separar selección de evaluación) y DT-3
(validación cruzada). Sustituye a las cifras de [DT1_RESULTS.md](DT1_RESULTS.md) como
vigentes; las de DT-1 quedan como históricas.

- Protocolo preregistrado: [DT4_PROTOCOL.md](DT4_PROTOCOL.md), commit `0d8d4c9`
  (sha256 `6a2c70b2…978412ce`), commiteado **antes** de entrenar.
- Código: [`src/train_cardio_dt4.py`](../src/train_cardio_dt4.py), commit `ad518f2`,
  ejecutado con el árbol limpio.
- Evidencia completa: `models/dt4_manifest.json` (commit `41572ed`): resultados por
  fold de cada candidato, regla aplicada, métricas en test, entorno con `pip freeze`.
- Servicio migrado a estos modelos en `a7bf9ef`.

**Corrida:** 2026-09-25, 412 s, `seed=42`, Python 3.14.6 · scikit-learn 1.9.1 ·
xgboost 3.4.1 · pandas 2.3.3 · numpy 2.5.3 · joblib 1.6.0. Sin desviaciones del
protocolo, sin candidatos excluidos, sin avisos. Los archivos `dt1_*` no cambiaron
(sha256 idénticos antes y después).

---

## 1. Qué cambió respecto de DT-1

| | DT-1 | DT-4 |
|---|---|---|
| Selección | macro-F1 con umbral 0,5 **sobre el test** | log-loss media en CV de 5 folds **sobre el train** |
| Regla | máximo | 1 error estándar + orden de simplicidad declarado |
| Candidatos | 5 algoritmos, un juego de hiperparámetros | 25 por experimento (LR, árbol, RF, XGBoost); SVM excluida a priori |
| Escalado | `StandardScaler` sobre todo el train | por fold dentro de un `Pipeline`; sobre todo el train solo en el reajuste final |
| Test | evaluado 5 veces por experimento | evaluado **una vez**, sobre el seleccionado |

Partición y limpieza son las de DT-1 (68.678 filas; train 54.942, test 13.736),
verificadas por hash de índices al arrancar la corrida.

**Limitación declarada en el protocolo.** Ese test ya se había observado en DT-1, y
algunas decisiones de diseño (atacar el sobreajuste, excluir la SVM) salieron de lo que
DT-1 reportó sobre él. La selección de DT-4 no mira el test, pero el test no es virgen.

## 2. Validación cruzada y regla de un error estándar

Log-loss media ± sd sobre los 5 folds de validación. Tabla completa por fold en el
manifiesto.

| Experimento | Mejor en CV | EE | Umbral | Elegibles | **Seleccionado** | Mejor LR |
|---|---|---|---|---|---|---|
| A′ con PA | XGB(depth 3, mcw 20): 0,54125 | 0,00282 | 0,54408 | 6 | **RF(depth 12, hoja 20)**: 0,54234 | 0,57464 |
| A′ sin PA | XGB(depth 3, mcw 1): 0,62735 | 0,00119 | 0,62855 | 5 | **RF(depth 8, hoja 20)**: 0,62850 | 0,63154 |
| B1 | XGB(depth 3, mcw 1): 0,58678 | 0,00217 | 0,58895 | 7 | **RF(depth 8, hoja 100)**: 0,58805 | 0,59190 |

- En los tres experimentos el mínimo de log-loss es un XGBoost poco profundo, pero
  siempre hay un Random Forest dentro de 1 EE, y el RF va antes en el orden de
  simplicidad.
- **La regresión logística queda fuera de 1 EE en los tres.** En B1, su 0,59190 supera
  el umbral de 0,58895. La lectura de DT-1 «una regresión logística alcanza el techo»
  era cierta en AUC, no en log-loss.
- Las configuraciones de DT-1 sobreajustan de forma visible en la CV: árbol de
  profundidad 8 con hoja 1, 0,66237 en A′ con PA (el mejor árbol, 0,54776); RF de
  profundidad 12 con hoja 1, log-loss 0,509 en los folds de entrenamiento de B1 frente
  a 0,590 en validación.
- El EE es aproximado: los folds comparten datos de entrenamiento y no son
  independientes.

## 3. Métricas en test del seleccionado

Test n = 13.736. Comparación con DT-1 sobre **las mismas filas**.

| | A′ con PA: DT-4 / DT-1 | A′ sin PA: DT-4 / DT-1 | B1: DT-4 / DT-1 |
|---|---|---|---|
| Modelo | RF(12, 20) / RF(12, 1) | RF(8, 20) / SVM-RBF | RF(8, 100) / RF(12, 1) |
| AUC [IC 95 %] | 0,8019 [0,7938 · 0,8088] / 0,8017 | 0,7025 [0,6941 · 0,7112] / 0,6914 | 0,6955 [0,6866 · 0,7049] / 0,6941 |
| Log-loss | 0,5414 / 0,5416 | 0,6276 / 0,6376 | 0,5889 / 0,5899 |
| Brier (baseline) | 0,1807 (0,2500) / 0,1808 | 0,2190 (0,2500) / 0,2231 | 0,2015 (0,2254) / 0,2020 |
| ECE, 10 bins | 0,0141 / 0,0118 | 0,0170 / 0,0227 | 0,0079 / 0,0069 |
| PR-AUC (baseline) | 0,7858 (0,4948) | 0,6860 (0,4948) | 0,5327 (0,3433) |
| Accuracy, umbral 0,5 (baseline) | 73,49 % (50,52 %) | 64,84 % (50,52 %) | 69,02 % (65,67 %) |
| AUC train → test | 0,828 → 0,802 / 0,859 → 0,802 | 0,717 → 0,703 | 0,710 → 0,696 / 0,803 → 0,694 |
| `.pkl` del modelo | 26,3 MB / ≈ 60 MB | 8,5 MB / ≈ 4 MB | 5,5 MB / ≈ 70 MB |

Lectura:

1. **B1: la mejora real es el sobreajuste y el tamaño, no el AUC.** La brecha de AUC
   train/test pasa de 0,109 a 0,014 y el modelo de 70 a 5,5 MB. El AUC pasa de 0,6941 a
   0,6955, dentro del ruido (IC de ±0,009). La accuracy supera al baseline en +3,35 pp;
   con umbral 0,5 quedan 3.257 falsos negativos frente a 1.458 verdaderos positivos. La
   información sigue estando en el ranking, no en la decisión binaria.
2. **A′ con PA: igual en AUC, log-loss y Brier; el ECE empeora de 0,0118 a 0,0141.**
   No hay IC del ECE, así que no se califica esa diferencia.
3. **A′ sin PA mejora** (AUC +0,011, log-loss −0,010). No es sesgo de selección: el
   macro-F1 de DT-1 eligió la SVM, que tenía peor ranking. Es el problema que resuelve
   seleccionar con una regla de puntuación propia.
4. **El techo en AUC sin presión sigue siendo ≈ 0,69-0,70.**

**Sensibilidad.** Excluyendo del test las filas con presión implausible (24 en A′,
20 en B1), el AUC no se mueve: A′ 0,8019; B1 0,6955 → 0,6958.

**Robustez de umbral de B1.** La misma configuración, reajustada con
`hta_gt = (ap_hi > 140) ∨ (ap_lo > 90)`: AUC 0,6847 [0,6719 · 0,6963] (DT-1: 0,6657 con
el ganador de entonces, XGBoost). No es una segunda selección y no guarda artefactos.

## 4. Ablación de A′: cuánto vale medir la presión

Dos medidas preregistradas, calculadas después de fijar los seleccionados y sin
intervenir en la selección. Bootstrap pareado, 1.000 remuestreos, mismas filas de test.

| Medida | Δ AUC [IC 95 %] |
|---|---|
| **Principal (controlada):** RF(12, 20) con PA vs. la misma configuración sin `ap_hi`/`ap_lo` | **0,0991 [0,0919 · 0,1061]** |
| Secundaria (práctica): RF(12, 20) con PA vs. RF(8, 20) sin PA | 0,0994 [0,0923 · 0,1065] |
| *DT-1, histórica:* RF con PA vs. SVM sin PA | *0,1103 [0,1027 · 0,1181]* |

La cifra vigente es la principal: **Δ AUC = 0,0991 [0,0919 · 0,1061]**. El 0,1103 de
DT-1 estaba inflado: comparaba contra una SVM elegida por macro-F1, con peor ranking
que el RF. La medida controlada de DT-4 coincide con el Δ por algoritmo del RF que ya
reportaba DT-1 (0,1004). Las dos medidas de DT-4 prácticamente coinciden, así que el
aporte de la presión no depende del cambio de hiperparámetros.

## 5. Predicciones que no se cumplieron

- **El protocolo esperaba que DT-4 no superara a DT-1** («las de DT-1 estaban sesgadas
  al alza»). No se cumplió literalmente: en A′ con PA y en B1, DT-4 queda igual o
  marginalmente por encima (AUC +0,0002 y +0,0014, dentro del IC). Lectura: el sesgo
  por selección de DT-1 era pequeño porque los candidatos empataban, como ya advertía
  DT1_RESULTS §6. Se registra como resultado, sin reinterpretar el protocolo.
- **Predicción de Luis, fuera del protocolo: que en B1 se seleccionaría la regresión
  logística.** No se cumplió: queda fuera de 1 EE en log-loss (0,59190 frente al umbral
  de 0,58895).

## 6. Recursos del servicio

Medido en local con un venv solo con `requirements-serve.txt`: RSS de la API con A′ + B1
cargados, **352 → 202 MiB**; `.pkl` servidos (A′ + B1), **130 → 32 MB**. No hay
despliegue público.

## 7. Reproducibilidad

```bash
make verify-env    # versiones + sha256 + carga de los dt4_*.pkl
make train-dt4     # ~7 min, en segundo plano, log en logs/dt4_run.log
```

El script aborta con el árbol sucio o con un CSV distinto al de DT-1. Los tests
`tests/test_equivalencia_dt4.py` congelan los folds de CV y comprueban que la regla de
1 EE, reaplicada sobre los resultados guardados, reproduce la selección, y que el
protocolo no cambió desde la corrida.

---

*Aviso médico: este sistema es una demostración de ingeniería de ML. No es un
dispositivo médico ni sustituye la medición de presión arterial por personal
sanitario.*
