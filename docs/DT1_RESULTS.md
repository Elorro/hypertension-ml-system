# DT-1 — Resultados sobre el dataset real

Cierre del defecto central del proyecto: el entrenamiento deja de correr sobre el
dataset sintético con target leakage y pasa a `data/real/cardio/cardio_train.csv`
(70.000 pacientes, Kaggle *Cardiovascular Disease*).

- Script: [`src/train_cardio_real.py`](../src/train_cardio_real.py)
- Auditoría de aceptación: [`scripts/audit_cardio_leakage.py`](../scripts/audit_cardio_leakage.py)
- Evidencia completa de la corrida: `models/dt1_manifest.json` (métricas, hashes,
  versiones, curvas de calibración, tiempos)
- Contexto del defecto que se cierra: [LEAKAGE_ANALYSIS.md](LEAKAGE_ANALYSIS.md)

**Corrida de referencia:** 2026-09-17, 3.071 s (51 min), `seed=42`,
Python 3.14.6 · scikit-learn 1.9.1 · xgboost 3.4.1 · pandas 2.3.3 · numpy 2.5.3.
SHA-256 del CSV en el manifiesto.

---

## 1. Qué se preguntó

El dataset sintético hacía la pregunta trivial *«¿es hipertenso alguien cuya PAS es
≥ 140?»*, con PAS entre las features. Sobre datos reales se formulan dos preguntas
genuinas, y se entrenan los mismos 5 algoritmos en cada una:

| Experimento | Target | Features | Pregunta |
|---|---|---|---|
| **A′** (principal) | `cardio` (enfermedad cardiovascular) | 12 con presión / 10 sin presión | ¿Cuánto aporta medir la presión arterial? |
| **B1** (experimento) | `hta = (ap_hi ≥ 140) ∨ (ap_lo ≥ 90)` | 10, **sin presión** | ¿Se puede estimar el estado hipertensivo sin medir la presión? |

El umbral de `hta` es JNC7/ESC, fijado *a priori* — no se eligió mirando resultados.
Las features nunca incluyen `ap_hi`/`ap_lo` en B1, y el script lo verifica
estructuralmente por linaje: cada feature declara de qué columnas crudas deriva y
`assert_no_pressure_lineage` falla si alguna toca la presión o el target.

## 2. Limpieza de datos

Aplicada **antes** del split, sobre criterios declarados por adelantado:

| Paso | Criterio | Descartadas | Quedan |
|---|---|---|---|
| Duplicados exactos | idénticas en todo salvo `id` | 24 | 69.976 |
| Presión invertida | `ap_lo ≥ ap_hi` → descartar (no se intercambian columnas) | 1.236 | 68.740 |
| Antropometría imposible | talla ∉ [120, 220] cm, peso ∉ [30, 200] kg o IMC ∉ [12, 70] | 62 | **68.678** |

Se conserva el 98,11 % de las filas. Quedan 92 filas con presión implausible
(fuera de PAS ∈ [70, 250] o PAD ∈ [40, 200]) que la limpieza acordada no cubre; se
usan para un análisis de sensibilidad sobre test, no se eliminan. Prevalencias
resultantes: `cardio` 49,48 %, `hta` 34,33 %.

## 3. A′ — ¿cuánto vale medir la presión?

Misma partición estratificada 80/20 para ambas variantes, `StandardScaler` ajustado
solo sobre train. Selección por macro-F1 en test.

| Modelo | AUC con PA | AUC sin PA | Δ AUC |
|---|---|---|---|
| Regresión logística | 0,7821 | 0,6975 | 0,0846 |
| SVM-RBF | 0,7904 | 0,6914 | 0,0990 |
| Árbol (`max_depth=8`) | 0,7931 | 0,6908 | 0,1023 |
| **Random Forest** | **0,8017** | 0,7013 | 0,1004 |
| XGBoost | 0,7986 | 0,6940 | 0,1046 |

**Ablación (bootstrap pareado, 1.000 remuestreos, mismas filas):**
Δ AUC = **0,1103**, IC 95 % **[0,1027 · 0,1181]**.

Medir la presión arterial vale ~0,11 de AUC para predecir enfermedad cardiovascular
y el intervalo no se acerca a cero. Es el resultado cuantitativo más sólido del
proyecto, y es una ablación honesta: el mismo dato, la misma partición, lo único que
cambia es la presencia de dos columnas.

Ganador con PA (Random Forest, test n = 13.736): AUC 0,8017 [0,7937 · 0,8088],
PR-AUC 0,7874 (baseline por prevalencia 0,4948), accuracy 73,33 % (baseline
mayoritaria 50,52 %), Brier 0,1808 (baseline 0,2500), ECE 0,0118.

## 4. B1 — hipertensión sin medir la presión

Test n = 13.736, prevalencia 34,33 %, baseline de clase mayoritaria 65,67 %.

| Modelo | AUC test | AUC train | Macro-F1 | Accuracy | Brier | ECE |
|---|---|---|---|---|---|---|
| Regresión logística | 0,6924 | 0,6954 | 0,5867 | 68,94 % | 0,2026 | 0,0143 |
| SVM-RBF | 0,6605 | 0,6791 | 0,5812 | 68,99 % | 0,2088 | 0,0308 |
| Árbol (`max_depth=8`) | 0,6793 | 0,7143 | 0,5920 | 68,20 % | 0,2070 | 0,0256 |
| **Random Forest** | **0,6941** | 0,8027 | 0,6048 | 68,92 % | 0,2020 | 0,0069 |
| XGBoost | 0,6854 | 0,8301 | 0,6018 | 68,21 % | 0,2052 | 0,0258 |

**Lectura honesta, en tres puntos:**

1. **Hay señal, y es modesta.** AUC 0,6941 con IC 95 % bootstrap
   [0,6849 · 0,7035]: el intervalo está lejos de 0,5, así que la señal es real. Pero
   la accuracy (68,92 %) apenas supera al baseline de clase mayoritaria (65,67 %) en
   **+3,25 pp**. Con umbral 0,5 el modelo casi no clasifica: 3.156 falsos negativos
   contra 1.559 verdaderos positivos. La información está en el *ranking* de riesgo,
   no en la decisión binaria.
2. **Los cinco algoritmos empatan.** El IC del ganador mide ±0,009 de AUC; la
   diferencia entre Random Forest (0,6941) y regresión logística (0,6924) es 0,0017.
   Está enteramente dentro del ruido. Declarar «ganó Random Forest» sería sobreleer
   el resultado: lo defendible es que **el techo sin presión arterial es AUC ≈ 0,69 y
   una regresión logística lo alcanza**.
3. **Los hiperparámetros heredados sobreajustan.** Random Forest pasa de AUC 0,803
   en train a 0,694 en test; XGBoost, de 0,830 a 0,685. La regresión logística no se
   mueve (0,695 → 0,692). Los hiperparámetros venían del pipeline sintético, donde la
   flexibilidad servía para memorizar un `if`; aquí solo memorizan ruido.

Calibración del ganador: ECE 0,0069 sobre 10 bins uniformes, Brier 0,2020 frente a
0,2254 de predecir siempre la prevalencia. Las probabilidades son usables como
probabilidades — lo que no eran en el pipeline sintético.

**Robustez de umbral.** Reentrenado sobre la misma partición con el umbral estricto
`hta_gt = (ap_hi > 140) ∨ (ap_lo > 90)`, que cambia el 19,11 % de las etiquetas
positivas: AUC 0,6657 [0,6530 · 0,6775]. La conclusión no depende de la aritmética
del `≥`. Es un reporte de robustez, **no** una segunda elección de umbral: no se
guardaron artefactos de esa corrida. Nota: con prevalencia 14,77 % el SVM-RBF
colapsa (AUC 0,4979, indistinguible del azar) — la calibración de Platt sobre clases
desbalanceadas no aguanta.

**Sensibilidad.** Excluyendo de test las filas con presión implausible (20 filas),
las métricas del ganador no se mueven: AUC 0,6944 vs. 0,6941.

## 5. Criterio de aceptación de DT-1

El ROADMAP lo fija así: *ninguna regla determinista sobre las features recupera el
target por encima del baseline de clase mayoritaria + 5 pp.*

`python scripts/audit_cardio_leakage.py` busca reglas deterministas sin entrenar
modelos: mejor umbral univariado por feature y dirección, y un árbol CART propio de
profundidad ≤ 3 (gini, umbrales en deciles), ambos ajustados en train y evaluados en
test. Solo numpy y pandas.

```
                                      baseline   mejor regla   ganancia
B1 — hta SIN presión (real)            65,67 %      67,95 %     +2,28 pp   PASA
CONTROL POSITIVO — con ap_hi/ap_lo     65,67 %      99,27 %    +33,60 pp   FALLA (correcto)
```

El control positivo es la parte que hace válida a la auditoría: la misma maquinaria
que reporta +2,28 pp en la configuración real reconstruye el target casi perfecto
(99,27 %) en cuanto se le devuelven las dos columnas de presión. Un «pasa» sin ese
control no probaría nada sobre el poder del buscador de reglas.

La auditoría reimplementa la limpieza de forma independiente del script de
entrenamiento y verifica que llega a las mismas 68.678 filas.

**Linaje de las features de B1** (correlación de Spearman con la presión arterial):

| feature | AUC univariada | corr `ap_hi` | corr `ap_lo` |
|---|---|---|---|
| `bmi` | 0,6504 | 0,2809 | 0,2483 |
| `weight` | 0,6388 | 0,2778 | 0,2521 |
| `cholesterol` | 0,5930 | 0,2082 | 0,1659 |
| `age_years` | 0,5915 | 0,2225 | 0,1584 |
| `gluc` | 0,5343 | 0,1064 | 0,0820 |
| resto (`gender`, `alco`, `height`, `smoke`, `active`) | ≤ 0,5179 | ≤ 0,0633 | ≤ 0,0650 |

Ninguna feature supera |ρ| = 0,29 con la presión. El IMC es el mejor predictor
individual y aun así su AUC univariada es 0,65. Contraste con el dataset sintético,
donde `PAS` sola reconstruía la etiqueta por definición.

**Veredicto: DT-1 cerrado.** El problema es ahora genuinamente predictivo.

## 6. Lo que este resultado NO cierra

Defectos que siguen abiertos y que sesgan al alza las cifras de arriba:

- **DT-4 (selección = evaluación).** El ganador se elige por macro-F1 sobre el mismo
  test que después reporta sus métricas. Con 5 modelos empatados, eso sesga las
  cifras del ganador al alza; el sesgo es pequeño aquí precisamente porque están
  empatados, pero existe. Hace falta split en tres.
- **DT-3 (partición única).** El split es estratificado — eso sí se corrigió — pero
  sigue siendo uno solo, sin validación cruzada ni desviación estándar entre folds.
  El IC bootstrap del AUC acota el ruido de muestreo del test, no el de partición.
- **DT-2 en el pipeline heredado.** `src/train_cardio_real.py` ajusta el scaler solo
  sobre train, pero `src/train_classical_models.py` (el que alimenta a la API) sigue
  escalando antes del split.
- **El servicio sigue sirviendo el modelo sintético.** `api/main.py` y el dashboard
  cargan `models/modelo_*.pkl`, no los `dt1_*`. Mientras eso no cambie, lo que el
  sistema expone en producción sigue siendo el `if` tautológico.
- **Sin validación externa ni análisis de equidad.** Un solo dataset, de una sola
  procedencia, sin desagregar desempeño por sexo ni grupo etario (DT-15).

## 7. Reproducibilidad

```bash
make train-real    # ~51 min; el SVM-RBF se lleva 40 de ellos
make audit-real    # segundos; solo numpy + pandas
```

El manifiesto registra semilla, versiones, SHA-256 del CSV y SHA-256 de cada
artefacto `.pkl`. **Advertencia:** el entorno de la corrida de referencia ya no es
reproducible tal cual — a septiembre de 2026 el entorno local tiene pandas 3.0.3 y
carece de scikit-learn. Reproducir las cifras exactas exige fijar las versiones del
manifiesto (ver DT-11).

---

*Aviso médico: este sistema es una demostración de ingeniería de ML. No es un
dispositivo médico ni sustituye la medición de presión arterial por personal
sanitario.*
