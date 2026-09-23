# Auditoría de Target Leakage

**Estado:** confirmado · **Severidad:** crítica · **Afecta:** ambos pipelines

Este documento describe el defecto metodológico central del proyecto, la evidencia
que lo respalda y sus consecuencias. Se publica de forma explícita porque un
resultado de ML no auditable no vale nada, y porque las métricas de este repositorio
serían engañosas sin este contexto.

---

## 1. El defecto

En `src/generate_dataset.py` la etiqueta se construye así (sin el bucle por fila;
misma lógica, con los mismos umbrales y ramas que
`scripts/verify_leakage.py::regla_completa`):

```python
if PAS < 120 and PAD < 80:
    diagnostico = 0  # Normal
elif 120 <= PAS < 130 or 80 <= PAD < 85:
    diagnostico = 1  # Prehipertensión
elif 130 <= PAS < 140 or 85 <= PAD < 90:
    diagnostico = 2  # HTA Grado 1
else:
    diagnostico = 3  # HTA Grado 2

# ajustes deterministas posteriores
riesgo_extra = (IMC > 32) + (Colesterol > 240) + (Glucosa > 140) + (Estres > 7) + (Herencia == 1)
if riesgo_extra >= 3 and diagnostico < 3:
    diagnostico += 1
if Ejercicio >= 6 and diagnostico > 0:
    diagnostico -= 1
```

Y en `src/train_classical_models.py` el conjunto de features es:

```python
X = df.drop("Diagnostico", axis=1)  # incluye PAS, PAD, IMC, Colesterol, Glucosa, Estres, Herencia, Ejercicio
```

Es decir: **todas las variables que determinan la etiqueta se entregan al modelo
como entrada.** No hay ningún componente aleatorio en la generación del target.

El mismo patrón aparece en el pipeline heredado de la raíz, en `data_pipeline.py`,
donde `np.select` define `HTA_Nivel` a partir de `pas` y `pad`.

## 2. Evidencia

`scripts/verify_leakage.py` reconstruye la etiqueta a partir de las columnas del
CSV publicado, sin entrenar ningún modelo:

```
$ python scripts/verify_leakage.py

Dataset: data/raw/dataset_hipertension_sintetico.csv  (50,000 filas)
Distribución de clases:  {0: 0.2619, 1: 0.2961, 2: 0.2681, 3: 0.1739}

Baseline clase mayoritaria                →  29.61%
Regla PAS/PAD pura                        →  61.87% de coincidencia
Regla completa (con ajustes de riesgo)    →  91.09% de coincidencia
```

**Interpretación.** Una regla `if` determinista, sin entrenar
(`scripts/verify_leakage.py::regla_completa`), recupera el 91,1 % de las etiquetas
del dataset sintético. El 8,9 % restante no es señal aprendible: es **ruido de
redondeo**. El generador clasifica usando los valores en punto flotante (`pas[i]`,
`imc[i]`) pero guarda en el CSV versiones redondeadas (`pas.round(0)`,
`imc.round(1)`), lo que desplaza filas cercanas a las fronteras de decisión. Sobre
los valores sin redondear, la reconstrucción sería exacta.

Consecuencia directa: **≈91 % es el techo de información útil del dataset, y ese
techo lo alcanza un `if`.** Cualquier modelo que reporte accuracy en ese rango no
ha aprendido nada que no estuviera ya escrito en el generador.

Es el defecto que DT-1 cerró. Sobre el dataset real, sin la presión en las features,
la mejor regla determinista que encuentra `scripts/audit_cardio_leakage.py` supera al
baseline de clase mayoritaria en solo +2,28 pp, mientras que con `ap_hi`/`ap_lo`
(control positivo) reconstruye el target al 99,27 %.

## 3. Confirmación empírica

Entrenando efectivamente los cinco modelos sobre el dataset (`make train`,
scikit-learn 1.9, split 80/20):

| Modelo | Accuracy | Macro-F1 | ROC-AUC OVR |
|--------|----------|----------|-------------|
| Regresión Logística | 47,70 % | 0,4711 | 0,748 |
| SVM RBF | 68,82 % | 0,6961 | 0,913 |
| Árbol de Decisión (`max_depth=8`) | 94,83 % | 0,9497 | 0,997 |
| Random Forest (300 árboles) | 94,84 % | 0,9498 | 0,998 |
| XGBoost (400 árboles) | **95,32 %** | **0,9547** | 0,998 |

Dos patrones confirman el diagnóstico:

**a) La brecha entre familias de modelos.** Los métodos basados en árboles alcanzan
~95 %; los lineales y de kernel se quedan en 48–69 %. Esto es exactamente lo que
predice la estructura del target: una partición por umbrales sobre `PAS`/`PAD`. Los
árboles representan cortes rectangulares de forma nativa; un modelo lineal no puede
expresar esa geometría. La forma del problema **es** un `if`, y la tabla lo delata.

**b) Un árbol de profundidad 8 empata con 400 árboles boosteados.** Un solo
`DecisionTreeClassifier` logra 94,83 %, y XGBoost con 400 estimadores solo añade
0,49 puntos. Cuando el gradient boosting no consigue despegarse de un modelo
trivial, no queda estructura estadística que extraer: solo una regla que memorizar.

**Nota sobre el techo real.** Los modelos superan el 91,09 % de la reconstrucción
manual de §2 porque esa reconstrucción arrastra el ruido de redondeo del CSV,
mientras que el modelo aprende directamente sobre los valores redondeados que
observa. El 91,09 % es el techo de la reconstrucción, no del problema; el techo
efectivo ronda el 95–96 %, limitado por los casos que el redondeo vuelve ambiguos
en las fronteras de decisión.

## 4. Por qué esto invalida las métricas

Un accuracy de 0.95 en este dataset no significa "el sistema detecta hipertensión
con 95 % de acierto". Significa "el sistema memorizó los umbrales de las guías AHA".
Las implicaciones:

- **No hay generalización que medir.** No existe distribución poblacional real de la
  cual el modelo esté aprendiendo; se aprende una función determinista conocida.
- **El baseline correcto no es la clase mayoritaria** (29,6 %), sino la regla clínica
  misma (91,1 % reconstruida a mano, ~95 % como techo efectivo). El 95,3 % de XGBoost
  se lee contra ese baseline, no contra el 29,6 %: la ganancia real es de unos pocos
  puntos sobre un `if`, no de 66 puntos sobre el azar.
- **La importancia de variables es tautológica.** `PAS` y `PAD` dominarán siempre,
  no porque sean predictores clínicos descubiertos, sino porque son la definición.
- **El modelo no aporta valor sobre un `if`.** Si la etiqueta se computa con dos
  comparaciones, desplegar un XGBoost de 400 árboles para reproducirlas es
  ingeniería innecesaria.

## 5. Defectos metodológicos secundarios

Encontrados durante la misma auditoría, en orden de severidad:

| # | Defecto | Ubicación | Efecto |
|---|---------|-----------|--------|
| 1 | Escalado antes del split | `train_classical_models.py` — `scaler.fit_transform(X)` sobre todo `X`, luego `train_test_split` | Leakage de estadísticos (media/σ) del test al entrenamiento. Sesgo optimista, pequeño con n=50k pero incorrecto. |
| 2 | Split sin estratificar | `train_test_split(..., random_state=42)` sin `stratify=y` | Proporciones de clase distintas entre train y test; afecta la comparabilidad del macro-F1. |
| 3 | Sin validación cruzada | Todo el entrenamiento | Selección del "mejor modelo" sobre un único split. La diferencia de F1 entre modelos puede estar dentro del ruido del split. |
| 4 | Selección sobre el test | El mismo conjunto elige el modelo y reporta su desempeño | El F1 reportado del ganador está sesgado al alza. Faltaría un split train/val/test. |
| 5 | `except:` desnudo | `train_classical_models.py`, cálculo de ROC-AUC | Silencia cualquier excepción, incluidas las no previstas. |
| 6 | Sin tests | Todo el repositorio | Ninguna garantía de no-regresión. |

## 6. Cómo se corrige

El plan está en [ROADMAP.md](ROADMAP.md). En resumen, dos caminos válidos:

**Camino A — Reformular el problema sobre datos reales.**
Usar `data/real/cardio/cardio_train.csv` (70.000 pacientes reales), definir la
etiqueta de hipertensión a partir de `ap_hi`/`ap_lo`, y **excluir `ap_hi` y `ap_lo`
de las features**. La pregunta pasa a ser interesante y no trivial: *¿se puede
estimar el estado hipertensivo a partir de edad, IMC, colesterol, glucosa,
tabaquismo, alcohol y actividad física, sin medir la presión?* Ahí sí hay señal
que aprender, el techo es desconocido, y el baseline honesto es la clase mayoritaria.

**Camino B — Mantener el sintético, pero honesto.**
Introducir ruido estocástico real en la etiqueta (p. ej. probabilidad de
mal-clasificación dependiente de covariables) y excluir `PAS`/`PAD` de las features.
Menos valioso que A, porque sigue sin haber validez externa.

Recomendación: **camino A**. El dataset real ya está en el repositorio y el EDA ya
está hecho en `notebooks/EDA_cardiovascular_real.ipynb`.

---

*Última verificación: 2026-07-24, sobre `data/raw/dataset_hipertension_sintetico.csv`
(50.000 filas, seed=42).*
