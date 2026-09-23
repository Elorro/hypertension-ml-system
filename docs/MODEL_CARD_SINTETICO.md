# Model Card (histórica) — Clasificador sintético de Hipertensión Arterial

Formato basado en *Model Cards for Model Reporting* (Mitchell et al., 2019).

> **Documento histórico.** Describe el modelo **sintético** (`models/modelo_*.pkl`),
> cuyo target es una función determinista de las features, tal como lo servía
> `api/main.py` hasta el commit `6e59035` (2026-09-23). Desde entonces ningún servicio
> lo carga: el pipeline sintético se conserva solo como evidencia reproducible del
> target leakage. El texto de abajo describe ese estado y no se actualiza; las
> menciones al «servicio» y al «dashboard» se refieren al contrato retirado
> (`POST /predecir`). Los modelos que se sirven hoy (A′ y B1, datos reales) están en
> [MODEL_CARD.md](MODEL_CARD.md).

---

## Detalles del modelo

| Campo | Valor |
|-------|-------|
| **Desarrollador** | Luis Araque |
| **Versión** | 1.0.0 |
| **Fecha** | Enero 2026 |
| **Tipo** | Clasificador multiclase (4 clases), aprendizaje supervisado |
| **Algoritmos evaluados** | Regresión Logística, SVM-RBF, Árbol de Decisión, Random Forest, XGBoost |
| **Criterio de selección** | Macro-F1 sobre conjunto de test (20 %) |
| **Preprocesamiento** | `StandardScaler` sobre las 15 features numéricas |
| **Licencia** | MIT |

**Entrada:** vector de 15 features clínicas y de estilo de vida (ver
[DATA.md](DATA.md)).
**Salida:** clase predicha (0–3) y, cuando el algoritmo lo soporta, distribución de
probabilidad sobre las cuatro clases.

## Uso previsto

**Uso primario:** demostración educativa de un pipeline de ML end-to-end —
generación de datos, comparación de algoritmos, despliegue como servicio y consumo
desde una interfaz.

**Usuarios previstos:** estudiantes y desarrolladores que evalúan la arquitectura
del sistema o el código.

**Fuera de alcance — usos explícitamente desaconsejados:**

- Diagnóstico, tamizaje o triaje de pacientes reales.
- Cualquier decisión clínica, de tratamiento o de seguimiento.
- Estimación de riesgo cardiovascular individual o poblacional.
- Uso en contextos de seguros, empleo o cualquier decisión que afecte a personas.

## Factores

**Grupos evaluados:** ninguno. No se ha realizado análisis de desempeño
desagregado por sexo, grupo etario ni ninguna otra subpoblación.

**Instrumentación:** los datos son sintéticos; no provienen de ningún dispositivo
de medición real. En un despliegue real, la variabilidad de esfigmomanómetros, la
hipertensión de bata blanca y la hora de la medición serían factores de primer orden
que este modelo ignora por completo.

## Métricas y por qué desconfiar de ellas

**Métricas calculadas:** accuracy, macro-F1, ROC-AUC one-vs-rest.

**Advertencia central:** las métricas de este modelo **no miden capacidad
predictiva**. El target del dataset de entrenamiento es una función determinista de
las features de entrada. Un modelo con accuracy de 0,95 no detecta hipertensión con
95 % de acierto: memorizó el umbral que generó la etiqueta.

Evidencia cuantitativa, reproducible con `python scripts/verify_leakage.py`:

| Referencia | Accuracy |
|------------|----------|
| Baseline clase mayoritaria | 29,6 % |
| **Regla `if` sobre PAS/PAD, sin entrenamiento** | **61,9 %** |
| **Regla completa del generador, sin entrenamiento** | **91,1 %** |
| Regresión Logística (entrenada) | 47,7 % |
| SVM RBF (entrenada) | 68,8 % |
| Árbol de Decisión, `max_depth=8` (entrenado) | 94,8 % |
| Random Forest, 300 árboles (entrenado) | 94,8 % |
| XGBoost, 400 árboles (entrenado) | 95,3 % |

Una regla `if` determinista, sin entrenar (`scripts/verify_leakage.py::regla_completa`),
recupera el 91,1 % de las etiquetas del dataset sintético. Cualquier modelo entrenado
compite contra esa cifra, no contra el 29,6 % del azar. Leída así, la tabla dice que
400 árboles boosteados mejoran unos pocos puntos sobre esa regla — y que **un solo
árbol de profundidad 8 ya alcanza 94,8 %**, a 0,5 puntos del ganador.

Contraste con el dataset real (DT-1, sin la presión en las features): la mejor regla
determinista que encuentra `scripts/audit_cardio_leakage.py` supera al baseline de
clase mayoritaria en solo +2,28 pp, mientras que con `ap_hi`/`ap_lo` (control
positivo) reconstruye el target al 99,27 %.

El patrón más diagnóstico está en la brecha entre familias: los modelos basados en
árboles llegan a ~95 % y los lineales o de kernel se quedan en 48–69 %. El target es
una partición por umbrales; los árboles la representan de forma nativa y los modelos
lineales no pueden. La geometría del problema es la de un `if`, no la de un fenómeno
clínico.

Análisis completo en [LEAKAGE_ANALYSIS.md](LEAKAGE_ANALYSIS.md).

**Sesgo adicional en el reporte:** el conjunto de test se usa tanto para
seleccionar el mejor modelo como para reportar su desempeño. El F1 publicado del
ganador está sesgado al alza por selección. Faltaría un split train/validación/test.

## Datos de entrenamiento

50.000 registros sintéticos generados por `src/generate_dataset.py` con `seed=42`.
Las features se muestrean de distribuciones **independientes** — normales o
binomiales — recortadas a rangos plausibles.

**Limitación estructural:** la independencia entre variables es falsa. En población
real, la edad, el IMC, el colesterol y la presión arterial están fuertemente
correlacionados. Un modelo entrenado sobre features independientes aprende una
geometría del espacio de entrada que no existe fuera del generador.

Distribución de clases: 26,2 % / 29,6 % / 26,8 % / 17,4 % (clases 0–3). Balance
artificial, muy distinto de la prevalencia poblacional real de HTA.

## Datos de evaluación

Split aleatorio del 20 % del **mismo dataset sintético**. No hay conjunto de
validación externo ni evaluación sobre datos reales.

El repositorio incluye un dataset clínico real (`data/real/cardio/cardio_train.csv`,
70.000 pacientes) usado **solo para EDA**, nunca para entrenar ni validar. Cerrar
esa brecha es el hito principal del [ROADMAP](ROADMAP.md).

## Consideraciones éticas

**Riesgo clínico.** Un sistema que emite la cadena "Hipertensión Grado 2" puede
inducir confianza injustificada, tanto en falsos positivos (ansiedad, consultas
innecesarias) como en falsos negativos (retraso en atención real). Por eso cada
respuesta del servicio y cada vista del dashboard incluyen un aviso explícito de no
sustitución de criterio médico.

**Ausencia de análisis de equidad.** No se ha medido el desempeño por sexo ni por
grupo etario. Sobre datos sintéticos con generación simétrica el análisis sería
vacío; sobre datos reales sería obligatorio antes de cualquier uso.

**Privacidad.** El entrenamiento no usa datos de personas reales. El servicio no
persiste las peticiones recibidas: procesa en memoria y responde. Un despliegue
real con datos de pacientes quedaría sujeto a normativa de datos sensibles de salud
(en Colombia, Ley 1581 de 2012 y sus decretos reglamentarios), lo que exigiría
consentimiento informado, cifrado en tránsito y reposo, y registro de auditoría.

## Advertencias y recomendaciones

1. **No usar en contexto clínico.** Sin excepción.
2. **No citar las métricas fuera de contexto.** Cualquier número de este proyecto
   debe acompañarse del análisis de leakage.
3. **Antes de cualquier uso serio:** reentrenar sobre datos reales excluyendo
   `PAS`/`PAD` de las features, con validación cruzada estratificada, split
   train/val/test separado, y análisis de desempeño desagregado por subgrupo.
4. **Validar rangos de entrada.** El esquema Pydantic valida tipos, no plausibilidad
   fisiológica. Valores absurdos producen predicciones sin sentido, sin aviso.

---

*Para dudas o correcciones sobre esta model card: abrir un issue en el repositorio.*
