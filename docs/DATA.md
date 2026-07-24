# Datos: procedencia, diccionario y licencias

El proyecto usa dos conjuntos de datos con roles distintos: uno **sintético**, sobre
el que se entrenan los modelos, y uno **real**, usado únicamente para análisis
exploratorio.

---

## 1. Dataset sintético (entrenamiento)

| Campo | Valor |
|-------|-------|
| Ruta | `data/raw/dataset_hipertension_sintetico.csv` |
| Generador | `src/generate_dataset.py` |
| Filas | 50.000 |
| Columnas | 16 (15 features + 1 target) |
| Semilla | `42` (reproducible) |
| Versionado | No — se regenera con `make data` |

### Diccionario de variables

| Columna | Tipo | Unidad / codificación | Rango | Generación |
|---------|------|----------------------|-------|------------|
| `Edad` | int | años | 18–89 | Uniforme |
| `Sexo` | int | 0 = Femenino, 1 = Masculino | {0,1} | Bernoulli(p=0,52) |
| `Peso` | float | kg | 40–160 | Normal(75, 15), recortada |
| `Talla` | float | metros | 1,40–2,00 | Normal(1,68, 0,12), recortada |
| `IMC` | float | kg/m² | derivado | Peso / Talla² |
| `PAS` | float | mmHg | 85–240 | Normal(130, 20), recortada |
| `PAD` | float | mmHg | 55–140 | Normal(82, 12), recortada |
| `PAM` | float | mmHg | derivado | (PAS + 2·PAD) / 3 |
| `Frec_Card` | float | latidos/min | 45–160 | Normal(78, 12), recortada |
| `Colesterol` | float | mg/dL | 100–350 | Normal(200, 40), recortada |
| `Glucosa` | float | mg/dL | 60–300 | Normal(105, 25), recortada |
| `Tabaquismo` | int | 0 = No, 1 = Sí | {0,1} | Bernoulli(p=0,30) |
| `Ejercicio` | int | horas/semana | 0–9 | Uniforme entera |
| `Estres` | int | escala subjetiva | 1–9 | Uniforme entera |
| `Herencia_HTA` | int | 0 = No, 1 = Sí | {0,1} | Bernoulli(p=0,40) |
| `Diagnostico` | int | **target**, 0–3 | {0,1,2,3} | Regla determinista (ver abajo) |

### Variable objetivo

| Clase | Etiqueta | Criterio base |
|-------|----------|---------------|
| 0 | Normal | PAS < 120 **y** PAD < 80 |
| 1 | Prehipertensión | PAS 120–129 **o** PAD 80–84 |
| 2 | Hipertensión Grado 1 | PAS 130–139 **o** PAD 85–89 |
| 3 | Hipertensión Grado 2 | PAS ≥ 140 **o** PAD ≥ 90 |

Sobre esa clasificación base se aplican dos ajustes deterministas: la clase sube un
nivel si el paciente acumula ≥ 3 factores de riesgo (IMC > 32, colesterol > 240,
glucosa > 140, estrés > 7, herencia positiva), y baja un nivel si registra ≥ 6 horas
semanales de ejercicio.

Distribución resultante: 26,2 % / 29,6 % / 26,8 % / 17,4 %.

> ⚠️ **Los umbrales son una simplificación docente de las guías AHA 2017 y no
> reproducen fielmente ninguna guía clínica vigente.** En particular, la categoría
> "Prehipertensión" fue reemplazada en la clasificación AHA 2017 por "Elevated" e
> "Hypertension Stage 1" con cortes distintos a los usados aquí. No usar esta tabla
> como referencia clínica.

### Limitaciones del dataset sintético

1. **El target es una función determinista de las features.** Defecto central del
   proyecto — ver [LEAKAGE_ANALYSIS.md](LEAKAGE_ANALYSIS.md).
2. **Independencia falsa entre variables.** Cada feature se muestrea de forma
   independiente. En población real, edad, IMC, colesterol, glucosa y presión
   arterial están fuertemente correlacionados. La estructura de covarianza del
   dataset no existe en la realidad.
3. **Distribuciones no calibradas.** Las medias y desviaciones son plausibles pero
   no derivan de ninguna cohorte publicada.
4. **Balance artificial de clases.** No refleja la prevalencia poblacional de HTA.
5. **Ruido de redondeo.** El CSV guarda valores redondeados
   (`pas.round(0)`, `imc.round(1)`) mientras la etiqueta se calculó sobre los
   valores sin redondear, lo que introduce ~9 % de inconsistencia en las fronteras
   de decisión.

---

## 2. Dataset real (solo EDA)

| Campo | Valor |
|-------|-------|
| Ruta | `data/real/cardio/cardio_train.csv` |
| Fuente | *Cardiovascular Disease Dataset*, Kaggle |
| Filas | 70.000 |
| Separador | `;` (punto y coma, no coma) |
| Uso actual | Exclusivamente `notebooks/EDA_cardiovascular_real.ipynb` |

### Diccionario de variables

| Columna | Descripción | Unidad / codificación |
|---------|-------------|----------------------|
| `id` | Identificador | entero |
| `age` | Edad | **días** (dividir por 365,25 para años) |
| `gender` | Sexo | 1 = Mujer, 2 = Hombre |
| `height` | Estatura | cm |
| `weight` | Peso | kg |
| `ap_hi` | Presión sistólica | mmHg |
| `ap_lo` | Presión diastólica | mmHg |
| `cholesterol` | Colesterol | 1 = normal, 2 = elevado, 3 = muy elevado |
| `gluc` | Glucosa | 1 = normal, 2 = elevado, 3 = muy elevado |
| `smoke` | Tabaquismo | 0 / 1 |
| `alco` | Consumo de alcohol | 0 / 1 |
| `active` | Actividad física | 0 / 1 |
| `cardio` | **Target original**: enfermedad cardiovascular | 0 / 1 |

### Advertencias de calidad

- **`age` viene en días.** Olvidarlo produce análisis sin sentido.
- **Valores de presión imposibles.** El dataset contiene `ap_hi` y `ap_lo` negativos
  o de magnitud absurda (miles). El notebook filtra con `between(80, 250)` y
  `between(40, 200)`. Cualquier uso posterior debe repetir ese filtrado.
- **Codificación de `gender` no documentada oficialmente.** La convención
  1 = Mujer / 2 = Hombre es la mayoritariamente aceptada, pero conviene verificarla
  con un cruce contra la distribución de estatura antes de confiar en ella.
- **`cholesterol` y `gluc` son ordinales, no continuas.** No son directamente
  comparables con las columnas homónimas del dataset sintético, que están en mg/dL.

### Por qué este dataset importa

Es el camino de salida del problema de leakage. Definiendo la etiqueta de
hipertensión desde `ap_hi`/`ap_lo` y **excluyendo esas dos columnas de las
features**, el problema pasa a ser genuinamente predictivo: estimar estado
hipertensivo sin medir la presión, a partir de edad, IMC, colesterol, glucosa y
hábitos. Ver [ROADMAP.md](ROADMAP.md).

---

## 3. Licencias

**Código del proyecto:** MIT (ver [LICENSE](../LICENSE)).

**Dataset sintético:** generado por este repositorio, cubierto por la misma licencia
MIT. No contiene datos de personas reales.

**Dataset de Kaggle:** se distribuye bajo los términos definidos por su autor
original en la plataforma. ⚠️ **Los términos de redistribución no han sido
verificados para este repositorio.** Antes de publicarlo, conviene confirmar la
licencia en la página de origen del dataset. Si es restrictiva, la práctica correcta
es **no versionar el archivo** y documentar en su lugar las instrucciones de
descarga:

```bash
# Requiere credenciales de Kaggle en ~/.kaggle/kaggle.json
kaggle datasets download -d sulianova/cardiovascular-disease-dataset -p data/real/
unzip data/real/cardiovascular-disease-dataset.zip -d data/real/cardio/
```

**Nota de higiene del repositorio:** `data/real/` contiene actualmente tanto el ZIP
(744 KB) como el CSV ya descomprimido (2,9 MB). El ZIP es redundante y puede
retirarse del control de versiones con `git rm --cached`.
