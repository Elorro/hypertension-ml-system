# Referencia del servicio de inferencia

Servicio REST (FastAPI) que expone **probabilidades** de dos modelos seleccionados en DT-4,
entrenados sobre el dataset real *Cardiovascular Disease* (Kaggle, 68.678 filas tras
limpieza). No es una herramienta clínica y ninguna respuesta es un diagnóstico.

| Modelo | Endpoint | Rol | Target |
|--------|----------|-----|--------|
| A′ `riesgo_cv_con_pa` | `POST /v1/riesgo-cardiovascular` | principal | `cardio` (enfermedad cardiovascular, etiqueta del dataset) |
| B1 `hta_b1` | `POST /v1/hipertension-sin-pa` | experimental | `hta = (ap_hi >= 140) \| (ap_lo >= 90)`, **sin** presión en la entrada |

**Base URL (desarrollo):** `http://127.0.0.1:8000`
**Documentación interactiva:** `/docs` (Swagger UI) · `/redoc` · `/openapi.json`
**Versión:** `2.0.0`, la del proyecto (`pyproject.toml`), publicada en `/health` y en
OpenAPI. El prefijo `/v1` de las rutas es el namespace del contrato nuevo, no la versión
del proyecto.

Métricas, IC y calibración de cada modelo: `GET /v1/modelos` y
[DT4_RESULTS.md](DT4_RESULTS.md). Cada modelo se eligió por log-loss en validación
cruzada sobre el train (protocolo preregistrado, [DT4_PROTOCOL.md](DT4_PROTOCOL.md)) y
el test se evaluó una sola vez; ese test ya se había observado en DT-1 (limitación
declarada).

---

## Arranque

```bash
pip install -r requirements-serve.txt   # mínimo: sin pandas ni xgboost
make serve-api                           # uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Requiere `models/dt4_*.pkl` (no versionados; se generan con `make train-dt4`, ≈ 7 min).

| Variable | Por defecto | Uso |
|----------|-------------|-----|
| `MODEL_DIR` | `models/` del repo | Directorio de los `.pkl` |
| `MANIFEST_PATH` | `models/dt4_manifest.json` | Manifiesto de DT-4 (versionado). El de DT-1, `models/dt1_manifest.json`, queda como registro y sigue siendo servible |

Los modelos se cargan **una vez, al arrancar** (lifespan), tras verificar contra el
manifiesto (`src/artefactos.py`, perfil `servicio`):

1. Python por major.minor; scikit-learn, numpy y joblib exactos (determinan la
   compatibilidad del pickle). xgboost solo si algún ganador servido es XGBoost.
2. sha256 de cada modelo y scaler.
3. Carga sin `UserWarning` de versión; `mean_`/`scale_` del scaler iguales a los del
   manifiesto; número de features coherente.

Si algo falla, **el servicio no arranca**:

```
src.artefactos.ArtefactoInvalido: [hta_b1] sha256 distinto en …/dt4_hta_b1__scaler.pkl: 10f0df0d684139d1… != manifiesto 3454db2fbdb132ef…
ERROR:    Application startup failed. Exiting.
```

Medido en local (Python 3.14.6, `requirements-serve.txt`): venv de 293 MB; proceso con
A′ + B1 de DT-4 cargados, RSS ≈ 202 MiB (≈ 207 MiB tras lotes de 1.000 filas), frente a
352 MiB con los modelos de DT-1. `.pkl` servidos: 32 MB (130 MB con los de DT-1).

---

## Entrada

Unidades humanas. Los nombres son los del dataset salvo la edad, que llega en años.
Todos los campos son obligatorios; **cualquier campo no declarado devuelve 422**
(`extra="forbid"`). Los rangos son el dominio de entrenamiento, inclusivos.

| Campo | Tipo | Unidad / codificación | Dominio |
|-------|------|-----------------------|---------|
| `age_years` | float | años (admite decimales) | [29, 65] — entrenamiento: 29,56–64,92, redondeado hacia afuera |
| `gender` | 1 \| 2 | código crudo del dataset; «2 = hombre» es **inferido** (manifiesto, `dataset.codificacion_gender`), no documentado por la fuente | 1, 2 |
| `height` | float | cm | [120, 220] |
| `weight` | float | kg | [30, 200] |
| `cholesterol` | 1 \| 2 \| 3 | 1 normal, 2 elevado, 3 muy elevado | 1, 2, 3 |
| `gluc` | 1 \| 2 \| 3 | 1 normal, 2 elevada, 3 muy elevada | 1, 2, 3 |
| `smoke` | 0 \| 1 | fuma | 0, 1 |
| `alco` | 0 \| 1 | consume alcohol | 0, 1 |
| `active` | 0 \| 1 | físicamente activo | 0, 1 |
| `ap_hi` | float | PAS, mmHg — **solo A′** | [70, 250] |
| `ap_lo` | float | PAD, mmHg — **solo A′** | [40, 200] y `ap_lo < ap_hi` |

Validaciones adicionales:

- **IMC calculado en el servidor**, `weight / (height/100)²`, debe quedar en
  [12, 70] kg/m². El cliente nunca lo envía: mandar `bmi` es un campo extra → 422.
- **B1 rechaza `ap_hi`/`ap_lo`** con 422: el contrato anti-leakage aplicado en la API.
- Fuera del rango de edad se rechaza en vez de avisar: un árbol no extrapola, aplana
  la predicción al valor del extremo.

Todas las cotas viven en `src/cardio_features.py`, compartido con el entrenamiento.

---

## `POST /v1/riesgo-cardiovascular` — A′

```bash
curl -X POST http://127.0.0.1:8000/v1/riesgo-cardiovascular \
  -H "Content-Type: application/json" \
  -d '{"age_years":52,"gender":1,"height":165,"weight":72,"cholesterol":1,"gluc":1,
       "smoke":0,"alco":0,"active":1,"ap_hi":130,"ap_lo":85}'
```

```json
{
  "probabilidad": 0.5133575026752739,
  "prevalencia_base": 0.49479451057478796,
  "modelo": {
    "experimento": "riesgo_cv_con_pa",
    "algoritmo": "RandomForest",
    "sha256": "fa42bc7e86936ba753adec2e5b9f098654a8c87cf56de4734e4b9698a88d0428"
  },
  "aviso": "Demostración de ingeniería de ML, no herramienta clínica. Esta probabilidad no es un diagnóstico ni sustituye la medición de la presión arterial ni la valoración de un profesional de salud.",
  "clase": 1,
  "umbral": 0.5
}
```

| Campo | Descripción |
|-------|-------------|
| `probabilidad` | Probabilidad estimada de la clase positiva, en [0, 1]. Sin calibración post-hoc; ECE medido en test: 0,0141. |
| `prevalencia_base` | Prevalencia de la clase positiva en el train del modelo: la referencia sin información. |
| `modelo` | Experimento, algoritmo y sha256 del `.pkl`, verificado al arrancar. |
| `clase` | `1` si `probabilidad >= umbral`. Solo en A′. |
| `umbral` | `0.5`, fijo y explícito. |
| `aviso` | Siempre presente. |

## `POST /v1/hipertension-sin-pa` — B1 (experimental)

Estima hipertensión **sin medir la presión arterial**. AUC 0,696 [0,687 · 0,705].
La respuesta **no incluye `clase`**: con umbral 0,5 el modelo servido deja 3.257 falsos
negativos frente a 1.458 verdaderos positivos en test. La información está en el
ranking; la probabilidad ordena riesgo, no diagnostica. ECE medido en test: 0,0079.

```bash
curl -X POST http://127.0.0.1:8000/v1/hipertension-sin-pa \
  -H "Content-Type: application/json" \
  -d '{"age_years":52,"gender":1,"height":165,"weight":72,"cholesterol":1,"gluc":1,
       "smoke":0,"alco":0,"active":1}'
```

```json
{
  "probabilidad": 0.24104744560537572,
  "prevalencia_base": 0.3432710858723745,
  "modelo": {
    "experimento": "hta_b1",
    "algoritmo": "RandomForest",
    "sha256": "d12ee51fc5df48f558de867e5c40f5d51c7f0d05a9c7273e95f5633cc6a52424"
  },
  "aviso": "Demostración de ingeniería de ML, no herramienta clínica. Esta probabilidad no es un diagnóstico ni sustituye la medición de la presión arterial ni la valoración de un profesional de salud."
}
```

---

## Lote: `POST /v1/riesgo-cardiovascular/lote` · `POST /v1/hipertension-sin-pa/lote`

Cuerpo: lista JSON de filas con el esquema del endpoint individual, entre 1 y
**1.000** filas. Cada fila se valida por separado: una fila inválida devuelve su error
en su `indice` (desde 0) y no tumba el lote. Las filas válidas se predicen en una sola
llamada al modelo.

```bash
curl -X POST http://127.0.0.1:8000/v1/hipertension-sin-pa/lote \
  -H "Content-Type: application/json" \
  -d '[{"age_years":62,"gender":2,"height":168,"weight":95,"cholesterol":3,"gluc":3,"smoke":1,"alco":1,"active":0},
       {"age_years":52,"gender":1,"height":300,"weight":72,"cholesterol":1,"gluc":1,"smoke":0,"alco":0,"active":1}]'
```

```json
{
  "n_filas": 2,
  "n_ok": 1,
  "n_error": 1,
  "filas": [
    {
      "indice": 0,
      "ok": true,
      "resultado": {
        "probabilidad": 0.6493759992470167,
        "prevalencia_base": 0.3432710858723745,
        "modelo": {
          "experimento": "hta_b1",
          "algoritmo": "RandomForest",
          "sha256": "d12ee51fc5df48f558de867e5c40f5d51c7f0d05a9c7273e95f5633cc6a52424"
        },
        "aviso": "Demostración de ingeniería de ML, no herramienta clínica. Esta probabilidad no es un diagnóstico ni sustituye la medición de la presión arterial ni la valoración de un profesional de salud."
      },
      "errores": null
    },
    {
      "indice": 1,
      "ok": false,
      "resultado": null,
      "errores": [
        {"loc": ["height"], "msg": "Input should be less than or equal to 220", "type": "less_than_equal"}
      ]
    }
  ],
  "aviso": "Demostración de ingeniería de ML, no herramienta clínica. Esta probabilidad no es un diagnóstico ni sustituye la medición de la presión arterial ni la valoración de un profesional de salud."
}
```

Más de 1.000 filas es un error del lote completo (sin eco de la entrada):

```json
{"detail": [{"loc": ["body"], "msg": "List should have at most 1000 items after validation, not 1001", "type": "too_long"}]}
```

---

## `GET /health`

```json
{
  "estado": "ok",
  "version_api": "2.0.0",
  "modelos": [
    {"experimento": "riesgo_cv_con_pa", "algoritmo": "RandomForest", "sha256": "fa42bc7e86936ba753adec2e5b9f098654a8c87cf56de4734e4b9698a88d0428"},
    {"experimento": "hta_b1", "algoritmo": "RandomForest", "sha256": "d12ee51fc5df48f558de867e5c40f5d51c7f0d05a9c7273e95f5633cc6a52424"}
  ],
  "entorno": {"python": "3.14.6", "scikit_learn": "1.9.1", "numpy": "2.5.3", "joblib": "1.6.0"}
}
```

`503` si el servicio se construyó sin modelos cargados.

## `GET /v1/modelos`

Metadatos desde el manifiesto, por modelo: `endpoint`, `definicion_target`,
`entrada` (JSON Schema de cada campo: `title` legible, `minimum`/`maximum`,
`description`, `examples` y, en los campos codificados, `x-etiquetas` con el significado
de cada código; con eso, y nada más, el dashboard construye sus formularios),
`derivadas_en_servidor`,
`orden_features_modelo`, `prevalencia_base`, `metricas_test` (AUC con IC 95 %
bootstrap, Brier y su baseline, ECE de 10 bins, matriz de confusión con umbral 0,5),
`devuelve_clase`, `umbral_clase`, `seleccion` (criterio, candidato e hiperparámetros,
mejor en CV, cuántos quedaron dentro de 1 EE, y que el test no intervino) y, en B1,
`advertencia`. Incluye `max_filas_lote` y `nota_metricas` (cómo se seleccionó y la
limitación del test ya observado en DT-1). Extracto de B1:

```json
{
  "experimento": "hta_b1",
  "rol": "experimental",
  "definicion_target": "hta = (ap_hi >= 140) | (ap_lo >= 90)",
  "seleccion": {
    "criterio": "log-loss media en validación cruzada de 5 folds sobre train; regla de 1 EE",
    "candidato": "RandomForest(max_depth=8,min_samples_leaf=100)",
    "hiperparametros": {"max_depth": 8, "min_samples_leaf": 100},
    "mejor_en_cv": "XGBoost(max_depth=3,min_child_weight=1)",
    "n_elegibles_dentro_de_1_ee": 7,
    "test_intervino_en_la_seleccion": false
  },
  "prevalencia_base": 0.3432710858723745,
  "metricas_test": {
    "n_test": 13736,
    "roc_auc": 0.695460280436728,
    "roc_auc_ic95_bootstrap": [0.6866486677440173, 0.7049276641456452],
    "brier": 0.20154744492370275,
    "brier_baseline_prevalencia": 0.22543213072444335,
    "ece_uniform_10": 0.00792668979225035,
    "confusion_matrix_umbral_0_5": {"tn": 8023, "fp": 998, "fn": 3257, "tp": 1458}
  },
  "devuelve_clase": false,
  "umbral_clase": null,
  "advertencia": "Experimental. Estima hipertensión SIN medir la presión arterial. La información está en el ranking de riesgo, no en la decisión binaria: con umbral 0,5 el modelo servido deja 3257 falsos negativos frente a 1458 verdaderos positivos en test. Por eso la respuesta no incluye clase. La probabilidad ordena riesgo; no diagnostica."
}
```

---

## Errores

| Código | Causa |
|--------|-------|
| `422` | Campo faltante, de tipo o código inválido, fuera de dominio, campo extra, o lote fuera de [1, 1.000] filas. |
| `503` | Servicio sin modelos cargados. |

Formato de `422`: `{"detail": [{"loc", "msg", "type"}]}`, el mismo de los errores por
fila del lote. Ejemplos reales:

```json
{"detail": [{"loc": ["body", "ap_hi"], "msg": "Extra inputs are not permitted", "type": "extra_forbidden"}]}
{"detail": [{"loc": ["body"], "msg": "Value error, IMC calculado 84.44 kg/m² (weight / (height/100)²) fuera de la cota de entrenamiento [12, 70] kg/m²", "type": "value_error"}]}
{"detail": [{"loc": ["body", "age_years"], "msg": "Input should be less than or equal to 65", "type": "less_than_equal"}]}
{"detail": [{"loc": ["body"], "msg": "Value error, ap_lo (90) debe ser menor que ap_hi (80); el entrenamiento descartó las filas con presión invertida", "type": "value_error"}]}
```

El contrato sintético anterior (`POST /predecir`, 4 clases, 15 features) se eliminó:
devuelve `404`.

---

## Cliente Python

```python
import requests

paciente = {
    "age_years": 52,
    "gender": 1,
    "height": 165,
    "weight": 72,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}

resp = requests.post("http://127.0.0.1:8000/v1/hipertension-sin-pa", json=paciente, timeout=10)
resp.raise_for_status()
r = resp.json()
print(f"P = {r['probabilidad']:.3f}  (prevalencia base {r['prevalencia_base']:.3f})")
```

---

## Limitaciones conocidas

| Limitación | Impacto | Estado |
|------------|---------|--------|
| Test ya observado en DT-1 | La selección de DT-4 no lo mira, pero no es un test virgen | limitación declarada, [DT4_PROTOCOL.md](DT4_PROTOCOL.md) |
| `.pkl` fuera de git (≈ 32 MB servidos) | Un despliegue necesita otra vía para los artefactos | pendiente |
| Sin autenticación ni rate limiting | Solo apto para uso local | [ROADMAP](ROADMAP.md) |
| Sin logging estructurado | Solo el log de uvicorn | [ROADMAP](ROADMAP.md) |
| Validez externa | Un solo dataset, sin análisis de equidad | DT-15, [ROADMAP](ROADMAP.md) |

No existe un despliegue público: el servicio corre en local.
