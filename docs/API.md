# Referencia del servicio de inferencia

Servicio REST construido con FastAPI que expone el modelo seleccionado durante el
entrenamiento.

**Base URL (desarrollo):** `http://127.0.0.1:8000`
**Documentación interactiva:** `/docs` (Swagger UI) · `/redoc` (ReDoc)

---

## Arranque

```bash
# Prerrequisito: los artefactos deben existir en models/
python src/generate_dataset.py
python src/train_classical_models.py

uvicorn api.main:app --reload
```

El módulo carga `scaler.pkl`, `mejor_modelo.txt` y el `.pkl` del modelo ganador
**en tiempo de import**. Si falta alguno, el proceso termina con `FileNotFoundError`
antes de aceptar conexiones.

---

## `GET /`

Verificación de estado y modelo activo.

**Respuesta `200 OK`**

```json
{
  "mensaje": "API de clasificación de hipertensión funcionando.",
  "modelo": "XGBoost",
  "aviso": "No es diagnóstico médico. Consulta siempre a tu profesional de salud."
}
```

---

## `POST /predecir`

Clasifica un paciente individual.

### Cuerpo de la petición

Todos los campos son **obligatorios**. El orden en el JSON es irrelevante; el
servicio lo normaliza internamente.

| Campo | Tipo | Unidad / codificación | Rango plausible |
|-------|------|----------------------|-----------------|
| `Edad` | int | años | 18–100 |
| `Sexo` | int | 0 = Femenino, 1 = Masculino | 0 / 1 |
| `Peso` | float | kg | 30–200 |
| `Talla` | float | metros | 1,30–2,10 |
| `IMC` | float | kg/m² | Peso / Talla² |
| `PAS` | float | mmHg | 80–250 |
| `PAD` | float | mmHg | 50–150 |
| `PAM` | float | mmHg | (PAS + 2·PAD) / 3 |
| `Frec_Card` | float | latidos/min | 40–160 |
| `Colesterol` | float | mg/dL | 100–400 |
| `Glucosa` | float | mg/dL | 60–300 |
| `Tabaquismo` | int | 0 = No, 1 = Sí | 0 / 1 |
| `Ejercicio` | int | horas/semana | 0–14 |
| `Estres` | int | escala subjetiva | 1–10 |
| `Herencia_HTA` | int | 0 = No, 1 = Sí | 0 / 1 |

> ⚠️ La columna de la petición se llama `Estres`, **sin tilde**. El pipeline
> heredado de la raíz usa `Estrés` con tilde; no son intercambiables.

> ⚠️ **La validación es de tipo, no de rango.** Un `PAS` de 900 se acepta y devuelve
> una predicción sin sentido. Validar rangos está en el [ROADMAP](ROADMAP.md).

### Ejemplo

```bash
curl -X POST http://127.0.0.1:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "Edad": 52, "Sexo": 1, "Peso": 88.0, "Talla": 1.74, "IMC": 29.1,
    "PAS": 145.0, "PAD": 92.0, "PAM": 109.7, "Frec_Card": 78.0,
    "Colesterol": 215.0, "Glucosa": 112.0, "Tabaquismo": 1,
    "Ejercicio": 2, "Estres": 7, "Herencia_HTA": 1
  }'
```

### Respuesta `200 OK`

```json
{
  "diagnostico_numerico": 3,
  "diagnostico_texto": "Hipertensión Grado 2",
  "probabilidades": {
    "0": 0.0012,
    "1": 0.0087,
    "2": 0.0341,
    "3": 0.9560
  },
  "aviso": "Resultado orientativo. No sustituye valoración médica profesional."
}
```

| Campo | Descripción |
|-------|-------------|
| `diagnostico_numerico` | Clase predicha, 0–3 |
| `diagnostico_texto` | Etiqueta legible correspondiente |
| `probabilidades` | Distribución sobre las 4 clases, o `null` si el modelo no expone `predict_proba` |
| `aviso` | Descargo de responsabilidad, siempre presente |

### Errores

| Código | Causa |
|--------|-------|
| `422 Unprocessable Entity` | Campo faltante o de tipo incorrecto. FastAPI detalla el campo en `detail`. |
| `500 Internal Server Error` | Fallo en el escalado o la predicción. |

Ejemplo de `422` al omitir `PAS`:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "PAS"],
      "msg": "Field required"
    }
  ]
}
```

---

## Cliente Python

```python
import requests

paciente = {
    "Edad": 52,
    "Sexo": 1,
    "Peso": 88.0,
    "Talla": 1.74,
    "IMC": 29.1,
    "PAS": 145.0,
    "PAD": 92.0,
    "PAM": 109.7,
    "Frec_Card": 78.0,
    "Colesterol": 215.0,
    "Glucosa": 112.0,
    "Tabaquismo": 1,
    "Ejercicio": 2,
    "Estres": 7,
    "Herencia_HTA": 1,
}

resp = requests.post("http://127.0.0.1:8000/predecir", json=paciente, timeout=5)
resp.raise_for_status()
print(resp.json()["diagnostico_texto"])
```

---

## Limitaciones conocidas

| Limitación | Impacto | Estado |
|------------|---------|--------|
| Sin endpoint batch | El dashboard emite una petición por fila del CSV; lento en archivos grandes | [ROADMAP](ROADMAP.md) |
| Sin validación de rangos | Entradas fisiológicamente imposibles producen respuestas sin aviso | [ROADMAP](ROADMAP.md) |
| Sin autenticación ni rate limiting | Solo apto para uso local | [ROADMAP](ROADMAP.md) |
| Sin logging estructurado | No hay trazabilidad de peticiones | [ROADMAP](ROADMAP.md) |
| Orden de features implícito | Acoplado por convención al orden de columnas del CSV de entrenamiento | [ROADMAP](ROADMAP.md) |

**Sobre la validez de las predicciones:** el modelo subyacente sufre target leakage.
Las respuestas de este servicio reproducen una regla de umbral, no una predicción
clínica. Ver [LEAKAGE_ANALYSIS.md](LEAKAGE_ANALYSIS.md) y [MODEL_CARD.md](MODEL_CARD.md).
