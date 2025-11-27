from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field, conint, confloat

from data_pipeline import FEATURE_COLUMNS
from model_utils import LABEL_MAP, load_model_bundle, predict_single


class PatientData(BaseModel):
    Edad: conint(ge=18, le=100)
    Sexo: conint(ge=0, le=1)
    Peso: confloat(gt=0)
    Talla: confloat(gt=0)
    IMC: confloat(gt=0)
    PAS: confloat(gt=0)
    PAD: confloat(gt=0)
    PAM: confloat(gt=0)
    Frec_Card: confloat(gt=0)
    Colesterol: confloat(gt=0)
    Glucosa: confloat(gt=0)
    Tabaquismo: conint(ge=0, le=1)
    Ejercicio: conint(ge=0, le=1)
    Estres: confloat(ge=0, le=10) = Field(alias="Estrés")
    Herencia_HTA: conint(ge=0, le=1)

    def to_model_dict(self) -> Dict:
        data = self.dict(by_alias=True)
        # Convertimos alias Estres -> Estrés para mantener los nombres de columnas
        if "Estres" in data and "Estrés" not in data:
            data["Estrés"] = data.pop("Estres")
        return data


app = FastAPI(title="HTA Classifier API", version="1.0.0")
model_bundle = load_model_bundle()


@app.get("/")
def root():
    return {
        "mensaje": "API de clasificación de hipertensión arterial",
        "features": FEATURE_COLUMNS,
        "clases": LABEL_MAP,
    }


@app.post("/predict")
def predict(patient: PatientData):
    payload = patient.to_model_dict()
    prediction = predict_single(model_bundle, payload)
    return prediction


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
