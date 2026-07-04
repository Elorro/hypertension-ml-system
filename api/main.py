from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

# ======================================================
# Carga de scaler y mejor modelo
# ======================================================
MODELS_DIR = "models"

# Cargar scaler
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
scaler = joblib.load(scaler_path)

# Leer nombre del mejor modelo
mejor_modelo_txt = os.path.join(MODELS_DIR, "mejor_modelo.txt")
with open(mejor_modelo_txt, "r") as f:
    mejor_modelo_nombre = f.read().strip()

# Mapear nombre lógico a archivo .pkl
nombre_a_archivo = {
    "LogisticRegression": "modelo_logreg.pkl",
    "SVM_RBF": "modelo_svm.pkl",
    "DecisionTree": "modelo_tree.pkl",
    "RandomForest": "modelo_rf.pkl",
    "XGBoost": "modelo_xgb.pkl",
}

modelo_path = os.path.join(MODELS_DIR, nombre_a_archivo[mejor_modelo_nombre])
model = joblib.load(modelo_path)

# Diccionario de clases
CLASES = {
    0: "Normal",
    1: "Prehipertensión",
    2: "Hipertensión Grado 1",
    3: "Hipertensión Grado 2",
}

# ======================================================
# Definición de esquema de entrada (Pydantic)
# ======================================================
class Paciente(BaseModel):
    Edad: int
    Sexo: int            # 0 = F, 1 = M
    Peso: float
    Talla: float
    IMC: float
    PAS: float
    PAD: float
    PAM: float
    Frec_Card: float
    Colesterol: float
    Glucosa: float
    Tabaquismo: int      # 0 = No, 1 = Sí
    Ejercicio: int       # horas/semana
    Estres: int          # 1-10
    Herencia_HTA: int    # 0 = No, 1 = Sí


# ======================================================
# Inicializar FastAPI
# ======================================================
app = FastAPI(
    title="API Clasificación Hipertensión",
    description=(
        "API para clasificar hipertensión arterial usando un modelo de Machine Learning "
        "(mejor modelo: {})\n\n"
        "⚠ AVISO: Esta herramienta es solo para fines educativos y NO reemplaza una "
        "valoración médica profesional."
    ).format(mejor_modelo_nombre),
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {
        "mensaje": "API de clasificación de hipertensión funcionando.",
        "modelo": mejor_modelo_nombre,
        "aviso": "No es diagnóstico médico. Consulta siempre a tu profesional de salud.",
    }


@app.post("/predecir")
def predecir(paciente: Paciente):
    """
    Recibe los datos de un paciente y devuelve el diagnóstico de hipertensión.
    """

    # Orden de las features debe coincidir EXACTAMENTE con el usado en el entrenamiento
    # ["Edad", "Sexo", "Peso", "Talla", "IMC", "PAS", "PAD", "PAM",
    #  "Frec_Card", "Colesterol", "Glucosa", "Tabaquismo", "Ejercicio",
    #  "Estres", "Herencia_HTA"]

    x = np.array([[
        paciente.Edad,
        paciente.Sexo,
        paciente.Peso,
        paciente.Talla,
        paciente.IMC,
        paciente.PAS,
        paciente.PAD,
        paciente.PAM,
        paciente.Frec_Card,
        paciente.Colesterol,
        paciente.Glucosa,
        paciente.Tabaquismo,
        paciente.Ejercicio,
        paciente.Estres,
        paciente.Herencia_HTA,
    ]])

    # Escalar
    x_scaled = scaler.transform(x)

    # Predicción de clase
    pred = int(model.predict(x_scaled)[0])

    # Probabilidades (si el modelo las soporta)
    proba_dict = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_scaled)[0]
        for i, p in enumerate(proba):
            proba_dict[int(i)] = float(p)
    else:
        proba_dict = None

    return {
        "diagnostico_numerico": pred,
        "diagnostico_texto": CLASES.get(pred, "Desconocido"),
        "probabilidades": proba_dict,
        "aviso": "Resultado orientativo. No sustituye valoración médica profesional.",
    }
