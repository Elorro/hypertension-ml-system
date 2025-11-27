import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from data_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, generate_synthetic_hypertension_data, save_dataset
from model_utils import LABEL_MAP, predict_single

DATA_PATH = Path("data/hypertension_synthetic.csv")
MODEL_PATH = Path("models/best_model.joblib")
REPORTS_PATH = Path("reports/metrics.json")


@st.cache_data
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        df = generate_synthetic_hypertension_data()
        save_dataset(df, path)
    else:
        df = pd.read_csv(path)
    return df


@st.cache_resource
def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        st.error("Modelo no encontrado. Ejecuta train_models.py primero.")
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_metrics(path: Path = REPORTS_PATH):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def plot_feature_importance(model_bundle) -> pd.DataFrame:
    model = model_bundle["model"]
    estimator = getattr(model, "named_steps", {}).get("model", model)

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.mean(np.abs(estimator.coef_), axis=0)
    else:
        return pd.DataFrame()

    return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances}).sort_values(
        "importance", ascending=False
    )


def main():
    st.set_page_config(page_title="Clasificador HTA", layout="wide")
    st.title("Clasificación de Hipertensión Arterial")
    st.caption("Modelos ML (LogReg, SVM, RF, XGBoost, ANN) entrenados sobre datos sintéticos.")

    data = load_data()
    model_bundle = load_model()
    metrics = load_metrics()

    tab_overview, tab_importance, tab_predict = st.tabs(["Panorama", "Importancia", "Predicción"])

    with tab_overview:
        st.subheader("Estadísticas del dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filas", len(data))
            st.metric("Clases", data[TARGET_COLUMN].nunique())
            class_counts = data[TARGET_COLUMN].value_counts().rename(index=LABEL_MAP)
            st.bar_chart(class_counts)
        with col2:
            st.write(data[["Edad", "IMC", "PAS", "PAD", "Colesterol", "Glucosa"]].describe())

        if metrics:
            st.subheader("Métricas por modelo")
            st.json(metrics)

    with tab_importance:
        st.subheader("Importancia de variables (modelo en producción)")
        importance_df = plot_feature_importance(model_bundle)
        if importance_df.empty:
            st.info("El modelo actual no expone importancias directamente.")
        else:
            st.dataframe(importance_df)
            st.bar_chart(importance_df.set_index("feature"))

    with tab_predict:
        st.subheader("Predicción individual")
        user_input = {}
        col_left, col_right = st.columns(2)
        with col_left:
            user_input["Edad"] = st.slider("Edad", 18, 90, 45)
            user_input["Sexo"] = st.selectbox("Sexo (0=F,1=M)", [0, 1], index=1)
            user_input["Peso"] = st.number_input("Peso (kg)", 40.0, 160.0, 80.0, step=0.5)
            user_input["Talla"] = st.number_input("Talla (m)", 1.4, 2.1, 1.72, step=0.01)
            user_input["IMC"] = st.number_input("IMC", 15.0, 45.0, 27.0, step=0.1)
            user_input["PAS"] = st.number_input("PAS (mmHg)", 90.0, 230.0, 135.0, step=1.0)
            user_input["PAD"] = st.number_input("PAD (mmHg)", 50.0, 140.0, 85.0, step=1.0)
        with col_right:
            user_input["PAM"] = st.number_input("PAM (mmHg)", 60.0, 180.0, 100.0, step=0.5)
            user_input["Frec_Card"] = st.number_input("Frec. cardíaca (lpm)", 40.0, 160.0, 75.0, step=1.0)
            user_input["Colesterol"] = st.number_input("Colesterol (mg/dL)", 100.0, 350.0, 200.0, step=1.0)
            user_input["Glucosa"] = st.number_input("Glucosa (mg/dL)", 60.0, 300.0, 105.0, step=1.0)
            user_input["Tabaquismo"] = st.selectbox("Tabaquismo (0/1)", [0, 1], index=0)
            user_input["Ejercicio"] = st.selectbox("Ejercicio (0/1)", [0, 1], index=1)
            user_input["Estrés"] = st.slider("Estrés (0-10)", 0.0, 10.0, 5.0, step=0.5)
            user_input["Herencia_HTA"] = st.selectbox("Herencia HTA (0/1)", [0, 1], index=0)

        if st.button("Predecir"):
            result = predict_single(model_bundle, user_input)
            st.success(f"Diagnóstico: {result['clase']} (clase={result['prediccion']})")
            st.write("Probabilidades:")
            st.json(result["probabilidades"])


if __name__ == "__main__":
    main()
