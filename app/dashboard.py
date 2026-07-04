import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "http://127.0.0.1:8000/predecir"

st.set_page_config(
    page_title="Dashboard Hipertensión",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Sistema de Clasificación de Hipertensión Arterial")
st.markdown("""
Este dashboard se conecta a una **API FastAPI** que utiliza un modelo de *Machine Learning* para 
clasificar el nivel de hipertensión arterial.

⚠ **Aviso importante**:  
Los resultados son orientativos y **NO reemplazan** una valoración médica profesional.
""")


# ---------------------------------------------------------
# Función para llamar a la API
# ---------------------------------------------------------
def llamar_api(payload: dict):
    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Error al llamar a la API: {e}")
        return None


# ---------------------------------------------------------
# PESTAÑAS
# ---------------------------------------------------------
tab_individual, tab_csv = st.tabs(["👤 Paciente individual", "📊 Análisis masivo (CSV)"])


# =========================================================
# TAB 1: PACIENTE INDIVIDUAL
# =========================================================
with tab_individual:
    st.header("👤 Clasificación de un paciente")

    col1, col2 = st.columns(2)

    with col1:
        edad = st.number_input("Edad", min_value=18, max_value=100, value=40)
        sexo_str = st.selectbox("Sexo", ["Femenino", "Masculino"])
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=75.0)
        talla = st.number_input("Talla (m)", min_value=1.3, max_value=2.1, value=1.70)
        pas = st.number_input("Presión Sistólica (PAS)", min_value=80, max_value=250, value=130)
        pad = st.number_input("Presión Diastólica (PAD)", min_value=50, max_value=150, value=85)

    with col2:
        frec_card = st.number_input("Frecuencia cardíaca (lat/min)", min_value=40, max_value=160, value=80)
        colesterol = st.number_input("Colesterol (mg/dL)", min_value=100, max_value=400, value=200)
        glucosa = st.number_input("Glucosa (mg/dL)", min_value=60, max_value=300, value=100)
        tabaquismo_str = st.selectbox("¿Fuma?", ["No", "Sí"])
        ejercicio = st.slider("Horas de ejercicio/semana", 0, 14, 3)
        estres = st.slider("Nivel de estrés (1-10)", 1, 10, 5)
        herencia_str = st.selectbox("¿Antecedentes familiares de HTA?", ["No", "Sí"])

    # Cálculos derivados
    imc = peso / (talla ** 2)
    pam = (pas + 2 * pad) / 3

    st.markdown(f"**IMC calculado:** `{imc:.1f}` kg/m²")
    st.markdown(f"**PAM calculada:** `{pam:.1f}` mmHg")

    # Codificar variables categóricas
    sexo = 1 if sexo_str == "Masculino" else 0
    tabaquismo = 1 if tabaquismo_str == "Sí" else 0
    herencia = 1 if herencia_str == "Sí" else 0

    if st.button("🔍 Analizar paciente"):
        payload = {
            "Edad": edad,
            "Sexo": sexo,
            "Peso": peso,
            "Talla": talla,
            "IMC": imc,
            "PAS": pas,
            "PAD": pad,
            "PAM": pam,
            "Frec_Card": frec_card,
            "Colesterol": colesterol,
            "Glucosa": glucosa,
            "Tabaquismo": tabaquismo,
            "Ejercicio": ejercicio,
            "Estres": estres,
            "Herencia_HTA": herencia
        }

        resultado = llamar_api(payload)

        if resultado is not None:
            st.subheader("🧠 Resultado del modelo")
            st.write(f"**Diagnóstico numérico:** `{resultado['diagnostico_numerico']}`")
            st.write(f"**Diagnóstico textual:** 👉 **{resultado['diagnostico_texto']}**")
            st.info(resultado.get("aviso", ""))

            proba = resultado.get("probabilidades", None)
            if proba:
                st.subheader("📊 Probabilidades por clase")
                clases = []
                valores = []
                for k, v in proba.items():
                    clases.append(int(k))
                    valores.append(float(v))

                fig, ax = plt.subplots()
                sns.barplot(x=clases, y=valores, ax=ax)
                ax.set_xlabel("Clase")
                ax.set_ylabel("Probabilidad")
                ax.set_title("Distribución de probabilidades")
                st.pyplot(fig)


# =========================================================
# TAB 2: ANÁLISIS MASIVO (CSV)
# =========================================================
with tab_csv:
    st.header("📊 Análisis masivo de pacientes (CSV)")

    st.markdown("""
    Sube un archivo CSV con las columnas necesarias.  
    Si no tiene `IMC` y `PAM`, el sistema las calculará si existen las columnas `Peso`, `Talla`, `PAS`, `PAD`.
    """)

    archivo = st.file_uploader("Subir archivo CSV", type=["csv"])

    if archivo is not None:
        df = pd.read_csv(archivo)
        st.write("Vista previa de los datos:")
        st.dataframe(df.head())

        # Asegurar columnas requeridas
        columnas_necesarias = [
            "Edad", "Sexo", "Peso", "Talla", "PAS", "PAD",
            "Frec_Card", "Colesterol", "Glucosa",
            "Tabaquismo", "Ejercicio", "Estres", "Herencia_HTA"
        ]

        faltantes = [c for c in columnas_necesarias if c not in df.columns]
        if faltantes:
            st.error(f"Faltan columnas obligatorias en el CSV: {faltantes}")
        else:
            # Calcular IMC si no existe
            if "IMC" not in df.columns:
                df["IMC"] = df["Peso"] / (df["Talla"] ** 2)

            # Calcular PAM si no existe
            if "PAM" not in df.columns:
                df["PAM"] = (df["PAS"] + 2 * df["PAD"]) / 3

            if st.button("🚀 Clasificar todos los pacientes"):
                resultados = []
                for i, row in df.iterrows():
                    payload = {
                        "Edad": int(row["Edad"]),
                        "Sexo": int(row["Sexo"]),
                        "Peso": float(row["Peso"]),
                        "Talla": float(row["Talla"]),
                        "IMC": float(row["IMC"]),
                        "PAS": float(row["PAS"]),
                        "PAD": float(row["PAD"]),
                        "PAM": float(row["PAM"]),
                        "Frec_Card": float(row["Frec_Card"]),
                        "Colesterol": float(row["Colesterol"]),
                        "Glucosa": float(row["Glucosa"]),
                        "Tabaquismo": int(row["Tabaquismo"]),
                        "Ejercicio": int(row["Ejercicio"]),
                        "Estres": int(row["Estres"]),
                        "Herencia_HTA": int(row["Herencia_HTA"]),
                    }

                    res = llamar_api(payload)
                    if res is not None:
                        resultados.append(res["diagnostico_numerico"])
                    else:
                        resultados.append(None)

                df["Diagnostico_Predicho"] = resultados

                st.subheader("📋 Resultados de clasificación")
                st.dataframe(df.head())

                st.subheader("📊 Distribución de diagnósticos predichos")
                fig2, ax2 = plt.subplots()
                sns.countplot(x="Diagnostico_Predicho", data=df, ax=ax2)
                ax2.set_xlabel("Clase predicha")
                ax2.set_ylabel("Cantidad de pacientes")
                st.pyplot(fig2)

                # Tabla de frecuencias
                st.write("Frecuencia por clase:")
                st.write(df["Diagnostico_Predicho"].value_counts().sort_index())
