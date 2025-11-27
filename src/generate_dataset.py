import numpy as np
import pandas as pd
import os

def generar_dataset(n=50000, seed=42):
    """
    Genera un dataset sintético grande de hipertensión arterial.
    Las distribuciones son realistas y permiten entrenar modelos ML de forma estable.
    """

    np.random.seed(seed)

    # -----------------------------
    # Variables demográficas
    # -----------------------------
    edad = np.random.randint(18, 90, n)

    sexo = np.random.choice([0, 1], size=n, p=[0.48, 0.52])  
    # 0 = F, 1 = M

    # -----------------------------
    # Variables antropométricas
    # -----------------------------
    peso = np.random.normal(75, 15, n).clip(40, 160)
    talla = np.random.normal(1.68, 0.12, n).clip(1.40, 2.00)

    imc = peso / (talla ** 2)

    # -----------------------------
    # Cardiovasculares
    # -----------------------------
    pas = np.random.normal(130, 20, n).clip(85, 240)   # presión sistólica
    pad = np.random.normal(82, 12, n).clip(55, 140)    # presión diastólica

    pam = (pas + 2 * pad) / 3

    frec_card = np.random.normal(78, 12, n).clip(45, 160)

    # -----------------------------
    # Bioquímicas
    # -----------------------------
    colesterol = np.random.normal(200, 40, n).clip(100, 350)
    glucosa = np.random.normal(105, 25, n).clip(60, 300)

    # -----------------------------
    # Estilo de vida / riesgo
    # -----------------------------
    tabaquismo = np.random.choice([0, 1], n, p=[0.7, 0.3])
    ejercicio = np.random.randint(0, 10, n)
    estres = np.random.randint(1, 10, n)
    herencia = np.random.choice([0, 1], n, p=[0.6, 0.4])

    # -----------------------------
    # Diagnóstico (reglas médicas + ruido)
    # -----------------------------
    diagnostico = np.zeros(n, dtype=int)

    for i in range(n):

        PAS = pas[i]
        PAD = pad[i]

        # Clasificación base por presión (Guías AHA)
        if PAS < 120 and PAD < 80:
            diagnostico[i] = 0  # Normal
        elif 120 <= PAS < 130 or 80 <= PAD < 85:
            diagnostico[i] = 1  # Prehipertensión
        elif 130 <= PAS < 140 or 85 <= PAD < 90:
            diagnostico[i] = 2  # HTA Grado 1
        else:
            diagnostico[i] = 3  # HTA Grado 2

        # Ruido fisiológico basado en otros factores
        riesgo_extra = (
            (imc[i] > 32) +
            (colesterol[i] > 240) +
            (glucosa[i] > 140) +
            (estres[i] > 7) +
            (herencia[i] == 1)
        )

        # Si tiene mucho riesgo, puede subir de nivel
        if riesgo_extra >= 3 and diagnostico[i] < 3:
            diagnostico[i] += 1

        # Con buen ejercicio podría bajar un nivel
        if ejercicio[i] >= 6 and diagnostico[i] > 0:
            diagnostico[i] -= 1

    # -----------------------------
    # Construcción del DataFrame
    # -----------------------------
    df = pd.DataFrame({
        "Edad": edad,
        "Sexo": sexo,
        "Peso": peso.round(1),
        "Talla": talla.round(2),
        "IMC": imc.round(1),
        "PAS": pas.round(0),
        "PAD": pad.round(0),
        "PAM": pam.round(1),
        "Frec_Card": frec_card.round(0),
        "Colesterol": colesterol.round(0),
        "Glucosa": glucosa.round(0),
        "Tabaquismo": tabaquismo,
        "Ejercicio": ejercicio,
        "Estres": estres,
        "Herencia_HTA": herencia,
        "Diagnostico": diagnostico
    })

    return df


def guardar_dataset(df, ruta="data/raw/dataset_hipertension_sintetico.csv"):
    """Guarda el DataFrame generado en la ruta indicada."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df.to_csv(ruta, index=False)
    print(f"✔ Dataset generado y guardado en: {ruta}")


if __name__ == "__main__":
    df = generar_dataset()
    guardar_dataset(df)
