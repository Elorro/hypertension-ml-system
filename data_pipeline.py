import argparse
from pathlib import Path
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "Edad",
    "Sexo",
    "Peso",
    "Talla",
    "IMC",
    "PAS",
    "PAD",
    "PAM",
    "Frec_Card",
    "Colesterol",
    "Glucosa",
    "Tabaquismo",
    "Ejercicio",
    "Estrés",
    "Herencia_HTA",
]

TARGET_COLUMN = "HTA_Nivel"
LABEL_MAP = {
    0: "Normal",
    1: "Prehipertensión",
    2: "Hipertensión Grado 1",
    3: "Hipertensión Grado 2",
}


def generate_synthetic_hypertension_data(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    """
    Genera un DataFrame sintético de pacientes con variables plausibles para clasificación de HTA.
    """
    rng = np.random.default_rng(seed)

    sexo = rng.binomial(1, 0.5, size=n_samples)
    edad = rng.integers(18, 86, size=n_samples)

    talla = rng.normal(loc=np.where(sexo == 1, 1.75, 1.63), scale=0.07)
    talla = np.clip(talla, 1.45, 2.0)

    imc = rng.normal(loc=26, scale=4, size=n_samples)
    imc = np.clip(imc, 17, 42)
    peso = imc * (talla ** 2)

    colesterol = rng.normal(loc=190, scale=35, size=n_samples)
    colesterol = np.clip(colesterol, 120, 320)

    glucosa = rng.normal(loc=100, scale=20, size=n_samples)
    glucosa = np.clip(glucosa, 70, 240)

    tabaquismo = rng.binomial(1, 0.25, size=n_samples)
    ejercicio = rng.binomial(1, 0.55, size=n_samples)
    estres = rng.normal(loc=4, scale=2, size=n_samples)
    estres = np.clip(estres, 0, 10)
    herencia = rng.binomial(1, 0.3, size=n_samples)

    # Riesgo continuo que alimenta las variables de presión arterial y FC.
    riesgo = (
        (edad - 40) * 0.02
        + (imc - 25) * 0.05
        + (colesterol - 190) * 0.003
        + (glucosa - 100) * 0.003
        + 0.45 * tabaquismo
        - 0.25 * ejercicio
        + 0.5 * (estres / 10)
        + 0.35 * herencia
    )

    pas = 115 + riesgo * 18 + edad * 0.05 + rng.normal(0, 10, size=n_samples)
    pad = 75 + riesgo * 10 + edad * 0.03 + rng.normal(0, 8, size=n_samples)
    pas = np.clip(pas, 90, 220)
    pad = np.clip(pad, 50, 130)

    pam = (2 * pad + pas) / 3
    frec_card = 70 + 8 * tabaquismo - 6 * ejercicio + estres + rng.normal(0, 5, size=n_samples)
    frec_card = np.clip(frec_card, 45, 140)

    # Etiquetado según guías clínicas de HTA.
    target = np.select(
        [
            (pas < 120) & (pad < 80),
            ((pas >= 120) & (pas < 140)) | ((pad >= 80) & (pad < 90)),
            ((pas >= 140) & (pas < 160)) | ((pad >= 90) & (pad < 100)),
            (pas >= 160) | (pad >= 100),
        ],
        [0, 1, 2, 3],
    )

    df = pd.DataFrame(
        {
            "Edad": edad,
            "Sexo": sexo,
            "Peso": peso.round(2),
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
            "Estrés": estres.round(1),
            "Herencia_HTA": herencia,
            "HTA_Nivel": target.astype(int),
        }
    )
    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Genera dataset sintético de HTA.")
    parser.add_argument("-n", "--n_samples", type=int, default=50000, help="Número de filas a generar.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/hypertension_synthetic.csv"),
        help="Ruta de salida para el CSV.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad.")
    args = parser.parse_args()

    df = generate_synthetic_hypertension_data(n_samples=args.n_samples, seed=args.seed)
    save_dataset(df, args.output)
    print(f"Dataset guardado en {args.output.resolve()} (shape={df.shape})")


if __name__ == "__main__":
    main()
