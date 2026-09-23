"""Auditoría reproducible del target leakage del dataset sintético.

Reconstruye la etiqueta `Diagnostico` a partir de las features publicadas en el CSV,
sin entrenar ningún modelo. Si la reconstrucción es casi perfecta, el target es una
función determinista de las entradas y las métricas de cualquier modelo entrenado
sobre este dataset carecen de valor predictivo.

Uso:
    python scripts/verify_leakage.py [--csv RUTA]

Ver docs/LEAKAGE_ANALYSIS.md para la interpretación de los resultados.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path("data/raw/dataset_hipertension_sintetico.csv")


def regla_presion(pas: pd.Series, pad: pd.Series) -> np.ndarray:
    """Clasificación base por presión arterial (guías AHA simplificadas)."""
    return np.select(
        [
            (pas < 120) & (pad < 80),
            ((pas >= 120) & (pas < 130)) | ((pad >= 80) & (pad < 85)),
            ((pas >= 130) & (pas < 140)) | ((pad >= 85) & (pad < 90)),
        ],
        [0, 1, 2],
        default=3,
    ).astype(int)


def regla_completa(df: pd.DataFrame) -> np.ndarray:
    """Reproduce la lógica completa de src/generate_dataset.py, ajustes incluidos."""
    diagnostico = regla_presion(df["PAS"], df["PAD"])

    riesgo_extra = (
        (df["IMC"] > 32).astype(int)
        + (df["Colesterol"] > 240).astype(int)
        + (df["Glucosa"] > 140).astype(int)
        + (df["Estres"] > 7).astype(int)
        + (df["Herencia_HTA"] == 1).astype(int)
    )

    diagnostico = np.where((riesgo_extra >= 3) & (diagnostico < 3), diagnostico + 1, diagnostico)
    diagnostico = np.where((df["Ejercicio"] >= 6) & (diagnostico > 0), diagnostico - 1, diagnostico)
    return diagnostico.astype(int)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Ruta al dataset sintético.")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: no existe {args.csv}. Genera el dataset con:", file=sys.stderr)
        print("    python src/generate_dataset.py", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    y = df["Diagnostico"]

    dist = y.value_counts(normalize=True).sort_index().round(4).to_dict()
    acc_base = float((regla_presion(df["PAS"], df["PAD"]) == y).mean())
    acc_full = float((regla_completa(df) == y).mean())
    mayoritaria = float(y.value_counts(normalize=True).max())

    print(f"Dataset: {args.csv}  ({len(df):,} filas)")
    print(f"Distribución de clases:  {dist}\n")
    print(f"Baseline clase mayoritaria                → {mayoritaria:7.2%}")
    print(f"Regla PAS/PAD pura                        → {acc_base:7.2%} de coincidencia")
    print(f"Regla completa (con ajustes de riesgo)    → {acc_full:7.2%} de coincidencia\n")

    if acc_full > 0.85:
        print("VEREDICTO: target leakage confirmado.")
        print("  El target es una función determinista de las features. La brecha")
        print("  restante es ruido de redondeo del generador, no señal aprendible.")
        print("  Las métricas de los modelos NO miden capacidad predictiva.")
        print("  Ver docs/LEAKAGE_ANALYSIS.md")
    else:
        print("VEREDICTO: la reconstrucción no es determinista; revisar el generador.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
