"""Verificación del entorno: los artefactos de DT-1 cargan y predicen.

Criterio de éxito del entorno reproducible. Comprueba, en este orden:

1. Las versiones instaladas coinciden con `entorno` del manifiesto de DT-1
   (perfil «entorno» de src/artefactos.py: Python por major.minor; sklearn,
   numpy, pandas, joblib y xgboost exactos). Un .pkl de scikit-learn cargado
   con otra versión puede fallar en silencio.
2. El sha256 de cada artefacto coincide con el registrado en el manifiesto.
3. Cada modelo y su scaler cargan sin excepción ni UserWarning de versión.
4. Control de identidad: `mean_` y `scale_` del scaler cargado reproducen los
   valores del manifiesto. Cargar «un» scaler no prueba nada; hay que
   demostrar que es *el* scaler de la corrida de referencia.
5. Predicción de prueba sobre dos perfiles y control de monotonía: el perfil
   de riesgo alto debe recibir probabilidad mayor que el de riesgo bajo.

Los pasos 1-4 son los mismos que ejecuta la API al arrancar (src/artefactos.py).

Uso (desde la raíz del repositorio):
    python -m scripts.verify_env

Salida: código 0 si todo pasa, 1 en el primer fallo.

AVISO: este proyecto es una demostración de ingeniería de ML, no una
herramienta clínica. Ninguna salida aquí es un diagnóstico médico.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.artefactos import ArtefactoInvalido, cargar_experimento, comparar_versiones, experimentos
from src.cardio_features import bmi

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MANIFEST_PATH = MODEL_DIR / "dt1_manifest.json"

# Perfiles de prueba en el orden de features de cada experimento. Valores
# plausibles, no reales: sirven para ejercitar el grafo de inferencia.
PERFILES: dict[str, dict[str, float]] = {
    "riesgo_bajo": {
        "age_years": 40.0,
        "gender": 1.0,
        "height": 170.0,
        "weight": 65.0,
        "bmi": bmi(65.0, 170.0),
        "cholesterol": 1.0,
        "gluc": 1.0,
        "smoke": 0.0,
        "alco": 0.0,
        "active": 1.0,
        "ap_hi": 110.0,
        "ap_lo": 70.0,
    },
    "riesgo_alto": {
        "age_years": 62.0,
        "gender": 2.0,
        "height": 168.0,
        "weight": 95.0,
        "bmi": bmi(95.0, 168.0),
        "cholesterol": 3.0,
        "gluc": 3.0,
        "smoke": 1.0,
        "alco": 1.0,
        "active": 0.0,
        "ap_hi": 165.0,
        "ap_lo": 100.0,
    },
}


def fallo(mensaje: str) -> None:
    print(f"  ✗ {mensaje}")
    sys.exit(1)


def verificar_versiones(manifiesto: dict, ganadores: list[str]) -> None:
    print("1. Versiones contra el manifiesto de la corrida de referencia (perfil entorno)")
    print(f"  {'paquete':16s} {'esperado':12s} {'instalado':12s}")
    filas = comparar_versiones(manifiesto, "entorno", ganadores)
    for f in filas:
        print(f"  {'✓' if f.ok else '✗'} {f.paquete:14s} {f.esperado:12s} {f.instalado:12s}")
    desviaciones = [f.paquete for f in filas if not f.ok]
    if desviaciones:
        fallo(f"versiones distintas a la corrida de referencia: {', '.join(desviaciones)}")
    print("  → coinciden todas (python por major.minor)\n")


def verificar_experimento(exp: dict) -> None:
    features = exp["features_orden"]
    print(f"[{exp['nombre']}]  target={exp['target']}  ganador={exp['ganador']}  n_features={len(features)}")

    try:
        cargado = cargar_experimento(exp, MODEL_DIR)
    except ArtefactoInvalido as exc:
        fallo(str(exc))
    print("  ✓ sha256 de modelo y scaler coinciden con el manifiesto")
    print(f"  ✓ cargan sin warnings — {type(cargado.modelo).__name__} + {type(cargado.scaler).__name__}")
    print("  ✓ control de identidad: mean_ y scale_ reproducen el manifiesto")

    # --- predicción de prueba ---
    X = np.array([[PERFILES[p][f] for f in features] for p in ("riesgo_bajo", "riesgo_alto")], dtype=float)
    Xs = cargado.scaler.transform(X)
    pred = cargado.modelo.predict(Xs)
    proba = cargado.modelo.predict_proba(Xs)[:, 1]

    for etiqueta, clase, p in zip(("riesgo_bajo", "riesgo_alto"), pred, proba, strict=True):
        print(f"    {etiqueta:12s} → clase {int(clase)}   P(positivo) = {p:.4f}")

    # --- control de monotonía ---
    if not proba[1] > proba[0]:
        fallo(f"el perfil de riesgo alto no supera al bajo ({proba[1]:.4f} <= {proba[0]:.4f})")
    print(f"  ✓ control de monotonía: alto > bajo (Δ = {proba[1] - proba[0]:+.4f})\n")


def main() -> None:
    if not MANIFEST_PATH.exists():
        fallo(f"no existe {MANIFEST_PATH.relative_to(ROOT)}")
    manifiesto = json.loads(MANIFEST_PATH.read_text())

    print("=" * 78)
    print("VERIFICACIÓN DEL ENTORNO — artefactos de DT-1")
    print("=" * 78 + "\n")

    bloques = list(experimentos(manifiesto).values())
    if not bloques:
        fallo("el manifiesto no declara ningún experimento con artefactos")
    verificar_versiones(manifiesto, [b["ganador"] for b in bloques])

    print("2. Carga y predicción por experimento\n")
    for exp in bloques:
        verificar_experimento(exp)

    print("=" * 78)
    print(f"ENTORNO LISTO — {len(bloques)} experimentos cargan y predicen sin error")
    print("Recordatorio: demostración de ingeniería de ML, no herramienta clínica.")
    print("=" * 78)


if __name__ == "__main__":
    main()
