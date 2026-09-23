"""Verificación del entorno: los artefactos de DT-1 cargan y predicen.

Criterio de éxito del entorno reproducible. Comprueba, en este orden:

1. Las versiones instaladas coinciden con `entorno` del manifiesto de DT-1.
   Un .pkl de scikit-learn cargado con otra versión puede fallar en silencio.
2. El sha256 de cada artefacto coincide con el registrado en el manifiesto.
3. Cada modelo y su scaler cargan sin excepción ni UserWarning de versión.
4. Control de identidad: `mean_` y `scale_` del scaler cargado reproducen los
   valores del manifiesto. Cargar «un» scaler no prueba nada; hay que
   demostrar que es *el* scaler de la corrida de referencia.
5. Predicción de prueba sobre dos perfiles y control de monotonía: el perfil
   de riesgo alto debe recibir probabilidad mayor que el de riesgo bajo.

Uso:
    python scripts/verify_env.py

Salida: código 0 si todo pasa, 1 en el primer fallo.

AVISO: este proyecto es una demostración de ingeniería de ML, no una
herramienta clínica. Ninguna salida aquí es un diagnóstico médico.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "dt1_manifest.json"
TOL = 1e-9

# Perfiles de prueba en el orden de features de cada experimento. Valores
# plausibles, no reales: sirven para ejercitar el grafo de inferencia.
PERFILES: dict[str, dict[str, float]] = {
    "riesgo_bajo": {
        "age_years": 40.0,
        "gender": 1.0,
        "height": 170.0,
        "weight": 65.0,
        "bmi": 65.0 / (1.70**2),
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
        "bmi": 95.0 / (1.68**2),
        "cholesterol": 3.0,
        "gluc": 3.0,
        "smoke": 1.0,
        "alco": 1.0,
        "active": 0.0,
        "ap_hi": 165.0,
        "ap_lo": 100.0,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fallo(mensaje: str) -> None:
    print(f"  ✗ {mensaje}")
    sys.exit(1)


def experimentos(manifiesto: dict) -> list[dict]:
    """Recorre el manifiesto y devuelve cada bloque con artefactos."""
    encontrados: list[dict] = []

    def recorrer(nodo: object) -> None:
        if isinstance(nodo, dict):
            if "artefactos" in nodo and "features_orden" in nodo:
                encontrados.append(nodo)
            for valor in nodo.values():
                recorrer(valor)

    recorrer(manifiesto["experimentos"])
    return encontrados


def verificar_versiones(esperado: dict[str, str]) -> None:
    import importlib.metadata as md

    print("1. Versiones contra el manifiesto de la corrida de referencia")
    print(f"  {'paquete':16s} {'manifiesto':12s} {'instalado':12s}")

    actual = {
        "python": ".".join(str(n) for n in sys.version_info[:3]),
        "scikit_learn": md.version("scikit-learn"),
        "xgboost": md.version("xgboost"),
        "pandas": md.version("pandas"),
        "numpy": md.version("numpy"),
    }

    desviaciones = []
    for paquete, version_ref in esperado.items():
        version_act = actual.get(paquete, "AUSENTE")
        marca = "✓" if version_act == version_ref else "✗"
        print(f"  {marca} {paquete:14s} {version_ref:12s} {version_act:12s}")
        if version_act != version_ref:
            desviaciones.append(paquete)

    if desviaciones:
        fallo(f"versiones distintas a la corrida de referencia: {', '.join(desviaciones)}")
    print("  → coinciden todas\n")


def verificar_experimento(exp: dict) -> None:
    nombre = exp["nombre"]
    features = exp["features_orden"]
    print(f"[{nombre}]  target={exp['target']}  ganador={exp['ganador']}  n_features={len(features)}")

    # --- sha256 de los artefactos ---
    rutas = {}
    for clave in ("modelo", "scaler"):
        ruta = ROOT / exp["artefactos"][clave]["ruta"]
        if not ruta.exists():
            fallo(f"falta {ruta.relative_to(ROOT)}")
        digest = sha256_file(ruta)
        if digest != exp["artefactos"][clave]["sha256"]:
            fallo(f"sha256 distinto en {ruta.relative_to(ROOT)}: {digest[:16]}… != manifiesto")
        rutas[clave] = ruta
    print("  ✓ sha256 de modelo y scaler coinciden con el manifiesto")

    # --- carga, tratando cualquier UserWarning de versión como fallo ---
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            modelo = joblib.load(rutas["modelo"])
            scaler = joblib.load(rutas["scaler"])
        except Exception as exc:  # noqa: BLE001 - el mensaje exacto es el diagnóstico
            fallo(f"carga fallida: {type(exc).__name__}: {exc}")
    print(f"  ✓ cargan sin warnings — {type(modelo).__name__} + {type(scaler).__name__}")

    # --- control de identidad del scaler ---
    ref = exp["scaler"]
    for atributo in ("mean_", "scale_"):
        delta = float(np.max(np.abs(getattr(scaler, atributo) - np.asarray(ref[atributo]))))
        if delta > TOL:
            fallo(f"{atributo} del scaler se desvía del manifiesto (máx |Δ| = {delta:.3e})")
    print("  ✓ control de identidad: mean_ y scale_ reproducen el manifiesto")

    if scaler.n_features_in_ != len(features):
        fallo(f"el scaler espera {scaler.n_features_in_} features, el manifiesto declara {len(features)}")

    # --- predicción de prueba ---
    X = np.array([[PERFILES[p][f] for f in features] for p in ("riesgo_bajo", "riesgo_alto")], dtype=float)
    Xs = scaler.transform(X)
    pred = modelo.predict(Xs)
    proba = modelo.predict_proba(Xs)[:, 1]

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

    verificar_versiones(manifiesto["entorno"])

    print("2. Carga y predicción por experimento\n")
    bloques = experimentos(manifiesto)
    if not bloques:
        fallo("el manifiesto no declara ningún experimento con artefactos")
    for exp in bloques:
        verificar_experimento(exp)

    print("=" * 78)
    print(f"ENTORNO LISTO — {len(bloques)} experimentos cargan y predicen sin error")
    print("Recordatorio: demostración de ingeniería de ML, no herramienta clínica.")
    print("=" * 78)


if __name__ == "__main__":
    main()
