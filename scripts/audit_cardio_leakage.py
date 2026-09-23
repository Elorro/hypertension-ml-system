"""Criterio de aceptación de DT-1: auditoría de leakage sobre el dataset real.

Contraparte de ``scripts/verify_leakage.py`` (que audita el dataset sintético).
Aquí se pregunta lo mismo sobre ``data/real/cardio/cardio_train.csv``:

    ¿Existe una regla determinista sobre las features que recupere el target muy
    por encima del baseline de clase mayoritaria?

Si la respuesta es sí, el target vuelve a ser una función de las entradas y las
métricas de cualquier modelo son tautológicas. El ROADMAP fija el umbral de
aceptación de DT-1 en **baseline + 5 puntos porcentuales**.

Se auditan dos configuraciones con la misma maquinaria:

* ``hta_sin_presion`` — la configuración real del experimento B1: target
  ``hta = (ap_hi >= 140) | (ap_lo >= 90)`` con las 10 features no-presión.
  Es la que debe **pasar**.
* ``hta_con_presion`` — control positivo: las mismas reglas con ``ap_hi``/``ap_lo``
  añadidas a las features. Debe **fallar** de forma escandalosa. Sirve para
  demostrar que el buscador de reglas sí detecta leakage cuando existe; un
  "pasa" sin este control no prueba nada.

Familias de reglas buscadas (todas ajustadas en train, evaluadas en test):

1. Clase mayoritaria (baseline).
2. Mejor umbral univariado, por feature y por dirección.
3. Árbol CART de profundidad <= 3 (implementación propia, gini, umbrales
   candidatos en deciles). Equivale a la regla ``if`` determinista del análisis
   sintético (``scripts/verify_leakage.py::regla_completa``): si una regla
   legible a mano recupera el target, se ve aquí.

Sin scikit-learn: solo numpy y pandas, para que la auditoría corra aunque el
entorno de entrenamiento no esté instalado.

Uso::

    python scripts/audit_cardio_leakage.py [--csv RUTA] [--manifiesto RUTA]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data/real/cardio/cardio_train.csv"
DEFAULT_MANIFEST = ROOT / "models/dt1_manifest.json"

# Mismos parámetros que src/train_cardio_real.py. Duplicados a propósito: la
# auditoría no importa el módulo de entrenamiento (arrastraría sklearn/xgboost) y
# así la limpieza queda reimplementada de forma independiente. La coincidencia con
# el manifiesto se verifica al final.
SEED = 42
TEST_SIZE = 0.20
HEIGHT_CM_RANGE = (120.0, 220.0)
WEIGHT_KG_RANGE = (30.0, 200.0)
BMI_RANGE = (12.0, 70.0)
HTA_SBP_THRESHOLD = 140
HTA_DBP_THRESHOLD = 90

FEATURES_SIN_PA = [
    "age_years",
    "gender",
    "height",
    "weight",
    "bmi",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]
FEATURES_CON_PA = FEATURES_SIN_PA + ["ap_hi", "ap_lo"]

MAX_DEPTH = 3
N_QUANTILES = 10  # umbrales candidatos por feature
TOLERANCIA_PP = 5.0  # criterio de aceptación de DT-1, en puntos porcentuales


# =======================================================
# Datos
# =======================================================
def cargar_y_limpiar(csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reimplementa la limpieza de src/train_cardio_real.py y reporta conteos."""
    raw = pd.read_csv(csv, sep=";")
    reporte: dict[str, Any] = {"filas_crudas": len(raw)}

    df = raw.drop_duplicates(subset=[c for c in raw.columns if c != "id"], keep="first")
    reporte["tras_duplicados"] = len(df)

    df = df[df["ap_lo"] < df["ap_hi"]]
    reporte["tras_presion_invertida"] = len(df)

    bmi = df["weight"] / (df["height"] / 100.0) ** 2
    df = df[
        df["height"].between(*HEIGHT_CM_RANGE)
        & df["weight"].between(*WEIGHT_KG_RANGE)
        & bmi.between(*BMI_RANGE)
    ].copy()
    reporte["filas_finales"] = len(df)

    df["age_years"] = df["age"] / 365.25
    df["bmi"] = df["weight"] / (df["height"] / 100.0) ** 2
    df["hta"] = ((df["ap_hi"] >= HTA_SBP_THRESHOLD) | (df["ap_lo"] >= HTA_DBP_THRESHOLD)).astype(int)
    reporte["prevalencia_hta"] = float(df["hta"].mean())
    return df.reset_index(drop=True), reporte


def split_estratificado(y: np.ndarray, test_size: float = TEST_SIZE, seed: int = SEED):
    """Partición estratificada propia (sin sklearn).

    NO reproduce fila a fila la partición de src/train_cardio_real.py: es una
    partición independiente. La auditoría no necesita compartir el test del
    modelo — busca reglas, no compara modelos — y así las conclusiones sobre
    reglas no dependen de la partición usada para seleccionarlos.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for clase in np.unique(y):
        idx = np.flatnonzero(y == clase)
        rng.shuffle(idx)
        corte = int(round(len(idx) * test_size))
        test_idx.append(idx[:corte])
        train_idx.append(idx[corte:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(test_idx))


# =======================================================
# Métricas sin sklearn
# =======================================================
def auc(y: np.ndarray, score: np.ndarray) -> float:
    """AUC por la identidad de Mann-Whitney, con empates promediados."""
    orden = np.argsort(score, kind="mergesort")
    rangos = np.empty(len(score), dtype=float)
    s = score[orden]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rangos[orden[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    return float((rangos[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# =======================================================
# Reglas deterministas
# =======================================================
def umbrales_candidatos(col: np.ndarray, n: int = N_QUANTILES) -> np.ndarray:
    qs = np.quantile(col, np.linspace(0, 1, n + 1)[1:-1])
    return np.unique(qs)


def regla_univariada(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, nombres: list[str]
) -> dict[str, Any]:
    """Mejor regla `feature <= t -> clase` ajustada en train, evaluada en test."""
    mejor = {"accuracy_train": -1.0}
    for j, nombre in enumerate(nombres):
        for t in umbrales_candidatos(X_tr[:, j]):
            izq = X_tr[:, j] <= t
            for clase_izq in (0, 1):
                pred = np.where(izq, clase_izq, 1 - clase_izq)
                acc = float((pred == y_tr).mean())
                if acc > mejor["accuracy_train"]:
                    pred_te = np.where(X_te[:, j] <= t, clase_izq, 1 - clase_izq)
                    mejor = {
                        "feature": nombre,
                        "umbral": float(t),
                        "clase_si_menor_igual": clase_izq,
                        "accuracy_train": acc,
                        "accuracy_test": float((pred_te == y_te).mean()),
                    }
    return mejor


@dataclass
class Nodo:
    feature: int | None = None
    umbral: float = 0.0
    valor: float = 0.0  # proporción de positivos en la hoja
    izq: Nodo | None = None
    der: Nodo | None = None
    n: int = 0

    @property
    def es_hoja(self) -> bool:
        return self.feature is None


def _gini_ponderado(y_izq: np.ndarray, y_der: np.ndarray) -> float:
    n = len(y_izq) + len(y_der)
    total = 0.0
    for parte in (y_izq, y_der):
        if len(parte) == 0:
            continue
        p = parte.mean()
        total += len(parte) * 2.0 * p * (1.0 - p)
    return total / n


def entrenar_arbol(X: np.ndarray, y: np.ndarray, cands: list[np.ndarray], prof: int) -> Nodo:
    """CART binario, criterio gini, umbrales candidatos fijos (deciles de train)."""
    nodo = Nodo(valor=float(y.mean()), n=len(y))
    if prof == 0 or len(np.unique(y)) == 1 or len(y) < 50:
        return nodo

    mejor_gini = _gini_ponderado(y, np.array([]))
    mejor: tuple[int, float] | None = None
    for j in range(X.shape[1]):
        for t in cands[j]:
            izq = X[:, j] <= t
            if izq.all() or (~izq).all():
                continue
            g = _gini_ponderado(y[izq], y[~izq])
            if g < mejor_gini - 1e-12:
                mejor_gini, mejor = g, (j, float(t))

    if mejor is None:
        return nodo
    j, t = mejor
    izq = X[:, j] <= t
    nodo.feature, nodo.umbral = j, t
    nodo.izq = entrenar_arbol(X[izq], y[izq], cands, prof - 1)
    nodo.der = entrenar_arbol(X[~izq], y[~izq], cands, prof - 1)
    return nodo


def predecir_arbol(nodo: Nodo, X: np.ndarray) -> np.ndarray:
    if nodo.es_hoja:
        return np.full(len(X), int(nodo.valor >= 0.5))
    izq = X[:, nodo.feature] <= nodo.umbral
    out = np.empty(len(X), dtype=int)
    if izq.any():
        out[izq] = predecir_arbol(nodo.izq, X[izq])
    if (~izq).any():
        out[~izq] = predecir_arbol(nodo.der, X[~izq])
    return out


def imprimir_arbol(nodo: Nodo, nombres: list[str], sangria: str = "    ") -> list[str]:
    if nodo.es_hoja:
        return [f"{sangria}return {int(nodo.valor >= 0.5)}  # n={nodo.n}, p(hta)={nodo.valor:.3f}"]
    lineas = [f"{sangria}if {nombres[nodo.feature]} <= {nodo.umbral:.4g}:"]
    lineas += imprimir_arbol(nodo.izq, nombres, sangria + "    ")
    lineas.append(f"{sangria}else:")
    lineas += imprimir_arbol(nodo.der, nombres, sangria + "    ")
    return lineas


# =======================================================
# Auditoría de una configuración
# =======================================================
def auditar(df: pd.DataFrame, features: list[str], target: str, etiqueta: str) -> dict[str, Any]:
    y = df[target].to_numpy()
    tr, te = split_estratificado(y)
    X = df[features].to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

    base = float(max(y_te.mean(), 1 - y_te.mean()))
    uni = regla_univariada(X_tr, y_tr, X_te, y_te, features)

    cands = [umbrales_candidatos(X_tr[:, j]) for j in range(X_tr.shape[1])]
    arbol = entrenar_arbol(X_tr, y_tr, cands, MAX_DEPTH)
    pred_te = predecir_arbol(arbol, X_te)
    acc_arbol = float((pred_te == y_te).mean())

    mejor_regla = max(uni["accuracy_test"], acc_arbol)
    ganancia_pp = (mejor_regla - base) * 100.0

    print(f"\n{'=' * 78}")
    print(f"{etiqueta}   target={target}   {len(features)} features   n_test={len(y_te):,}")
    print("=" * 78)
    print(f"  Baseline clase mayoritaria        {base * 100:6.2f} %")
    print(
        f"  Mejor umbral univariado           {uni['accuracy_test'] * 100:6.2f} %"
        f"   [{uni['feature']} <= {uni['umbral']:.4g}]"
    )
    print(f"  Árbol CART profundidad <= {MAX_DEPTH}       {acc_arbol * 100:6.2f} %")
    print(f"  -> ganancia de la mejor regla     {ganancia_pp:+6.2f} pp sobre el baseline")
    print("\n  Regla legible equivalente:")
    print("\n".join(imprimir_arbol(arbol, features)))

    return {
        "etiqueta": etiqueta,
        "target": target,
        "n_features": len(features),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "baseline_mayoritaria": base,
        "regla_univariada": uni,
        "arbol_cart": {"profundidad_max": MAX_DEPTH, "accuracy_test": acc_arbol},
        "mejor_regla_accuracy_test": mejor_regla,
        "ganancia_pp_sobre_baseline": ganancia_pp,
    }


def auditoria_univariada(df: pd.DataFrame, features: list[str], target: str) -> dict[str, Any]:
    """AUC univariada y correlación de cada feature con la presión arterial.

    Una feature contaminada por la presión se delata aquí: AUC muy alta o
    |spearman| cercano a 1 con ap_hi/ap_lo.
    """
    y = df[target].to_numpy()
    out = {}
    for f in features:
        col = df[f].to_numpy(dtype=float)
        a = auc(y, col)
        out[f] = {
            "auc_univariada": float(max(a, 1 - a)),
            "corr_ap_hi": spearman(col, df["ap_hi"].to_numpy(dtype=float)),
            "corr_ap_lo": spearman(col, df["ap_lo"].to_numpy(dtype=float)),
        }
    return out


# =======================================================
# Principal
# =======================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--manifiesto", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"ERROR: no existe {args.csv}", file=sys.stderr)
        return 2

    df, limpieza = cargar_y_limpiar(args.csv)
    print("Limpieza (reimplementada, independiente del script de entrenamiento):")
    for k, v in limpieza.items():
        print(f"  {k:<28} {v:,.6g}" if isinstance(v, float) else f"  {k:<28} {v:,}")

    # Contraste con el manifiesto de la corrida de entrenamiento, si está.
    if args.manifiesto.exists():
        man = json.loads(args.manifiesto.read_text())
        esperado = man["limpieza"]["filas_finales"]
        ok = esperado == limpieza["filas_finales"]
        print(
            f"\n  Filas finales vs. manifiesto DT-1: {limpieza['filas_finales']:,} vs "
            f"{esperado:,}  ->  {'coinciden' if ok else 'NO COINCIDEN'}"
        )
        if not ok:
            print("  AVISO: la limpieza divergió; el resto del informe no es comparable.")

    sin_pa = auditar(df, FEATURES_SIN_PA, "hta", "B1 — hta SIN presión en features (configuración real)")
    con_pa = auditar(df, FEATURES_CON_PA, "hta", "CONTROL POSITIVO — hta CON ap_hi/ap_lo en features")

    print(
        f"\n{'=' * 78}\nLinaje de las features de B1 (AUC univariada y correlación con la presión)\n{'=' * 78}"
    )
    lin = auditoria_univariada(df, FEATURES_SIN_PA, "hta")
    print(f"  {'feature':<12} {'AUC univ.':>10} {'corr ap_hi':>12} {'corr ap_lo':>12}")
    for f, v in sorted(lin.items(), key=lambda kv: -kv[1]["auc_univariada"]):
        print(f"  {f:<12} {v['auc_univariada']:>10.4f} {v['corr_ap_hi']:>12.4f} {v['corr_ap_lo']:>12.4f}")

    # --- Veredicto ---
    print(f"\n{'=' * 78}\nVEREDICTO — criterio de aceptación DT-1\n{'=' * 78}")
    print(
        f"  Criterio: ninguna regla determinista sobre las features supera al baseline "
        f"de clase mayoritaria por más de {TOLERANCIA_PP:.0f} pp."
    )
    print(
        f"  Control positivo (con presión):  {con_pa['ganancia_pp_sobre_baseline']:+6.2f} pp  "
        f"-> debe fallar: {'FALLA (correcto)' if con_pa['ganancia_pp_sobre_baseline'] > TOLERANCIA_PP else 'NO FALLA (la auditoría no detecta leakage; revisar)'}"
    )
    aprueba = sin_pa["ganancia_pp_sobre_baseline"] <= TOLERANCIA_PP
    print(
        f"  Configuración real (sin presión): {sin_pa['ganancia_pp_sobre_baseline']:+6.2f} pp  "
        f"-> {'PASA' if aprueba else 'NO PASA'}"
    )

    if args.manifiesto.exists():
        man = json.loads(args.manifiesto.read_text())
        b1 = man["experimentos"]["B1_experimento"]["principal"]
        g = b1["modelos"][b1["ganador"]]
        print(
            f"\n  Referencia — mejor modelo entrenado ({b1['ganador']}): "
            f"accuracy {g['accuracy'] * 100:.2f} % vs baseline {g['accuracy_baseline_mayoritaria'] * 100:.2f} % "
            f"({(g['accuracy'] - g['accuracy_baseline_mayoritaria']) * 100:+.2f} pp), AUC {g['roc_auc']:.4f}."
        )
        print("  El modelo tampoco despega del baseline en accuracy: la señal que tiene")
        print("  está en el ranking (AUC), no en la clasificación con umbral 0,5.")

    return 0 if aprueba else 1


if __name__ == "__main__":
    sys.exit(main())
