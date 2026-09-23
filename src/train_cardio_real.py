"""DT-1 — Entrenamiento sobre el dataset real de Kaggle (cardio_train.csv).

Dos problemas, cinco algoritmos cada uno, selección por macro-F1:

* **A′ (principal)** — riesgo cardiovascular, target ``cardio``.
  Ablación obligatoria: variante con ``ap_hi``/``ap_lo`` y variante sin ellas.
* **B1 (experimento)** — hipertensión honesta,
  ``hta = (ap_hi >= 140) | (ap_lo >= 90)`` (umbral JNC7/ESC fijado a priori).
  Las features excluyen toda la información de presión arterial.
  Robustez: se reentrena con ``>`` sobre la misma partición. Es un reporte, no una
  segunda elección de umbral: no se guardan artefactos de esa corrida.

Decisiones de diseño no negociables (ver BITACORA.md):

* Limpieza de etiqueta antes del split: duplicados exactos (ignorando ``id``),
  presión invertida (``ap_lo >= ap_hi``, se descarta, no se corrige) y
  antropometría fisiológicamente imposible.
* Split estratificado por el target, 80/20, ``random_state=42``.
* ``StandardScaler`` ajustado exclusivamente sobre train.
* Ningún ``.pkl`` del pipeline sintético se sobrescribe: todo artefacto lleva el
  prefijo ``dt1_``.

Uso::

    python src/train_cardio_real.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# =======================================================
# Configuración
# =======================================================
SEED: int = 42
TEST_SIZE: float = 0.20
N_BOOTSTRAP: int = 1000
N_CAL_BINS: int = 10

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/real/cardio/cardio_train.csv"
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "dt1_manifest.json"

# Cortes antropométricos. Criterio: imposibilidad fisiológica en adultos de 30-65
# años, no recorte de outliers. Documentados también en el manifiesto.
HEIGHT_CM_RANGE: tuple[float, float] = (120.0, 220.0)
WEIGHT_KG_RANGE: tuple[float, float] = (30.0, 200.0)
BMI_RANGE: tuple[float, float] = (12.0, 70.0)

# Rangos de presión plausibles. NO se usan para limpiar (fuera del alcance acordado);
# solo para el análisis de sensibilidad sobre test.
AP_HI_PLAUSIBLE: tuple[float, float] = (70.0, 250.0)
AP_LO_PLAUSIBLE: tuple[float, float] = (40.0, 200.0)

HTA_SBP_THRESHOLD: int = 140
HTA_DBP_THRESHOLD: int = 90

PRESSURE_COLUMNS: frozenset[str] = frozenset({"ap_hi", "ap_lo"})
FORBIDDEN_B1: frozenset[str] = PRESSURE_COLUMNS | {"cardio", "id", "hta", "hta_gt"}

# Orden de features: contrato con el manifiesto y, más adelante, con la API.
FEATURES_BASE: list[str] = [
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
FEATURES_A_CON_PA: list[str] = FEATURES_BASE + ["ap_hi", "ap_lo"]
FEATURES_A_SIN_PA: list[str] = list(FEATURES_BASE)
FEATURES_B1: list[str] = list(FEATURES_BASE)

# Linaje de cada feature: columnas crudas de las que se calcula. Es lo que permite
# verificar estructuralmente que ninguna feature de B1 deriva de la presión.
FEATURE_LINEAGE: dict[str, set[str]] = {
    "age_years": {"age"},
    "gender": {"gender"},
    "height": {"height"},
    "weight": {"weight"},
    "bmi": {"weight", "height"},
    "cholesterol": {"cholesterol"},
    "gluc": {"gluc"},
    "smoke": {"smoke"},
    "alco": {"alco"},
    "active": {"active"},
    "ap_hi": {"ap_hi"},
    "ap_lo": {"ap_lo"},
}


# =======================================================
# Datos
# =======================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_clean() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carga el CSV y aplica la limpieza de etiqueta, reportando cada paso."""
    raw = pd.read_csv(DATA_PATH, sep=";")
    report: dict[str, Any] = {"filas_crudas": len(raw), "pasos": []}

    def step(name: str, criterio: str, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
        report["pasos"].append(
            {
                "paso": name,
                "criterio": criterio,
                "descartadas": len(df_before) - len(df_after),
                "quedan": len(df_after),
            }
        )

    feature_cols = [c for c in raw.columns if c != "id"]
    df = raw.drop_duplicates(subset=feature_cols, keep="first")
    step(
        "duplicados_exactos",
        "duplicado exacto en todas las columnas salvo id; se conserva la primera aparición",
        raw,
        df,
    )

    before = df
    df = df[df["ap_lo"] < df["ap_hi"]]
    step("presion_invertida", "ap_lo >= ap_hi -> descartar (no se intercambian columnas)", before, df)

    before = df
    bmi = df["weight"] / (df["height"] / 100.0) ** 2
    mask = (
        df["height"].between(*HEIGHT_CM_RANGE)
        & df["weight"].between(*WEIGHT_KG_RANGE)
        & bmi.between(*BMI_RANGE)
    )
    df = df[mask]
    step(
        "antropometria_imposible",
        f"height fuera de {list(HEIGHT_CM_RANGE)} cm, weight fuera de {list(WEIGHT_KG_RANGE)} kg "
        f"o IMC fuera de {list(BMI_RANGE)} kg/m2 (límites inclusivos)",
        before,
        df,
    )

    df = df.copy()
    df["age_years"] = df["age"] / 365.25
    df["bmi"] = df["weight"] / (df["height"] / 100.0) ** 2
    df["hta"] = ((df["ap_hi"] >= HTA_SBP_THRESHOLD) | (df["ap_lo"] >= HTA_DBP_THRESHOLD)).astype(int)
    df["hta_gt"] = ((df["ap_hi"] > HTA_SBP_THRESHOLD) | (df["ap_lo"] > HTA_DBP_THRESHOLD)).astype(int)
    df["pa_plausible"] = df["ap_hi"].between(*AP_HI_PLAUSIBLE) & df["ap_lo"].between(*AP_LO_PLAUSIBLE)

    report["filas_finales"] = len(df)
    report["fraccion_conservada"] = len(df) / len(raw)
    report["filas_presion_implausible_no_limpiadas"] = int((~df["pa_plausible"]).sum())
    report["prevalencia_cardio"] = float(df["cardio"].mean())
    report["prevalencia_hta_ge"] = float(df["hta"].mean())
    report["prevalencia_hta_gt"] = float(df["hta_gt"].mean())
    report["fraccion_etiquetas_hta_que_cambian_ge_vs_gt"] = float((df["hta"] != df["hta_gt"]).mean())
    return df.reset_index(drop=True), report


def assert_no_pressure_lineage(features: list[str]) -> None:
    """Falla si alguna feature es, o deriva de, presión arterial o del target."""
    for feat in features:
        if feat in FORBIDDEN_B1:
            raise AssertionError(f"Feature prohibida en B1: {feat}")
        lineage = FEATURE_LINEAGE[feat]
        if lineage & FORBIDDEN_B1:
            raise AssertionError(f"{feat} deriva de columnas prohibidas: {lineage & FORBIDDEN_B1}")


# =======================================================
# Modelos
# =======================================================
def make_models() -> dict[str, ClassifierMixin]:
    """Los mismos 5 algoritmos e hiperparámetros del pipeline sintético.

    SVM: ``SVC(probability=True)`` está deprecado desde scikit-learn 1.9; se usa el
    reemplazo oficial, Platt scaling con 5 folds y ``ensemble=False``.
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
        "SVM_RBF": CalibratedClassifierCV(
            SVC(kernel="rbf", C=2, random_state=SEED),
            method="sigmoid",
            cv=5,
            ensemble=False,
        ),
        "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


# =======================================================
# Métricas
# =======================================================
def compute_metrics(y: np.ndarray, proba: np.ndarray, prevalence_train: float) -> dict[str, Any]:
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    prob_true, prob_pred = calibration_curve(y, proba, n_bins=N_CAL_BINS, strategy="uniform")
    # Misma asignación de bins que calibration_curve(strategy="uniform").
    bin_ids = np.searchsorted(np.linspace(0.0, 1.0, N_CAL_BINS + 1)[1:-1], proba)
    counts = np.bincount(bin_ids, minlength=N_CAL_BINS)
    # calibration_curve omite bins vacíos; los conteos se alinean con los no vacíos.
    nonempty = counts[counts > 0]
    ece = float(np.sum(nonempty / len(y) * np.abs(prob_true - prob_pred)))

    return {
        "n_test": len(y),
        "prevalencia_test": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "pr_auc_baseline_prevalencia": float(y.mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "accuracy_baseline_mayoritaria": float(max(y.mean(), 1 - y.mean())),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "umbral_decision": 0.5,
        "brier": float(brier_score_loss(y, proba)),
        # Brier de predecir siempre la prevalencia de train: referencia sin información.
        "brier_baseline_prevalencia": float(np.mean((prevalence_train - y) ** 2)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "ece_uniform_10": ece,
        "calibracion": {
            "estrategia": "uniform",
            "n_bins": N_CAL_BINS,
            "prob_media_predicha": [float(v) for v in prob_pred],
            "fraccion_positivos_observada": [float(v) for v in prob_true],
            "conteo_por_bin": [int(v) for v in nonempty],
        },
    }


def bootstrap_auc_ci(y: np.ndarray, proba: np.ndarray, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], proba[idx]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def paired_bootstrap_auc_delta(
    y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, seed: int = SEED
) -> dict[str, float]:
    """AUC(a) - AUC(b) sobre las mismas filas remuestreadas."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        deltas.append(roc_auc_score(y[idx], proba_a[idx]) - roc_auc_score(y[idx], proba_b[idx]))
    return {
        "delta_auc": float(roc_auc_score(y, proba_a) - roc_auc_score(y, proba_b)),
        "ic95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
    }


# =======================================================
# Experimento
# =======================================================
@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_split(df: pd.DataFrame, strat_col: str) -> Split:
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, stratify=df[strat_col].to_numpy()
    )
    return Split(np.sort(train_idx), np.sort(test_idx))


def run_experiment(
    df: pd.DataFrame,
    name: str,
    features: list[str],
    target: str,
    split: Split,
    save_artifacts: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    print(f"\n{'=' * 70}\n{name}  |  target={target}  |  {len(features)} features\n{'=' * 70}")
    train, test = df.iloc[split.train_idx], df.iloc[split.test_idx]
    assert not set(train["id"]) & set(test["id"]), "Solapamiento de id entre train y test"

    X_train_raw = train[features].to_numpy(dtype=float)
    X_test_raw = test[features].to_numpy(dtype=float)
    y_train = train[target].to_numpy()
    y_test = test[target].to_numpy()

    scaler = StandardScaler().fit(X_train_raw)  # solo train
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    prevalence_train = float(y_train.mean())

    results: dict[str, Any] = {}
    probas: dict[str, np.ndarray] = {}
    fitted: dict[str, ClassifierMixin] = {}
    for algo, template in make_models().items():
        model = clone(template)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0

        proba_test = model.predict_proba(X_test)[:, 1]
        proba_train = model.predict_proba(X_train)[:, 1]
        m = compute_metrics(y_test, proba_test, prevalence_train)
        m["roc_auc_train"] = float(roc_auc_score(y_train, proba_train))
        m["fit_segundos"] = round(fit_s, 1)
        results[algo] = m
        probas[algo] = proba_test
        fitted[algo] = model
        print(
            f"  {algo:<18} AUC={m['roc_auc']:.4f} (train {m['roc_auc_train']:.4f})  "
            f"PR-AUC={m['pr_auc']:.4f}  mF1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  "
            f"Brier={m['brier']:.4f}  ECE={m['ece_uniform_10']:.4f}  [{fit_s:.0f}s]"
        )

    winner = max(results, key=lambda k: results[k]["macro_f1"])
    results[winner]["roc_auc_ic95_bootstrap"] = bootstrap_auc_ci(y_test, probas[winner])

    # Sensibilidad: métricas del ganador excluyendo de test las filas con presión
    # implausible que la limpieza acordada no cubre. Sin reentrenar.
    plausible = test["pa_plausible"].to_numpy()
    sens = compute_metrics(y_test[plausible], probas[winner][plausible], prevalence_train)

    exp: dict[str, Any] = {
        "nombre": name,
        "target": target,
        "features_orden": features,
        "n_features": len(features),
        "n_train": len(train),
        "n_test": len(test),
        "prevalencia_train": prevalence_train,
        "criterio_seleccion": "macro_f1 en test (umbral 0.5)",
        "ganador": winner,
        "modelos": results,
        "sensibilidad_test_sin_presion_implausible": {
            "filas_excluidas_de_test": int((~plausible).sum()),
            "roc_auc": sens["roc_auc"],
            "macro_f1": sens["macro_f1"],
            "brier": sens["brier"],
        },
        "scaler": {
            "tipo": "StandardScaler",
            "ajustado_sobre": "train",
            "mean_": [float(v) for v in scaler.mean_],
            "scale_": [float(v) for v in scaler.scale_],
        },
    }
    print(f"  -> ganador (macro-F1): {winner}  AUC IC95 {results[winner]['roc_auc_ic95_bootstrap']}")

    if save_artifacts:
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"dt1_{name}__modelo.pkl"
        scaler_path = MODELS_DIR / f"dt1_{name}__scaler.pkl"
        joblib.dump(fitted[winner], model_path)
        joblib.dump(scaler, scaler_path)
        exp["artefactos"] = {
            "modelo": {"ruta": str(model_path.relative_to(ROOT)), "sha256": sha256_file(model_path)},
            "scaler": {"ruta": str(scaler_path.relative_to(ROOT)), "sha256": sha256_file(scaler_path)},
        }
    return exp, probas | {"_y_test": y_test}


def univariate_audit(df: pd.DataFrame, split: Split, features: list[str], target: str) -> dict[str, Any]:
    """AUC univariada (orientada, >=0.5) y correlación con la presión, solo en train.

    Una feature derivada de la presión se delataría con AUC univariada muy alta
    o |corr| cercana a 1 con ap_hi/ap_lo.
    """
    train = df.iloc[split.train_idx]
    out: dict[str, Any] = {}
    for f in features:
        auc = roc_auc_score(train[target], train[f])
        out[f] = {
            "auc_univariada": float(max(auc, 1 - auc)),
            "corr_ap_hi": float(train[f].corr(train["ap_hi"], method="spearman")),
            "corr_ap_lo": float(train[f].corr(train["ap_lo"], method="spearman")),
        }
    return out


# =======================================================
# Principal
# =======================================================
def main() -> None:
    t_start = time.perf_counter()
    df, cleaning = load_and_clean()
    print(json.dumps(cleaning, indent=2, ensure_ascii=False))

    assert_no_pressure_lineage(FEATURES_B1)

    # ---- A′: riesgo cardiovascular, misma partición para ambas variantes ----
    split_a = make_split(df, "cardio")
    a_con, p_con = run_experiment(df, "riesgo_cv_con_pa", FEATURES_A_CON_PA, "cardio", split_a, True)
    a_sin, p_sin = run_experiment(df, "riesgo_cv_sin_pa", FEATURES_A_SIN_PA, "cardio", split_a, True)
    ablation = {
        "comparacion": "ganador con PA vs ganador sin PA (misma partición, bootstrap pareado)",
        "ganador_con_pa": a_con["ganador"],
        "ganador_sin_pa": a_sin["ganador"],
        **paired_bootstrap_auc_delta(p_con["_y_test"], p_con[a_con["ganador"]], p_sin[a_sin["ganador"]]),
        "mismo_algoritmo_por_algoritmo": {
            algo: a_con["modelos"][algo]["roc_auc"] - a_sin["modelos"][algo]["roc_auc"]
            for algo in a_con["modelos"]
        },
    }

    # ---- B1: hipertensión sin presión en las features ----
    split_b = make_split(df, "hta")
    b1, _ = run_experiment(df, "hta_b1", FEATURES_B1, "hta", split_b, True)
    b1["umbral"] = {
        "definicion": f"hta = (ap_hi >= {HTA_SBP_THRESHOLD}) | (ap_lo >= {HTA_DBP_THRESHOLD})",
        "fuente": "JNC7 / ESC, fijado a priori",
    }
    b1["auditoria_univariada_train"] = univariate_audit(df, split_b, FEATURES_B1, "hta")

    # Robustez: misma partición que B1 (estratificada por '>='), solo cambia la etiqueta.
    b1_gt, _ = run_experiment(df, "hta_b1_robustez_gt", FEATURES_B1, "hta_gt", split_b, False)
    b1_gt["nota"] = (
        "Reporte de robustez, no re-selección de umbral. Misma partición que hta_b1; "
        "sin artefactos guardados."
    )
    b1_gt["umbral"] = {
        "definicion": f"hta_gt = (ap_hi > {HTA_SBP_THRESHOLD}) | (ap_lo > {HTA_DBP_THRESHOLD})"
    }

    manifest = {
        "hito": "DT-1",
        "generado_por": "src/train_cardio_real.py",
        "semilla": SEED,
        "test_size": TEST_SIZE,
        "entorno": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "dataset": {
            "ruta": str(DATA_PATH.relative_to(ROOT)),
            "sha256": sha256_file(DATA_PATH),
            "separador": ";",
            "codificacion_gender": "1/2 crudo; 2 = hombre (inferido: talla media 169.9 vs 161.4 cm)",
            "age_years": "age (días) / 365.25",
            "bmi": "weight / (height/100)^2",
        },
        "limpieza": cleaning,
        "experimentos": {
            "A_prima_principal": {"con_pa": a_con, "sin_pa": a_sin, "ablacion": ablation},
            "B1_experimento": {"principal": b1, "robustez_umbral_estricto": b1_gt},
        },
        "duracion_segundos": round(time.perf_counter() - t_start, 1),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nManifiesto: {MANIFEST_PATH.relative_to(ROOT)}  ({manifest['duracion_segundos']}s)")


if __name__ == "__main__":
    sys.exit(main())
