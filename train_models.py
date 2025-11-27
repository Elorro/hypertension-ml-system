from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - xgboost must be installed separately
    raise ImportError("xgboost no está instalado. Instálalo antes de ejecutar este script.") from exc

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - tensorflow must be installed separately
    raise ImportError("tensorflow no está instalado. Instálalo antes de ejecutar este script.") from exc

from data_pipeline import (
    FEATURE_COLUMNS,
    LABEL_MAP,
    TARGET_COLUMN,
    generate_synthetic_hypertension_data,
    save_dataset,
)

RANDOM_STATE = 42
DATA_PATH = Path("data/hypertension_synthetic.csv")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

def ensure_dataset(path: Path = DATA_PATH, n_samples: int = 50000) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    df = generate_synthetic_hypertension_data(n_samples=n_samples, seed=RANDOM_STATE)
    save_dataset(df, path)
    return df


def get_preprocessor() -> ColumnTransformer:
    numeric_features = FEATURE_COLUMNS
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    models = {
        "log_reg": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", LogisticRegression(max_iter=1000, multi_class="multinomial", n_jobs=-1)),
            ]
        ),
        "svm_linear": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    CalibratedClassifierCV(
                        base_estimator=LinearSVC(random_state=RANDOM_STATE),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=None,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.1,
                        max_depth=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        num_class=len(LABEL_MAP),
                    ),
                ),
            ]
        ),
    }
    return models


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def get_probabilities(model, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return softmax(scores)
    raise ValueError("El modelo no soporta predict_proba ni decision_function.")


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
    y_pred = model.predict(X_test)
    y_proba = get_probabilities(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(y_test, y_proba, multi_class="ovr")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def train_ann(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray) -> Tuple[tf.keras.Model, Dict]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(LABEL_MAP), activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.15,
        epochs=40,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0,
    )

    y_proba = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc_ovr": float(roc_auc_score(y_test, y_proba, multi_class="ovr")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "scaler": scaler,  # lo devolvemos para exportar si se necesita
    }
    return model, metrics


def save_metrics_report(metrics: Dict[str, Dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Métricas guardadas en {metrics_path.resolve()}")


def main():
    df = ensure_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = get_preprocessor()
    models = build_models(preprocessor)

    metrics_report = {}
    best_name = None
    best_f1 = -np.inf
    best_model = None

    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics_report[name] = metrics
        print(f"{name} -> accuracy={metrics['accuracy']:.3f} | f1_macro={metrics['f1_macro']:.3f} | roc_auc_ovr={metrics['roc_auc_ovr']:.3f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_name = name
            best_model = model

    # Entrenamos ANN (no entra al top de producción por simplicidad de despliegue, pero se reporta)
    print("Entrenando red neuronal (TensorFlow)...")
    ann_model, ann_metrics = train_ann(X_train, y_train, X_test, y_test)
    metrics_report["ann_tensorflow"] = {
        k: v for k, v in ann_metrics.items() if k != "scaler"
    }

    save_metrics_report(metrics_report)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(
        {"model": best_model, "label_map": LABEL_MAP, "features": FEATURE_COLUMNS},
        best_path,
    )
    print(f"Mejor modelo ({best_name}) guardado en {best_path.resolve()}")

    # Guardamos también la red neuronal y su scaler para referencia
    ann_path = MODEL_DIR / "ann_model.keras"
    ann_model.save(ann_path)
    joblib.dump(ann_metrics["scaler"], MODEL_DIR / "ann_scaler.joblib")
    print(f"Red neuronal guardada en {ann_path.resolve()}")


if __name__ == "__main__":
    main()
