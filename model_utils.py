from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from data_pipeline import FEATURE_COLUMNS, LABEL_MAP


def load_model_bundle(model_path: Path = Path("models/best_model.joblib")) -> Dict:
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}. Entrena primero con train_models.py")
    bundle = joblib.load(model_path)
    bundle.setdefault("features", FEATURE_COLUMNS)
    bundle.setdefault("label_map", LABEL_MAP)
    return bundle


def predict_single(model_bundle: Dict, payload: Dict) -> Dict:
    model = model_bundle["model"]
    features: List[str] = model_bundle.get("features", FEATURE_COLUMNS)
    label_map: Dict[int, str] = model_bundle.get("label_map", LABEL_MAP)
    df = pd.DataFrame([payload], columns=features)
    proba = model.predict_proba(df)[0]
    pred_idx = int(np.argmax(proba))
    return {
        "prediccion": pred_idx,
        "clase": label_map[pred_idx],
        "probabilidades": {label_map[i]: float(prob) for i, prob in enumerate(proba)},
    }
