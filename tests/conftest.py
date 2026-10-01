"""Fixtures comunes y salto explícito de los tests que necesitan archivos fuera de git.

* ``requires_artifacts`` — necesita ``models/dt4_*.pkl``, los servidos (no versionados).
* ``requires_data`` — necesita ``data/real/cardio/cardio_train.csv`` (no redistribuible).

Sin esos archivos el test se salta con el motivo visible en la salida (``-rs``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "dt1_manifest.json"  # registro de DT-1
MANIFEST_DT4_PATH = ROOT / "models" / "dt4_manifest.json"  # el servido
DATA_PATH = ROOT / "data" / "real" / "cardio" / "cardio_train.csv"
# Los que carga la API y publica el release: modelo y scaler de A′ con PA y de B1.
ARTEFACTOS = sorted(
    p for exp in ("riesgo_cv_con_pa", "hta_b1") for p in (ROOT / "models").glob(f"dt4_{exp}__*.pkl")
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    motivos = {
        "requires_artifacts": (
            len(ARTEFACTOS) < 4,
            "faltan models/dt4_*.pkl (no versionados; `make train-dt4`)",
        ),
        "requires_data": (
            not DATA_PATH.exists(),
            "falta data/real/cardio/cardio_train.csv (no redistribuible)",
        ),
    }
    for item in items:
        for marca, (falta, motivo) in motivos.items():
            if falta and marca in item.keywords:
                item.add_marker(pytest.mark.skip(reason=motivo))


@pytest.fixture(scope="session")
def manifiesto() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="session")
def entrenamiento():
    """Módulo de entrenamiento y CSV limpio (requiere el CSV; arrastra sklearn/xgboost)."""
    t = pytest.importorskip("src.train_cardio_real")
    df, _ = t.load_and_clean()
    return t, df
