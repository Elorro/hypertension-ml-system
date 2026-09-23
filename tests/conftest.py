"""Fixtures comunes y salto explícito de los tests que necesitan archivos fuera de git.

* ``requires_artifacts`` — necesita ``models/dt1_*.pkl`` (no versionados).
* ``requires_data`` — necesita ``data/real/cardio/cardio_train.csv`` (no redistribuible).

Sin esos archivos el test se salta con el motivo visible en la salida (``-rs``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "models" / "dt1_manifest.json"
DATA_PATH = ROOT / "data" / "real" / "cardio" / "cardio_train.csv"
ARTEFACTOS = sorted((ROOT / "models").glob("dt1_*.pkl"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    motivos = {
        "requires_artifacts": (
            len(ARTEFACTOS) < 6,
            "faltan models/dt1_*.pkl (no versionados; `make train-real`)",
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
