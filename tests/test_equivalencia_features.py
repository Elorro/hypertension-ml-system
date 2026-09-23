"""Equivalencia de features: el refactor a src/cardio_features.py no cambió ni un bit.

Los hashes de referencia se calcularon con ``src/train_cardio_real.py`` de c3814b2
(antes de extraer el módulo compartido), con la misma función ``_h``. Si alguno
cambia, las matrices que ve el modelo ya no son las de la corrida de DT-1.

El paso siguiente (API → misma matriz) está en test_equivalencia_api.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tests.conftest import DATA_PATH

pytestmark = pytest.mark.requires_data

N_FILAS = 68678
COLUMNAS = [
    "id", "age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke",
    "alco", "active", "cardio", "age_years", "bmi", "hta", "hta_gt", "pa_plausible",
]  # fmt: skip
DTYPES = ["int64"] * 4 + ["float64"] + ["int64"] * 8 + ["float64", "float64", "int64", "int64", "bool"]
HASH_DF = "77446dd34c403a331e96b7255391350d9d9bc757fae75e28cb3cc98339d01905"

# Referencia de c3814b2. A′ con y sin PA comparten partición (estratificada por cardio).
SPLIT_CARDIO = {
    "train_idx": "76fb8ffd6f1134b20ebfc99fca7e0d4723fbac5c4dfa78ad37870caee6b4542a",
    "test_idx": "dc8c2155ab32442dfba55132bcdf4a8a17b55eeaa8c9d766b28233de1b024e7f",
    "y_train": "4ebb7b219b65e472184b19eb2213d829fcb758011bffd51486ec1ab82e0a9318",
    "y_test": "6be5f0c8fa05f94742c29f6db737ff10e259988faa1b2cc548a2e502879ef916",
}
REFERENCIA: dict[str, dict[str, str]] = {
    "riesgo_cv_con_pa": SPLIT_CARDIO
    | {
        "X_train": "2362703e76c7a15f9a05d0e1cdc405ecb1dce9d5b6936bdcb1cabb864f06babe",
        "X_test": "b7c043ed3324473cf08ae75ef6bb5f0ce997bbb9d09a5ef1051ce1f30951e482",
    },
    "riesgo_cv_sin_pa": SPLIT_CARDIO
    | {
        "X_train": "ebac2ac7cab2310f0ca2cb440f8fba6dd9bc625f87c2e4e11799346d456d96ed",
        "X_test": "a61f4c947f5e6308c38889c5e7230b3beb1d068c6a1e1b89a0986f89e8b959e4",
    },
    "hta_b1": {
        "train_idx": "18afc3072cbbe96012bec8676517d59035916fbec58c0c78e325b43392be5d37",
        "test_idx": "1930b665c90eb30a48b77eca7f1b1a55437f761454ddd97e25876e791449c276",
        "X_train": "6d7618b89812596c0502b0e42665ecd543014b1bd49f94e3e39cf058df4d73b8",
        "X_test": "0f86ab544bae474c532f277b81f0d48721188f3175005ac3bb2bfcc156411967",
        "y_train": "84a3f953be5dfb485fd3394e7f0df2010540d8852c64cf237aeba24f98474a0d",
        "y_test": "e5a49f976a369d3902e0f835d4806e5c418eba47f119b74104ce59fdf5042d87",
    },
}


def _h(a: np.ndarray) -> str:
    """sha256 de dtype + forma + bytes: detecta cambios de valor, orden, forma o tipo."""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def test_dataframe_limpio_identico(entrenamiento):
    _, df = entrenamiento
    assert len(df) == N_FILAS
    assert list(df.columns) == COLUMNAS
    assert [str(d) for d in df.dtypes] == DTYPES
    assert _h(df.to_numpy(dtype=float)) == HASH_DF


@pytest.mark.parametrize("nombre", list(REFERENCIA))
def test_matrices_por_experimento_identicas(entrenamiento, nombre):
    t, df = entrenamiento
    features, target = {
        "riesgo_cv_con_pa": (t.FEATURES_A_CON_PA, "cardio"),
        "riesgo_cv_sin_pa": (t.FEATURES_A_SIN_PA, "cardio"),
        "hta_b1": (t.FEATURES_B1, "hta"),
    }[nombre]
    split = t.make_split(df, target)
    train, test = df.iloc[split.train_idx], df.iloc[split.test_idx]
    obtenido = {
        "train_idx": _h(split.train_idx),
        "test_idx": _h(split.test_idx),
        "X_train": _h(train[features].to_numpy(dtype=float)),
        "X_test": _h(test[features].to_numpy(dtype=float)),
        "y_train": _h(train[target].to_numpy()),
        "y_test": _h(test[target].to_numpy()),
    }
    assert obtenido == REFERENCIA[nombre]


def test_csv_presente_es_el_del_manifiesto(manifiesto):
    from src.artefactos import sha256_file

    assert sha256_file(DATA_PATH) == manifiesto["dataset"]["sha256"]
