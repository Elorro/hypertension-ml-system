"""DT-4: la corrida registrada es reproducible y coherente con su protocolo.

* Sin datos (corren en CI): la regla de 1 EE, reaplicada sobre los resultados por
  fold guardados, da el mismo seleccionado; el protocolo no cambió después de la
  corrida (sha256 igual al registrado).
* Con el CSV (``requires_data``): los folds de CV que genera hoy el código son los de
  la corrida (hashes congelados de ad518f2).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from src import train_cardio_dt4 as dt4
from src.artefactos import sha256_file
from tests.conftest import MANIFEST_DT4_PATH, ROOT

# sha256 encadenado de (train_idx, val_idx) de los 5 folds, por target de estratificación.
HASH_FOLDS = {
    "cardio": "093d4ea660bec684482a666b2d15a2ed9f43d85a17c388964d785db15f4c66e5",
    "hta": "091da49abd9d796b79a6d76443d3c39d05fe3303e93b230f9cc1c8d27ade31f1",
}


@pytest.fixture(scope="module")
def dt4_manifiesto() -> dict:
    return json.loads(MANIFEST_DT4_PATH.read_text())


def _bloques(m: dict) -> dict:
    e = m["experimentos"]
    return {
        "riesgo_cv_con_pa": e["A_prima_principal"]["con_pa"],
        "riesgo_cv_sin_pa": e["A_prima_principal"]["sin_pa"],
        "hta_b1": e["B1_experimento"]["principal"],
    }


@pytest.mark.parametrize("nombre", ["riesgo_cv_con_pa", "riesgo_cv_sin_pa", "hta_b1"])
def test_regla_un_ee_reproduce_la_seleccion_registrada(dt4_manifiesto, nombre):
    b = _bloques(dt4_manifiesto)[nombre]
    candidatos = b["cv"]["candidatos"]
    assert [c["id"] for c in candidatos] == dt4_manifiesto["orden_simplicidad"]
    sel = dt4.regla_un_ee(candidatos)
    assert sel["seleccionado"] == b["seleccion"]["seleccionado"] == b["candidato_seleccionado"]
    assert sel["elegibles"] == b["seleccion"]["elegibles"]


def test_protocolo_intacto_desde_la_corrida(dt4_manifiesto):
    assert sha256_file(ROOT / dt4_manifiesto["protocolo"]["ruta"]) == dt4_manifiesto["protocolo"]["sha256"]
    assert dt4_manifiesto["desviaciones"] == []
    assert dt4_manifiesto["codigo"]["arbol_limpio"] is True


@pytest.mark.requires_data
@pytest.mark.parametrize(("target", "nombre"), [("cardio", "riesgo_cv_con_pa"), ("hta", "hta_b1")])
def test_folds_de_cv_identicos_a_la_corrida(entrenamiento, dt4_manifiesto, target, nombre):
    t, df = entrenamiento
    split = t.make_split(df, target)
    folds = dt4.folds_estratificados(df.iloc[split.train_idx][target].to_numpy())
    h = hashlib.sha256()
    for tr, va in folds:
        h.update(dt4.hash_indices(tr).encode())
        h.update(dt4.hash_indices(va).encode())
    assert h.hexdigest() == HASH_FOLDS[target]
    assert [len(va) for _, va in folds] == _bloques(dt4_manifiesto)[nombre]["cv"]["tamanos_validacion"]
