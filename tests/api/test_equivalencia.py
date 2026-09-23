"""La matriz que arma la API desde unidades humanas es la del entrenamiento, bit a bit."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_equivalencia_features import N_FILAS

pytestmark = pytest.mark.requires_data


def test_api_reproduce_la_matriz_de_entrenamiento(entrenamiento):
    """Unidades humanas → matriz de la API == matriz de entrenamiento, bit a bit.

    Se recorren todas las filas del CSV limpio que caen en el dominio de la API
    (presión plausible); la edad entra en años float, como la vio el modelo.
    """
    from api.schemas import EntradaHTASinPA, EntradaRiesgoCV
    from api.servicio import matriz

    t, df = entrenamiento
    df = df[df["pa_plausible"]]
    campos = ["age_years", "gender", "height", "weight", "cholesterol", "gluc", "smoke", "alco", "active"]
    registros = df[campos + ["ap_hi", "ap_lo"]].to_dict("records")

    con_pa = [EntradaRiesgoCV.model_validate(r) for r in registros]
    sin_pa = [EntradaHTASinPA.model_validate({k: r[k] for k in campos}) for r in registros]

    esperado_a = df[t.FEATURES_A_CON_PA].to_numpy(dtype=float)
    esperado_b1 = df[t.FEATURES_B1].to_numpy(dtype=float)
    assert np.array_equal(matriz(con_pa, t.FEATURES_A_CON_PA), esperado_a)
    assert np.array_equal(matriz(sin_pa, t.FEATURES_B1), esperado_b1)
    assert len(con_pa) == N_FILAS - 92  # filas_presion_implausible_no_limpiadas del manifiesto
