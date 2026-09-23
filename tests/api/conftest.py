"""Dobles del modelo y cliente de la API para los tests de contrato."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest


# =======================================================
# Doble del modelo para los tests de contrato
# =======================================================
@dataclass
class PredictorDoble:
    """Devuelve una probabilidad fija y registra cada matriz recibida."""

    probabilidad: float = 0.73
    llamadas: list[np.ndarray] = field(default_factory=list)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.llamadas.append(X.copy())
        return np.full(X.shape[0], self.probabilidad)


@dataclass
class Dobles:
    riesgo_cv: PredictorDoble
    hta_b1: PredictorDoble


@pytest.fixture
def dobles() -> Dobles:
    return Dobles(riesgo_cv=PredictorDoble(0.73), hta_b1=PredictorDoble(0.41))


@pytest.fixture
def client(manifiesto: dict[str, Any], dobles: Dobles):
    """API sin artefactos: el registro real se sustituye por dobles (override de dependencia).

    Los metadatos (prevalencias, métricas, sha256) vienen del manifiesto versionado.
    """
    from fastapi.testclient import TestClient

    from api.main import create_app, get_registro
    from api.servicio import HTA_B1, RIESGO_CV, ModeloServido, Registro
    from src.artefactos import experimentos

    bloques = experimentos(manifiesto)

    def servido(nombre: str, predictor: PredictorDoble) -> ModeloServido:
        b = bloques[nombre]
        return ModeloServido(
            experimento=nombre,
            algoritmo=b["ganador"],
            sha256=b["artefactos"]["modelo"]["sha256"],
            features=list(b["features_orden"]),
            prevalencia_base=float(b["prevalencia_train"]),
            bloque=b,
            predictor=predictor,
        )

    registro = Registro(
        modelos={RIESGO_CV: servido(RIESGO_CV, dobles.riesgo_cv), HTA_B1: servido(HTA_B1, dobles.hta_b1)},
        manifiesto=manifiesto,
    )
    app = create_app(cargar_artefactos=False)
    app.dependency_overrides[get_registro] = lambda: registro
    with TestClient(app) as c:
        yield c


# Perfiles en unidades humanas, dentro del dominio de entrenamiento.
BASE_OK: dict[str, Any] = {
    "age_years": 52.0,
    "gender": 1,
    "height": 165.0,
    "weight": 72.0,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}
PA_OK: dict[str, Any] = {"ap_hi": 130.0, "ap_lo": 85.0}
