"""Registro de modelos servidos y construcción de la matriz de features.

La API sirve dos experimentos, los de DT-4 por defecto (``models/dt4_manifest.json``;
el manifiesto de DT-1 sigue siendo servible por ``MANIFEST_PATH``) y ninguno más:

* ``riesgo_cv_con_pa`` (A′) — riesgo cardiovascular, con presión arterial.
* ``hta_b1`` (B1, experimental) — hipertensión sin presión arterial.

El modelo se abstrae como ``Predictor`` (matriz → probabilidad de la clase
positiva) para que los tests de contrato sustituyan el ``.pkl`` por un doble.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from api.schemas import EntradaBase, InfoModelo
from src.artefactos import ModeloCargado, cargar_experimento, exigir_versiones, experimentos

Predictor = Callable[[np.ndarray], np.ndarray]

RIESGO_CV: str = "riesgo_cv_con_pa"
HTA_B1: str = "hta_b1"
EXPERIMENTOS_SERVIDOS: tuple[str, str] = (RIESGO_CV, HTA_B1)


@dataclass(frozen=True)
class ModeloServido:
    experimento: str
    algoritmo: str
    sha256: str
    features: list[str]
    prevalencia_base: float
    bloque: dict[str, Any]  # bloque del manifiesto: métricas y metadatos
    predictor: Predictor = field(repr=False)
    estimador: Any = field(default=None, repr=False)  # objeto cargado; None en los dobles de test

    def info(self) -> InfoModelo:
        return InfoModelo(experimento=self.experimento, algoritmo=self.algoritmo, sha256=self.sha256)

    def probabilidades(self, entradas: Sequence[EntradaBase]) -> np.ndarray:
        return self.predictor(matriz(entradas, self.features))


@dataclass(frozen=True)
class Registro:
    modelos: dict[str, ModeloServido]
    manifiesto: dict[str, Any]


def matriz(entradas: Sequence[EntradaBase], features: list[str]) -> np.ndarray:
    """Matriz float64 en el orden exacto de ``features_orden`` del manifiesto.

    ``age_years`` pasa tal cual (el modelo vio años float64, age/365.25); ``bmi``
    se calcula en el servidor con la fórmula de ``src/cardio_features.py``.
    """
    filas = []
    for e in entradas:
        valores = e.model_dump() | {"bmi": e.imc}
        filas.append([float(valores[f]) for f in features])
    return np.array(filas, dtype=np.float64).reshape(len(filas), len(features))


def servido_desde(cargado: ModeloCargado) -> ModeloServido:
    modelo, scaler = cargado.modelo, cargado.scaler
    # El RF se serializó con n_jobs=-1: para 1-1000 filas, lanzar un hilo por núcleo
    # cuesta más que predecir. Cambia el objeto en memoria, no el .pkl ni su sha256.
    if "n_jobs" in modelo.get_params():
        modelo.set_params(n_jobs=1)

    def predictor(X: np.ndarray) -> np.ndarray:
        return modelo.predict_proba(scaler.transform(X))[:, 1]

    return ModeloServido(
        experimento=cargado.experimento,
        algoritmo=cargado.algoritmo,
        sha256=cargado.sha256_modelo,
        features=cargado.features,
        prevalencia_base=float(cargado.bloque["prevalencia_train"]),
        bloque=cargado.bloque,
        predictor=predictor,
        estimador=modelo,
    )


def cargar_registro(manifest_path: Path, model_dir: Path) -> Registro:
    """Verifica versiones (perfil servicio) y carga A′ y B1. Levanta ArtefactoInvalido."""
    manifiesto = json.loads(manifest_path.read_text())
    bloques = experimentos(manifiesto)
    servidos = {nombre: bloques[nombre] for nombre in EXPERIMENTOS_SERVIDOS}
    exigir_versiones(manifiesto, "servicio", [b["ganador"] for b in servidos.values()])
    modelos = {nombre: servido_desde(cargar_experimento(b, model_dir)) for nombre, b in servidos.items()}
    return Registro(modelos=modelos, manifiesto=manifiesto)


def entorno_en_ejecucion() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit_learn": md.version("scikit-learn"),
        "numpy": md.version("numpy"),
        "joblib": md.version("joblib"),
    }


# =======================================================
# Metadatos para /v1/modelos
# =======================================================
def _metricas(servido: ModeloServido) -> dict[str, Any]:
    """Métricas en test del modelo servido: bloque ``test`` (manifiesto v2, DT-4) o las del
    ganador en ``modelos`` (manifiesto de DT-1)."""
    b = servido.bloque
    return b["test"] if "test" in b else b["modelos"][servido.algoritmo]


def metricas_test(servido: ModeloServido) -> dict[str, Any]:
    m = _metricas(servido)
    return {
        "n_test": m["n_test"],
        "prevalencia_test": m["prevalencia_test"],
        "roc_auc": m["roc_auc"],
        "roc_auc_ic95_bootstrap": m["roc_auc_ic95_bootstrap"],
        "brier": m["brier"],
        "brier_baseline_prevalencia": m["brier_baseline_prevalencia"],
        "ece_uniform_10": m["ece_uniform_10"],
        "confusion_matrix_umbral_0_5": m["confusion_matrix"],
    }


def seleccion(servido: ModeloServido) -> dict[str, Any]:
    """Cómo se eligió el modelo servido, según el manifiesto."""
    b = servido.bloque
    if "seleccion" not in b:  # DT-1
        return {"criterio": b.get("criterio_seleccion"), "algoritmo": servido.algoritmo}
    s = b["seleccion"]
    return {
        "criterio": "log-loss media en validación cruzada de 5 folds sobre train; regla de 1 EE",
        "candidato": b["candidato_seleccionado"],
        "hiperparametros": b["hiperparametros"],
        "mejor_en_cv": s["mejor"],
        "n_elegibles_dentro_de_1_ee": len(s["elegibles"]),
        "test_intervino_en_la_seleccion": False,
    }


def nota_metricas(manifiesto: dict[str, Any]) -> str:
    base = (
        "La probabilidad no lleva calibración post-hoc; su calibración se mide con ECE (10 bins "
        "uniformes) en test."
    )
    if manifiesto.get("hito") == "DT-4":
        return (
            "Métricas del modelo seleccionado sobre el test, evaluado una sola vez después de la "
            "selección (DT-4, protocolo preregistrado en docs/DT4_PROTOCOL.md). La selección se hizo "
            "por log-loss media en validación cruzada de 5 folds sobre el train, con la regla de 1 EE; "
            f"el test no intervino. {base} Limitación declarada: ese test ya se había observado en DT-1."
        )
    return (
        f"Métricas del ganador sobre el test de DT-1. {base} El mismo test eligió al ganador entre 5 "
        "algoritmos: las cifras están sesgadas al alza por selección (corregido en DT-4)."
    )


def advertencia_b1(servido: ModeloServido) -> str:
    cm = _metricas(servido)["confusion_matrix"]
    return (
        "Experimental. Estima hipertensión SIN medir la presión arterial. La información está en el "
        "ranking de riesgo, no en la decisión binaria: con umbral 0,5 el modelo servido deja "
        f"{cm['fn']} falsos negativos frente a {cm['tp']} verdaderos positivos en test. Por eso la "
        "respuesta no incluye clase. La probabilidad ordena riesgo; no diagnostica."
    )
