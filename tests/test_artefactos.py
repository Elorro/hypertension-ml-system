"""Perfiles de versiones de src/artefactos.py. Sin artefactos: corren en CI."""

from __future__ import annotations

import platform

import pytest

from src import artefactos
from src.artefactos import ArtefactoInvalido, comparar_versiones, exigir_versiones


def _manifiesto(python: str | None = None) -> dict:
    return {
        "entorno": {
            "python": python or platform.python_version(),
            "scikit_learn": "1.9.1",
            "numpy": "2.5.3",
            "pandas": "2.3.3",
            "xgboost": "3.4.1",
        }
    }


def _paquetes(filas) -> list[str]:
    return [f.paquete for f in filas]


def test_perfil_entorno_exige_todo():
    filas = comparar_versiones(_manifiesto(), "entorno", ["RandomForest"])
    assert _paquetes(filas) == ["python", "scikit_learn", "numpy", "joblib", "pandas", "xgboost"]


def test_perfil_servicio_sin_pandas_ni_xgboost():
    filas = comparar_versiones(_manifiesto(), "servicio", ["RandomForest", "RandomForest"])
    assert _paquetes(filas) == ["python", "scikit_learn", "numpy", "joblib"]


def test_perfil_servicio_exige_xgboost_si_un_ganador_lo_es():
    filas = comparar_versiones(_manifiesto(), "servicio", ["RandomForest", "XGBoost"])
    assert "xgboost" in _paquetes(filas) and "pandas" not in _paquetes(filas)


def test_python_por_major_minor():
    mayor, menor, _ = platform.python_version_tuple()
    filas = comparar_versiones(_manifiesto(f"{mayor}.{menor}.999"), "servicio", [])
    assert filas[0].ok and filas[0].esperado == f"{mayor}.{menor}"
    assert not comparar_versiones(_manifiesto("2.7.18"), "servicio", [])[0].ok


def test_joblib_se_compara_con_la_constante_no_con_el_manifiesto():
    (joblib_fila,) = [f for f in comparar_versiones(_manifiesto(), "servicio", []) if f.paquete == "joblib"]
    assert joblib_fila.esperado == artefactos.JOBLIB_REFERENCIA == "1.6.0"


def test_exigir_versiones_falla_con_mensaje(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(artefactos, "_instalada", lambda d: "0.0.0")
    with pytest.raises(ArtefactoInvalido, match=r"scikit_learn 0\.0\.0 \(esperado 1\.9\.1\)"):
        exigir_versiones(_manifiesto(), "servicio", [])
