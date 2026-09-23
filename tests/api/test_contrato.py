"""Tests de contrato de la API. Corren sin artefactos: el modelo es un doble."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from api.schemas import CODIFICACION_GENDER_MANIFIESTO, MAX_FILAS_LOTE
from src.artefactos import ArtefactoInvalido
from src.cardio_features import bmi
from tests.api.conftest import BASE_OK, PA_OK
from tests.conftest import ROOT

A = "/v1/riesgo-cardiovascular"
B1 = "/v1/hipertension-sin-pa"
CLAVES_B1 = {"probabilidad", "prevalencia_base", "modelo", "aviso"}


def _tipos(resp: Any) -> list[str]:
    return [e["type"] for e in resp.json()["detail"]]


def _mensajes(resp: Any) -> str:
    return " | ".join(e["msg"] for e in resp.json()["detail"])


# =======================================================
# Servicio
# =======================================================
def test_health(client, manifiesto):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["estado"] == "ok"
    assert {m["experimento"] for m in body["modelos"]} == {"riesgo_cv_con_pa", "hta_b1"}
    assert set(body["entorno"]) == {"python", "scikit_learn", "numpy", "joblib"}
    for m in body["modelos"]:
        assert len(m["sha256"]) == 64


def test_version_api_es_la_de_pyproject(client):
    from api.main import API_VERSION

    proyecto = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert proyecto == API_VERSION
    assert client.get("/health").json()["version_api"] == proyecto
    assert client.get("/openapi.json").json()["info"]["version"] == proyecto


def test_sin_registro_devuelve_503():
    from fastapi.testclient import TestClient

    from api.main import create_app

    with TestClient(create_app(cargar_artefactos=False)) as c:
        assert c.get("/health").status_code == 503
        assert c.post(A, json=BASE_OK | PA_OK).status_code == 503


def test_api_no_arranca_si_falta_artefacto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fail fast sin .pkl: el lifespan levanta ArtefactoInvalido y la API no sirve nada."""
    from fastapi.testclient import TestClient

    from api.main import create_app

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with pytest.raises(ArtefactoInvalido, match=r"\[riesgo_cv_con_pa\] falta"), TestClient(create_app()):
        pass


def test_contrato_sintetico_eliminado(client):
    assert client.post("/predecir", json={}).status_code == 404
    for archivo in (ROOT / "api").glob("*.py"):
        texto = archivo.read_text()
        assert "modelo_" not in texto and "mejor_modelo" not in texto, archivo


def test_modelos_metadatos(client, manifiesto):
    body = client.get("/v1/modelos").json()
    assert body["max_filas_lote"] == MAX_FILAS_LOTE
    assert "DT-4" in body["nota_metricas"] and "ECE" in body["nota_metricas"]
    assert "calibrada" not in body["nota_metricas"]
    por_exp = {m["experimento"]: m for m in body["modelos"]}

    a, b1 = por_exp["riesgo_cv_con_pa"], por_exp["hta_b1"]
    assert a["devuelve_clase"] is True and a["umbral_clase"] == 0.5 and a["advertencia"] is None
    assert b1["devuelve_clase"] is False and b1["umbral_clase"] is None
    cm = manifiesto["experimentos"]["B1_experimento"]["principal"]["modelos"]["RandomForest"][
        "confusion_matrix"
    ]
    assert str(cm["fn"]) in b1["advertencia"] and str(cm["tp"]) in b1["advertencia"]
    assert "ranking" in b1["advertencia"]

    # Entrada en unidades humanas: sin bmi, sin age en días; B1 sin presión.
    assert "bmi" not in a["entrada"]["campos"] and "age" not in a["entrada"]["campos"]
    assert {"ap_hi", "ap_lo"} <= set(a["entrada"]["campos"])
    assert not {"ap_hi", "ap_lo"} & set(b1["entrada"]["campos"])
    assert a["entrada"]["campos"]["height"]["minimum"] == 120.0
    assert a["entrada"]["campos"]["age_years"]["maximum"] == 65.0

    for m in (a, b1):
        met = m["metricas_test"]
        assert {"roc_auc", "roc_auc_ic95_bootstrap", "brier", "ece_uniform_10"} <= set(met)
        assert len(met["roc_auc_ic95_bootstrap"]) == 2


def test_gender_cita_el_manifiesto(client, manifiesto):
    assert manifiesto["dataset"]["codificacion_gender"] == CODIFICACION_GENDER_MANIFIESTO
    desc = client.get("/openapi.json").json()["components"]["schemas"]["EntradaRiesgoCV"]["properties"]
    assert CODIFICACION_GENDER_MANIFIESTO in desc["gender"]["description"]
    assert "inferida" in desc["gender"]["description"]


def test_campos_con_titulo_y_etiquetas_por_codigo(client):
    """El dashboard construye sus formularios solo con esto: nada del dominio a mano."""
    for m in client.get("/v1/modelos").json()["modelos"]:
        for nombre, spec in m["entrada"]["campos"].items():
            assert spec["title"] and spec["title"] != nombre.replace("_", " ").title(), nombre
            if "enum" in spec:
                assert sorted(spec["x-etiquetas"]) == sorted(str(v) for v in spec["enum"]), nombre
                for codigo, etiqueta in spec["x-etiquetas"].items():
                    assert f"{codigo} = {etiqueta}" in spec["description"] or nombre == "gender"
            else:
                assert {"minimum", "maximum", "examples"} <= set(spec), nombre


# =======================================================
# Respuestas individuales
# =======================================================
def test_riesgo_cv_forma_y_clase(client, manifiesto, dobles):
    r = client.post(A, json=BASE_OK | PA_OK)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == CLAVES_B1 | {"clase", "umbral"}
    assert body["probabilidad"] == pytest.approx(0.73)
    assert body["clase"] == 1 and body["umbral"] == 0.5
    assert (
        body["prevalencia_base"]
        == manifiesto["experimentos"]["A_prima_principal"]["con_pa"]["prevalencia_train"]
    )
    assert body["modelo"]["experimento"] == "riesgo_cv_con_pa"
    assert "no herramienta clínica" in body["aviso"]


@pytest.mark.parametrize(("p", "clase"), [(0.5, 1), (0.4999, 0)])
def test_riesgo_cv_umbral_inclusivo(client, dobles, p, clase):
    dobles.riesgo_cv.probabilidad = p
    assert client.post(A, json=BASE_OK | PA_OK).json()["clase"] == clase


def test_b1_sin_clase(client, manifiesto):
    r = client.post(B1, json=BASE_OK)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == CLAVES_B1
    assert body["probabilidad"] == pytest.approx(0.41)
    assert (
        body["prevalencia_base"]
        == manifiesto["experimentos"]["B1_experimento"]["principal"]["prevalencia_train"]
    )


def test_matriz_orden_y_derivacion(client, manifiesto, dobles):
    """La matriz que recibe el modelo sigue features_orden y deriva bmi en el servidor."""
    entrada = BASE_OK | {"age_years": 47.3, "gender": 2, "cholesterol": 3, "smoke": 1} | PA_OK
    client.post(A, json=entrada)
    client.post(B1, json={k: v for k, v in entrada.items() if k not in PA_OK})

    valores = entrada | {"bmi": bmi(entrada["weight"], entrada["height"])}
    for doble, bloque in (
        (dobles.riesgo_cv, manifiesto["experimentos"]["A_prima_principal"]["con_pa"]),
        (dobles.hta_b1, manifiesto["experimentos"]["B1_experimento"]["principal"]),
    ):
        (X,) = doble.llamadas
        esperado = np.array([[float(valores[f]) for f in bloque["features_orden"]]], dtype=np.float64)
        assert X.dtype == np.float64
        np.testing.assert_array_equal(X, esperado)


# =======================================================
# Anti-leakage y campos extra
# =======================================================
@pytest.mark.parametrize("extra", [{"ap_hi": 130.0}, {"ap_lo": 85.0}, PA_OK])
def test_b1_rechaza_presion(client, dobles, extra):
    r = client.post(B1, json=BASE_OK | extra)
    assert r.status_code == 422
    assert set(_tipos(r)) == {"extra_forbidden"}
    assert dobles.hta_b1.llamadas == []


@pytest.mark.parametrize("ruta", [A, B1])
@pytest.mark.parametrize("extra", [{"bmi": 26.4}, {"age": 18993}, {"campo_inventado": 1}])
def test_campo_extra_422(client, ruta, extra):
    cuerpo = BASE_OK | (PA_OK if ruta == A else {}) | extra
    r = client.post(ruta, json=cuerpo)
    assert r.status_code == 422
    assert "extra_forbidden" in _tipos(r)


# =======================================================
# Dominio de entrenamiento
# =======================================================
FUERA_DE_DOMINIO: list[tuple[str, dict[str, Any], str]] = [
    ("talla_baja", {"height": 119.9}, "greater than or equal to 120"),
    ("talla_alta", {"height": 220.5}, "less than or equal to 220"),
    ("peso_bajo", {"weight": 29.9}, "greater than or equal to 30"),
    ("peso_alto", {"weight": 200.1}, "less than or equal to 200"),
    ("edad_baja", {"age_years": 28.9}, "greater than or equal to 29"),
    ("edad_alta", {"age_years": 65.1}, "less than or equal to 65"),
    ("imc_alto", {"height": 150.0, "weight": 190.0}, "IMC calculado 84.44"),
    ("imc_bajo", {"height": 219.0, "weight": 55.0}, "IMC calculado 11.47"),
    ("gender", {"gender": 0}, "Input should be 1 or 2"),
    ("cholesterol", {"cholesterol": 4}, "Input should be 1, 2 or 3"),
    ("smoke", {"smoke": 2}, "Input should be 0 or 1"),
]
PRESION_FUERA: list[tuple[str, dict[str, Any], str]] = [
    ("ap_lo_igual", {"ap_hi": 120.0, "ap_lo": 120.0}, "ap_lo (120) debe ser menor que ap_hi (120)"),
    ("ap_lo_mayor", {"ap_hi": 80.0, "ap_lo": 90.0}, "ap_lo (90) debe ser menor que ap_hi (80)"),
    ("ap_hi_bajo", {"ap_hi": 69.0, "ap_lo": 45.0}, "greater than or equal to 70"),
    ("ap_hi_alto", {"ap_hi": 251.0, "ap_lo": 100.0}, "less than or equal to 250"),
    ("ap_lo_bajo", {"ap_hi": 100.0, "ap_lo": 39.0}, "greater than or equal to 40"),
    ("ap_lo_alto", {"ap_hi": 250.0, "ap_lo": 201.0}, "less than or equal to 200"),
]


@pytest.mark.parametrize(
    ("caso", "cambio", "mensaje"), FUERA_DE_DOMINIO, ids=[c[0] for c in FUERA_DE_DOMINIO]
)
@pytest.mark.parametrize("ruta", [A, B1])
def test_fuera_de_dominio_422(client, ruta, caso, cambio, mensaje):
    cuerpo = BASE_OK | (PA_OK if ruta == A else {}) | cambio
    r = client.post(ruta, json=cuerpo)
    assert r.status_code == 422
    assert mensaje in _mensajes(r), _mensajes(r)


@pytest.mark.parametrize(("caso", "cambio", "mensaje"), PRESION_FUERA, ids=[c[0] for c in PRESION_FUERA])
def test_presion_fuera_de_dominio_422(client, dobles, caso, cambio, mensaje):
    r = client.post(A, json=BASE_OK | cambio)
    assert r.status_code == 422
    assert mensaje in _mensajes(r), _mensajes(r)
    assert dobles.riesgo_cv.llamadas == []


@pytest.mark.parametrize(
    "cambio",
    [
        {"height": 120.0, "weight": 30.0},  # IMC 20,8
        {"height": 220.0, "weight": 200.0},  # IMC 41,3
        {"age_years": 29.0},
        {"age_years": 65.0},
        {"ap_hi": 250.0, "ap_lo": 200.0},
        {"ap_hi": 70.0, "ap_lo": 40.0},
    ],
)
def test_cotas_inclusivas(client, cambio):
    assert client.post(A, json=BASE_OK | PA_OK | cambio).status_code == 200


# =======================================================
# Lote
# =======================================================
def test_lote_fila_invalida_por_indice(client, dobles):
    filas = [
        BASE_OK | PA_OK,
        BASE_OK | {"ap_hi": 80.0, "ap_lo": 90.0},
        BASE_OK | PA_OK | {"height": 300.0},
        7,
    ]
    filas.append(BASE_OK | PA_OK | {"age_years": 60.0})
    r = client.post(f"{A}/lote", json=filas)
    assert r.status_code == 200
    body = r.json()
    assert (body["n_filas"], body["n_ok"], body["n_error"]) == (5, 2, 3)
    assert [f["indice"] for f in body["filas"]] == [0, 1, 2, 3, 4]
    assert [f["ok"] for f in body["filas"]] == [True, False, False, False, True]
    assert "debe ser menor que ap_hi" in body["filas"][1]["errores"][0]["msg"]
    assert body["filas"][2]["errores"][0]["loc"] == ["height"]
    assert body["filas"][0]["resultado"]["clase"] == 1
    # Una sola llamada al modelo, solo con las filas válidas.
    (X,) = dobles.riesgo_cv.llamadas
    assert X.shape == (2, 12)


def test_lote_b1_rechaza_presion_por_fila(client):
    r = client.post(f"{B1}/lote", json=[BASE_OK, BASE_OK | PA_OK])
    body = r.json()
    assert r.status_code == 200 and body["n_ok"] == 1
    assert "clase" not in body["filas"][0]["resultado"]
    assert {e["type"] for e in body["filas"][1]["errores"]} == {"extra_forbidden"}


@pytest.mark.parametrize("ruta", [A, B1])
def test_lote_tope(client, ruta):
    fila = BASE_OK | (PA_OK if ruta == A else {})
    assert client.post(f"{ruta}/lote", json=[fila] * MAX_FILAS_LOTE).status_code == 200
    r = client.post(f"{ruta}/lote", json=[fila] * (MAX_FILAS_LOTE + 1))
    assert r.status_code == 422
    assert _tipos(r) == ["too_long"] and "at most 1000" in _mensajes(r)
    assert "input" not in r.json()["detail"][0]  # sin eco del lote


def test_lote_vacio_422(client):
    assert client.post(f"{A}/lote", json=[]).status_code == 422


def test_api_no_importa_pandas_ni_xgboost():
    """El camino de serving no depende de pandas ni de xgboost (requirements-serve.txt)."""
    for archivo in [
        *Path(ROOT / "api").glob("*.py"),
        ROOT / "src/cardio_features.py",
        ROOT / "src/artefactos.py",
    ]:
        texto = archivo.read_text()
        assert "import pandas" not in texto and "import xgboost" not in texto, archivo
