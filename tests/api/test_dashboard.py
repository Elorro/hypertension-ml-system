"""Dashboard (AppTest) contra la API con modelos dobles: corre en CI sin artefactos.

``requests.get/post`` se enrutan al TestClient de la API, así que el dashboard
consume el contrato real (/health, /v1/modelos, endpoints) sin red ni .pkl.
"""

from __future__ import annotations

import ast
from typing import Any

import pytest
import requests
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT

DASHBOARD = str(ROOT / "app" / "dashboard.py")
URL = "http://api.test"
SUBMIT_A = "FormSubmitter:form_a-Calcular probabilidad"
SUBMIT_B1 = "FormSubmitter:form_b1-Calcular probabilidad"


@pytest.fixture
def app_test(client, monkeypatch: pytest.MonkeyPatch):
    def get(url: str, timeout: float) -> Any:
        return client.get(url.removeprefix(URL))

    def post(url: str, json: Any, timeout: float) -> Any:
        return client.post(url.removeprefix(URL), json=json)

    monkeypatch.setenv("API_URL", URL)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    at = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not at.exception, at.exception
    return at


def test_solo_importa_cliente_http():
    arbol = ast.parse((ROOT / "app" / "dashboard.py").read_text())
    modulos = {
        (n.module if isinstance(n, ast.ImportFrom) else a.name).split(".")[0]
        for n in ast.walk(arbol)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert modulos <= {"__future__", "io", "os", "time", "typing", "pandas", "requests", "streamlit"}


def test_limites_y_etiquetas_vienen_de_v1_modelos(app_test, client):
    catalogo = {m["experimento"]: m for m in client.get("/v1/modelos").json()["modelos"]}
    for prefijo, exp in (("a", "riesgo_cv_con_pa"), ("b1", "hta_b1")):
        campos = catalogo[exp]["entrada"]["campos"]
        for nombre, spec in campos.items():
            if "enum" in spec:
                w = app_test.selectbox(key=f"{prefijo}_{nombre}")
                assert w.label == spec["title"]
                assert w.options == [f"{v} — {spec['x-etiquetas'][str(v)]}" for v in spec["enum"]]
            else:
                w = app_test.number_input(key=f"{prefijo}_{nombre}")
                assert (w.label, w.min, w.max) == (spec["title"], spec["minimum"], spec["maximum"])
    assert app_test.number_input(key="a_age_years").min == 29.0
    assert app_test.number_input(key="a_age_years").max == 65.0


def test_gender_aclara_que_es_inferido(app_test):
    assert all("inferido" in o for o in app_test.selectbox(key="a_gender").options)
    assert any("La correspondencia es inferida" in c.value for c in app_test.caption)


def test_a_prima_muestra_probabilidad_y_clase(app_test):
    app_test.button(key=SUBMIT_A).click().run()
    metricas = {m.label: m.value for m in app_test.metric}
    assert metricas["Probabilidad estimada"] == "73,0%"  # doble: 0.73
    assert "Prevalencia base (train)" in metricas
    assert any("Clase con umbral 0,5" in m.value for m in app_test.markdown)
    assert any("no herramienta clínica" in i.value for i in app_test.info)


def test_b1_experimental_sin_clase(app_test):
    assert any("Sin presión arterial; AUC ≈ 0,69" in w.value for w in app_test.warning)
    app_test.button(key=SUBMIT_B1).click().run()
    assert {m.label: m.value for m in app_test.metric}["Probabilidad estimada"] == "41,0%"
    assert not any("Clase con umbral" in m.value for m in app_test.markdown)


def test_fuera_de_dominio_muestra_la_cota(app_test):
    app_test.number_input(key="b1_height").set_value(120.0)
    app_test.number_input(key="b1_weight").set_value(200.0)
    app_test.button(key=SUBMIT_B1).click().run()
    assert any("fuera del dominio" in e.value for e in app_test.error)
    assert any("entrada: IMC calculado 138.89" in m.value for m in app_test.markdown)


def test_metricas_sin_calibrada_y_con_nota_dt4(app_test):
    assert any("DT-4" in w.value for w in app_test.warning)
    tabla = app_test.dataframe[0].value
    assert tabla["probabilidad"].str.startswith("ECE medido en test").all()
    textos = " ".join(m.value for m in app_test.markdown) + tabla.to_string()
    assert "calibrada" not in textos


def test_lote_csv_error_por_fila(app_test):
    app_test.radio[0].set_value("hta_b1").run()
    csv = (
        b"age_years,gender,height,weight,cholesterol,gluc,smoke,alco,active\n"
        b"52,1,165,72,1,1,0,0,1\n"
        b"52,1,300,72,1,1,0,0,1\n"
        b"62,2,168,95,3,3,1,1,0\n"
    )
    app_test.file_uploader(key="csv_hta_b1").upload("lote.csv", csv, "text/csv").run()
    assert not app_test.exception, app_test.exception
    tabla = app_test.dataframe[0].value
    assert tabla["error"].tolist()[0] == "" and tabla["error"].tolist()[2] == ""
    assert "Talla (cm)" in tabla["error"].tolist()[1]
    assert tabla["probabilidad"].isna().tolist() == [False, True, False]
    assert "clase" not in tabla.columns


def test_api_caida_mensaje_explicito(monkeypatch: pytest.MonkeyPatch):
    def caida(*args: Any, **kwargs: Any) -> Any:
        raise requests.ConnectionError("Connection refused")

    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("API_WAIT_SECONDS", "0")
    monkeypatch.setattr(requests, "get", caida)
    at = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not at.exception
    assert any("No se pudo contactar la API" in e.value and "Connection refused" in e.value for e in at.error)
    assert [b.label for b in at.button] == ["Reintentar"]
    assert not at.tabs
    assert "no herramienta clínica" in at.markdown[0].value


def test_auc_de_b1_sale_de_v1_modelos(client, monkeypatch: pytest.MonkeyPatch):
    """Control: con métricas alteradas en /v1/modelos, el texto de B1 cambia con ellas."""
    import copy

    catalogo = copy.deepcopy(client.get("/v1/modelos").json())
    b1 = next(m for m in catalogo["modelos"] if m["experimento"] == "hta_b1")
    b1["metricas_test"] |= {"roc_auc": 0.5512, "roc_auc_ic95_bootstrap": [0.5011, 0.6049]}
    url = "http://api-control.test"  # otra clave para la caché de catalogo_modelos

    class Respuesta:
        def __init__(self, cuerpo: Any) -> None:
            self._cuerpo = cuerpo

        def raise_for_status(self) -> None:
            pass

        def json(self) -> Any:
            return self._cuerpo

    def get(u: str, timeout: float) -> Any:
        ruta = u.removeprefix(url)
        return Respuesta(catalogo) if ruta == "/v1/modelos" else client.get(ruta)

    monkeypatch.setenv("API_URL", url)
    monkeypatch.setattr(requests, "get", get)
    at = AppTest.from_file(DASHBOARD, default_timeout=30).run()
    assert not at.exception, at.exception
    textos = [w.value for w in at.warning]
    assert any("Sin presión arterial; AUC ≈ 0,55 [0,50 · 0,60]" in t for t in textos), textos
    assert not any("0,69" in t for t in textos)
