"""Integración con los .pkl reales. Se saltan si faltan (no están en git)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.artefactos import ArtefactoInvalido
from tests.api.conftest import BASE_OK, PA_OK
from tests.conftest import ARTEFACTOS, MANIFEST_PATH

pytestmark = pytest.mark.requires_artifacts

BAJO = {
    "age_years": 40.0,
    "gender": 1,
    "height": 170.0,
    "weight": 65.0,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}
ALTO = {
    "age_years": 62.0,
    "gender": 2,
    "height": 168.0,
    "weight": 95.0,
    "cholesterol": 3,
    "gluc": 3,
    "smoke": 1,
    "alco": 1,
    "active": 0,
}


@pytest.fixture(scope="module")
def client_real():
    from fastapi.testclient import TestClient

    from api.main import create_app

    with TestClient(create_app()) as c:
        yield c


def test_carga_real_y_sha(client_real, manifiesto):
    from src.artefactos import experimentos

    bloques = experimentos(manifiesto)
    for m in client_real.get("/health").json()["modelos"]:
        assert m["sha256"] == bloques[m["experimento"]]["artefactos"]["modelo"]["sha256"]
        assert m["algoritmo"] == bloques[m["experimento"]]["ganador"]


def test_n_jobs_uno_en_memoria(client_real):
    registro = client_real.app.state.registro
    for servido in registro.modelos.values():
        assert servido.estimador.get_params()["n_jobs"] == 1


def test_monotonia_riesgo_cv(client_real):
    bajo = client_real.post("/v1/riesgo-cardiovascular", json=BAJO | {"ap_hi": 110.0, "ap_lo": 70.0}).json()
    alto = client_real.post("/v1/riesgo-cardiovascular", json=ALTO | {"ap_hi": 165.0, "ap_lo": 100.0}).json()
    assert alto["probabilidad"] > bajo["probabilidad"]
    assert (bajo["clase"], alto["clase"]) == (0, 1)


def test_monotonia_b1(client_real):
    bajo = client_real.post("/v1/hipertension-sin-pa", json=BAJO).json()
    alto = client_real.post("/v1/hipertension-sin-pa", json=ALTO).json()
    assert alto["probabilidad"] > bajo["probabilidad"]


def test_lote_coincide_con_individual(client_real):
    filas = [BAJO | {"ap_hi": 110.0, "ap_lo": 70.0}, BASE_OK | PA_OK, ALTO | {"ap_hi": 165.0, "ap_lo": 100.0}]
    lote = client_real.post("/v1/riesgo-cardiovascular/lote", json=filas).json()
    for fila, res in zip(filas, lote["filas"], strict=True):
        individual = client_real.post("/v1/riesgo-cardiovascular", json=fila).json()
        assert res["resultado"]["probabilidad"] == individual["probabilidad"]


def test_no_arranca_con_sha_alterado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Copia alterada de un artefacto: el lifespan debe fallar antes de servir nada."""
    from fastapi.testclient import TestClient

    from api.main import create_app

    alterado = "dt1_hta_b1__scaler.pkl"
    for pkl in ARTEFACTOS:
        destino = tmp_path / pkl.name
        if pkl.name == alterado:
            shutil.copy(pkl, destino)
            with destino.open("ab") as fh:
                fh.write(b"\x00")
        else:
            destino.symlink_to(pkl)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MANIFEST_PATH", str(MANIFEST_PATH))

    with (
        pytest.raises(ArtefactoInvalido, match=rf"sha256 distinto en .*{alterado}"),
        TestClient(create_app()),
    ):
        pass
