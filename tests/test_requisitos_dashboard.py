"""app/requirements.txt (Streamlit Community Cloud) debe ser requirements-dashboard.txt."""

from __future__ import annotations

from tests.conftest import ROOT


def _paquetes(ruta) -> list[str]:
    lineas = (ROOT / ruta).read_text().splitlines()
    return [ln.split("#")[0].strip() for ln in lineas if ln.split("#")[0].strip()]


def test_app_requirements_igual_a_requirements_dashboard():
    assert _paquetes("app/requirements.txt") == _paquetes("requirements-dashboard.txt")
    assert not any(
        p.startswith(("scikit-learn", "xgboost", "joblib")) for p in _paquetes("app/requirements.txt")
    )
