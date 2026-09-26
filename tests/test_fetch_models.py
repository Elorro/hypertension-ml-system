"""scripts/fetch_models.py contra un servidor HTTP local. Sin artefactos reales: corre en CI."""

from __future__ import annotations

import functools
import hashlib
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts import fetch_models as fm


def _manifiesto(archivos: dict[str, bytes]) -> dict:
    def art(nombre: str) -> dict:
        return {"ruta": f"models/{nombre}", "sha256": hashlib.sha256(archivos[nombre]).hexdigest()}

    def bloque(exp: str) -> dict:
        return {
            "nombre": exp,
            "features_orden": ["x"],
            "artefactos": {"modelo": art(f"dt4_{exp}__modelo.pkl"), "scaler": art(f"dt4_{exp}__scaler.pkl")},
        }

    return {
        "experimentos": {
            "A_prima_principal": {"con_pa": bloque("riesgo_cv_con_pa"), "sin_pa": bloque("riesgo_cv_sin_pa")},
            "B1_experimento": {"principal": bloque("hta_b1")},
        }
    }


@pytest.fixture
def origen(tmp_path: Path) -> Iterator[tuple[str, Path, dict]]:
    """Servidor HTTP sobre un directorio con artefactos falsos; devuelve (url, dir, manifiesto)."""
    publicados = tmp_path / "release"
    publicados.mkdir()
    archivos = {
        f"dt4_{e}__{k}.pkl": f"{e}-{k}".encode() * 1000
        for e in ("riesgo_cv_con_pa", "riesgo_cv_sin_pa", "hta_b1")
        for k in ("modelo", "scaler")
    }
    for nombre, contenido in archivos.items():
        (publicados / nombre).write_bytes(contenido)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(publicados))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    hilo = threading.Thread(target=srv.serve_forever, daemon=True)
    hilo.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", publicados, _manifiesto(archivos)
    finally:
        srv.shutdown()


def _correr(url: str, manifiesto: dict, destino: Path, tmp_path: Path) -> int:
    import json

    ruta = tmp_path / "manifest.json"
    ruta.write_text(json.dumps(manifiesto))
    return fm.main(["--base-url", url, "--manifest", str(ruta), "--model-dir", str(destino)])


def test_servidos_coinciden_con_la_api():
    from api.servicio import EXPERIMENTOS_SERVIDOS

    assert fm.EXPERIMENTOS_SERVIDOS == EXPERIMENTOS_SERVIDOS


def test_solo_baja_los_experimentos_servidos(origen, tmp_path, capsys):
    url, _, manifiesto = origen
    destino = tmp_path / "models"
    assert _correr(url, manifiesto, destino, tmp_path) == 0
    assert sorted(p.name for p in destino.iterdir()) == sorted(
        f"dt4_{e}__{k}.pkl" for e in ("riesgo_cv_con_pa", "hta_b1") for k in ("modelo", "scaler")
    )
    assert capsys.readouterr().out.count("descargado") == 4


def test_segunda_ejecucion_no_descarga(origen, tmp_path, capsys, monkeypatch):
    url, _, manifiesto = origen
    destino = tmp_path / "models"
    assert _correr(url, manifiesto, destino, tmp_path) == 0
    capsys.readouterr()

    def prohibido(*a, **k):
        raise AssertionError("no debía descargar")

    monkeypatch.setattr(fm, "descargar", prohibido)
    assert _correr(url, manifiesto, destino, tmp_path) == 0
    assert capsys.readouterr().out.count("presente") == 4


def test_asset_alterado_falla_y_no_deja_archivo(origen, tmp_path, capsys):
    url, publicados, manifiesto = origen
    alterado = "dt4_hta_b1__scaler.pkl"
    with (publicados / alterado).open("ab") as fh:
        fh.write(b"\0")
    destino = tmp_path / "models"
    assert _correr(url, manifiesto, destino, tmp_path) == 1
    assert "sha256 distinto" in capsys.readouterr().err
    assert not (destino / alterado).exists()
    assert not list(destino.glob("*.part"))


def test_local_corrupto_se_reemplaza(origen, tmp_path):
    url, _, manifiesto = origen
    destino = tmp_path / "models"
    destino.mkdir()
    (destino / "dt4_hta_b1__modelo.pkl").write_bytes(b"basura")
    assert _correr(url, manifiesto, destino, tmp_path) == 0
    esperado = manifiesto["experimentos"]["B1_experimento"]["principal"]["artefactos"]["modelo"]["sha256"]
    assert fm.sha256_file(destino / "dt4_hta_b1__modelo.pkl") == esperado


def test_asset_ausente_falla(origen, tmp_path):
    url, publicados, manifiesto = origen
    (publicados / "dt4_riesgo_cv_con_pa__modelo.pkl").unlink()
    assert _correr(url, manifiesto, tmp_path / "models", tmp_path) == 1


def test_sin_base_url(tmp_path, monkeypatch):
    monkeypatch.delenv("MODELS_BASE_URL", raising=False)
    assert fm.main(["--model-dir", str(tmp_path)]) == 2
