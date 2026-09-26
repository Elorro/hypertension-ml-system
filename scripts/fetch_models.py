"""Descarga los artefactos servidos y verifica su sha256 contra el manifiesto.

Los ``.pkl`` no se versionan en git: se distribuyen como assets de un GitHub
Release. Este script los baja de ``MODELS_BASE_URL`` (p. ej.
``https://github.com/Elorro/hypertension-ml-system/releases/download/<tag>``) a
``MODEL_DIR`` (por defecto ``models/``) y comprueba cada uno contra el sha256
registrado en el manifiesto (por defecto ``models/dt4_manifest.json``).

* Solo baja los artefactos (modelo y scaler) de los experimentos **servidos**.
* Idempotente: si el archivo ya existe con el sha256 correcto, no lo descarga.
* Descarga a ``<archivo>.part`` y solo lo renombra si el sha256 coincide; si no
  coincide, borra el ``.part`` y termina con código 1. Un archivo existente con
  sha256 incorrecto se reemplaza por la descarga (y, si esta también falla, se
  borra: nunca queda en disco un artefacto que no sea el del manifiesto).

Solo biblioteca estándar, para poder correr antes de instalar dependencias.

Uso (desde la raíz del repositorio)::

    MODELS_BASE_URL=https://…/releases/download/<tag> python -m scripts.fetch_models
    python -m scripts.fetch_models --base-url http://127.0.0.1:8900 --manifest models/dt4_manifest.json

Variables: ``MODELS_BASE_URL`` (obligatoria salvo ``--base-url``), ``MODEL_DIR``,
``MANIFEST_PATH`` y ``MODELS_TOKEN`` (opcional: cabecera ``Authorization: Bearer``,
solo si los assets no son públicos).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Misma lista que api/servicio.py::EXPERIMENTOS_SERVIDOS (un test verifica que coinciden).
# Duplicada para no importar api/ (numpy, pydantic) en un script de solo stdlib.
EXPERIMENTOS_SERVIDOS: tuple[str, ...] = ("riesgo_cv_con_pa", "hta_b1")
TIMEOUT_S: float = 60.0
CHUNK: int = 1 << 20


class DescargaFallida(Exception):
    """El artefacto no se pudo descargar o no coincide con el manifiesto."""


@dataclass(frozen=True)
class Artefacto:
    experimento: str
    nombre: str  # nombre de archivo, igual en el release y en MODEL_DIR
    sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def artefactos_servidos(
    manifiesto: dict[str, Any], servidos: tuple[str, ...] = EXPERIMENTOS_SERVIDOS
) -> list[Artefacto]:
    """Modelo y scaler de cada experimento servido, en el orden de ``servidos``."""
    bloques: dict[str, dict[str, Any]] = {}

    def recorrer(nodo: object) -> None:
        if isinstance(nodo, dict):
            if "artefactos" in nodo and "features_orden" in nodo:
                bloques[nodo["nombre"]] = nodo
            for valor in nodo.values():
                recorrer(valor)

    recorrer(manifiesto["experimentos"])
    faltan = [e for e in servidos if e not in bloques]
    if faltan:
        raise DescargaFallida(f"el manifiesto no declara artefactos para: {', '.join(faltan)}")
    return [
        Artefacto(
            e, Path(bloques[e]["artefactos"][clave]["ruta"]).name, bloques[e]["artefactos"][clave]["sha256"]
        )
        for e in servidos
        for clave in ("modelo", "scaler")
    ]


def descargar(url: str, destino: Path, token: str | None = None) -> str:
    """Descarga ``url`` a ``destino`` calculando el sha256 al vuelo."""
    req = urllib.request.Request(url, headers={"User-Agent": "hypertension-ml/fetch_models"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp, destino.open("wb") as out:
            for chunk in iter(lambda: resp.read(CHUNK), b""):
                h.update(chunk)
                out.write(chunk)
    except OSError as exc:  # URLError, HTTPError y errores de disco heredan de OSError
        destino.unlink(missing_ok=True)
        raise DescargaFallida(f"no se pudo descargar {url}: {exc}") from exc
    return h.hexdigest()


def asegurar(art: Artefacto, base_url: str, model_dir: Path, token: str | None = None) -> str:
    """Deja ``model_dir/art.nombre`` con el sha256 del manifiesto. Devuelve qué hizo."""
    final = model_dir / art.nombre
    if final.exists() and sha256_file(final) == art.sha256:
        return "presente"
    parcial = final.with_name(final.name + ".part")
    url = f"{base_url.rstrip('/')}/{art.nombre}"
    digest = descargar(url, parcial, token)
    if digest != art.sha256:
        parcial.unlink(missing_ok=True)
        final.unlink(missing_ok=True)  # tampoco se deja uno viejo con sha256 incorrecto
        raise DescargaFallida(
            f"[{art.experimento}] sha256 distinto en {art.nombre}: {digest[:16]}… != manifiesto {art.sha256[:16]}…"
        )
    parcial.replace(final)
    return "descargado"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Descarga y verifica los artefactos servidos.")
    ap.add_argument("--base-url", default=os.environ.get("MODELS_BASE_URL"))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path(os.environ.get("MANIFEST_PATH", ROOT / "models" / "dt4_manifest.json")),
    )
    ap.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", ROOT / "models")))
    args = ap.parse_args(argv)

    if not args.base_url:
        print("✗ falta MODELS_BASE_URL (o --base-url)", file=sys.stderr)
        return 2
    try:
        arts = artefactos_servidos(json.loads(args.manifest.read_text()))
        args.model_dir.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("MODELS_TOKEN")
        for art in arts:
            estado = asegurar(art, args.base_url, args.model_dir, token)
            print(f"  ✓ {art.nombre:<36} {estado}  sha256 {art.sha256[:16]}…")
    except (DescargaFallida, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"  ✗ {exc}", file=sys.stderr)
        return 1
    print(f"artefactos servidos listos en {args.model_dir} ({len(arts)} archivos verificados)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
