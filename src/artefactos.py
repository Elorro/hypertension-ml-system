"""Verificación y carga de los artefactos de DT-1 contra su manifiesto.

Lo usan ``scripts/verify_env.py`` (perfil ``entorno``) y la API al arrancar
(perfil ``servicio``). Un fallo levanta ``ArtefactoInvalido`` con un mensaje que
dice qué no coincide: la API no arranca con artefactos o versiones dudosos.

Perfiles de versiones:

* ``entorno`` — desarrollo y reentreno. Reproducir DT-1 exige sklearn, numpy,
  pandas, joblib y xgboost exactos.
* ``servicio`` — deserializar y predecir. Exige sklearn, numpy y joblib exactos
  (son los que fijan la compatibilidad del pickle); xgboost solo si algún
  ganador servido es XGBoost; pandas no interviene en la inferencia.

En ambos Python se compara por major.minor: el formato del pickle no cambia
entre parches.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import platform
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

Perfil = Literal["entorno", "servicio"]

# El manifiesto de DT-1 no registra la versión de joblib. Procedencia de esta
# referencia: requirements.lock.txt (2026-09-23). Con ella los sha256 de los
# artefactos coinciden con los de la corrida y cargan sin warnings de versión.
# No está demostrado que sea la versión con la que se serializaron.
JOBLIB_REFERENCIA: str = "1.6.0"

# Nombre en el manifiesto -> distribución instalada.
DISTRIBUCIONES: dict[str, str] = {
    "scikit_learn": "scikit-learn",
    "numpy": "numpy",
    "pandas": "pandas",
    "joblib": "joblib",
    "xgboost": "xgboost",
}

TOL_SCALER: float = 1e-9


class ArtefactoInvalido(Exception):
    """El entorno o un artefacto no coincide con el manifiesto."""


@dataclass(frozen=True)
class Comparacion:
    paquete: str
    esperado: str
    instalado: str

    @property
    def ok(self) -> bool:
        return self.instalado == self.esperado


@dataclass(frozen=True)
class ModeloCargado:
    experimento: str
    algoritmo: str
    features: list[str]
    modelo: Any
    scaler: Any
    sha256_modelo: str
    sha256_scaler: str
    bloque: dict[str, Any]  # bloque del manifiesto, para metadatos


# =======================================================
# Manifiesto
# =======================================================
def experimentos(manifiesto: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Bloques del manifiesto que declaran artefactos, por nombre de experimento."""
    encontrados: dict[str, dict[str, Any]] = {}

    def recorrer(nodo: object) -> None:
        if isinstance(nodo, dict):
            if "artefactos" in nodo and "features_orden" in nodo:
                encontrados[nodo["nombre"]] = nodo
            for valor in nodo.values():
                recorrer(valor)

    recorrer(manifiesto["experimentos"])
    return encontrados


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =======================================================
# Versiones
# =======================================================
def _major_minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def _instalada(distribucion: str) -> str:
    try:
        return md.version(distribucion)
    except md.PackageNotFoundError:
        return "AUSENTE"


def comparar_versiones(manifiesto: dict[str, Any], perfil: Perfil, ganadores: list[str]) -> list[Comparacion]:
    """Versiones exigidas por el perfil frente a las instaladas."""
    entorno = manifiesto["entorno"]
    exigidos = ["scikit_learn", "numpy", "joblib"]
    if perfil == "entorno":
        exigidos += ["pandas", "xgboost"]
    elif "XGBoost" in ganadores:
        exigidos.append("xgboost")

    filas = [Comparacion("python", _major_minor(entorno["python"]), _major_minor(platform.python_version()))]
    for paquete in exigidos:
        esperado = JOBLIB_REFERENCIA if paquete == "joblib" else entorno[paquete]
        filas.append(Comparacion(paquete, esperado, _instalada(DISTRIBUCIONES[paquete])))
    return filas


def exigir_versiones(manifiesto: dict[str, Any], perfil: Perfil, ganadores: list[str]) -> list[Comparacion]:
    filas = comparar_versiones(manifiesto, perfil, ganadores)
    malas = [f for f in filas if not f.ok]
    if malas:
        detalle = ", ".join(f"{f.paquete} {f.instalado} (esperado {f.esperado})" for f in malas)
        raise ArtefactoInvalido(
            f"versiones distintas a la corrida de referencia [perfil {perfil}]: {detalle}"
        )
    return filas


# =======================================================
# Artefactos
# =======================================================
def ruta_artefacto(exp: dict[str, Any], clave: str, model_dir: Path) -> Path:
    """El manifiesto guarda rutas relativas al repo; se resuelven dentro de ``model_dir``."""
    return model_dir / Path(exp["artefactos"][clave]["ruta"]).name


def cargar_experimento(exp: dict[str, Any], model_dir: Path) -> ModeloCargado:
    """sha256 → carga sin warnings → identidad del scaler → dimensión."""
    nombre = exp["nombre"]
    rutas: dict[str, Path] = {}
    for clave in ("modelo", "scaler"):
        ruta = ruta_artefacto(exp, clave, model_dir)
        if not ruta.exists():
            raise ArtefactoInvalido(f"[{nombre}] falta {ruta}")
        digest = sha256_file(ruta)
        esperado = exp["artefactos"][clave]["sha256"]
        if digest != esperado:
            raise ArtefactoInvalido(
                f"[{nombre}] sha256 distinto en {ruta}: {digest[:16]}… != manifiesto {esperado[:16]}…"
            )
        rutas[clave] = ruta

    # Un pickle de sklearn cargado con otra versión emite UserWarning y puede
    # cargar con semántica distinta: se trata como fallo.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            modelo = joblib.load(rutas["modelo"])
            scaler = joblib.load(rutas["scaler"])
        except Exception as exc:  # noqa: BLE001 - el mensaje exacto es el diagnóstico
            raise ArtefactoInvalido(f"[{nombre}] carga fallida: {type(exc).__name__}: {exc}") from exc

    # Cargar «un» scaler no prueba nada: debe ser el de la corrida de referencia.
    ref = exp["scaler"]
    for atributo in ("mean_", "scale_"):
        delta = float(np.max(np.abs(getattr(scaler, atributo) - np.asarray(ref[atributo]))))
        if delta > TOL_SCALER:
            raise ArtefactoInvalido(
                f"[{nombre}] {atributo} del scaler se desvía del manifiesto (|Δ| = {delta:.3e})"
            )

    features = list(exp["features_orden"])
    if scaler.n_features_in_ != len(features) or modelo.n_features_in_ != len(features):
        raise ArtefactoInvalido(
            f"[{nombre}] el manifiesto declara {len(features)} features; scaler espera "
            f"{scaler.n_features_in_}, modelo {modelo.n_features_in_}"
        )

    return ModeloCargado(
        experimento=nombre,
        algoritmo=exp["ganador"],
        features=features,
        modelo=modelo,
        scaler=scaler,
        sha256_modelo=exp["artefactos"]["modelo"]["sha256"],
        sha256_scaler=exp["artefactos"]["scaler"]["sha256"],
        bloque=exp,
    )
