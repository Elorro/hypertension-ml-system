"""Servicio de inferencia de DT-1: probabilidad, no diagnóstico.

Sirve dos modelos entrenados sobre el dataset real (``models/dt1_*.pkl``):

* ``POST /v1/riesgo-cardiovascular`` — A′, con presión arterial (principal).
* ``POST /v1/hipertension-sin-pa`` — B1, sin presión arterial (experimental).

Los artefactos se cargan una vez al arrancar (lifespan), tras verificar versiones y
sha256 contra el manifiesto: si algo no coincide, el servicio no arranca.

Configuración por variables de entorno:

* ``MODEL_DIR`` — directorio de los ``.pkl`` (por defecto ``models/`` del repo).
* ``MANIFEST_PATH`` — manifiesto de DT-1 (por defecto ``models/dt1_manifest.json``).

Uso (desde la raíz del repositorio)::

    uvicorn api.main:app --host 127.0.0.1 --port 8000

AVISO: demostración de ingeniería de ML, no herramienta clínica.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import ValidationError

from api.schemas import (
    AVISO,
    MAX_FILAS_LOTE,
    UMBRAL_CLASE,
    EntradaBase,
    EntradaHTASinPA,
    EntradaRiesgoCV,
    ErrorCampo,
    FilaLote,
    RespuestaHTASinPA,
    RespuestaLote,
    RespuestaRiesgoCV,
)
from api.servicio import (
    HTA_B1,
    RIESGO_CV,
    ModeloServido,
    Registro,
    advertencia_b1,
    cargar_registro,
    entorno_en_ejecucion,
    metricas_test,
)
from src.cardio_features import BMI_RANGE

# Versión del proyecto (pyproject.toml); un test verifica que coinciden. /v1 en las
# rutas es el namespace del contrato, no esta versión.
API_VERSION: str = "2.0.0"
ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class Config:
    model_dir: Path
    manifest_path: Path

    @classmethod
    def desde_entorno(cls) -> Config:
        return cls(
            model_dir=Path(os.environ.get("MODEL_DIR", ROOT / "models")),
            manifest_path=Path(os.environ.get("MANIFEST_PATH", ROOT / "models" / "dt1_manifest.json")),
        )


def get_registro(request: Request) -> Registro:
    registro: Registro | None = getattr(request.app.state, "registro", None)
    if registro is None:
        raise HTTPException(status_code=503, detail="modelos no cargados")
    return registro


RegistroDep = Annotated[Registro, Depends(get_registro)]


# =======================================================
# Predicción
# =======================================================
def _respuesta(servido: ModeloServido, p: float) -> RespuestaRiesgoCV | RespuestaHTASinPA:
    base = {"probabilidad": p, "prevalencia_base": servido.prevalencia_base, "modelo": servido.info()}
    if servido.experimento == RIESGO_CV:
        return RespuestaRiesgoCV(**base, clase=int(p >= UMBRAL_CLASE), umbral=UMBRAL_CLASE)
    return RespuestaHTASinPA(**base)


def _predecir_uno(servido: ModeloServido, entrada: EntradaBase) -> Any:
    return _respuesta(servido, float(servido.probabilidades([entrada])[0]))


def _predecir_lote(servido: ModeloServido, esquema: type[EntradaBase], filas: list[Any]) -> dict[str, Any]:
    """Valida fila a fila; una fila inválida lleva su error por índice y no tumba el lote."""
    validas: list[tuple[int, EntradaBase]] = []
    salida: list[FilaLote[Any]] = []
    for i, fila in enumerate(filas):
        try:
            validas.append((i, esquema.model_validate(fila)))
        except ValidationError as exc:
            errores = [
                ErrorCampo(loc=list(e["loc"]), msg=e["msg"], type=e["type"])
                for e in exc.errors(include_url=False, include_context=False, include_input=False)
            ]
            salida.append(FilaLote(indice=i, ok=False, errores=errores))

    if validas:
        probas = servido.probabilidades([e for _, e in validas])
        salida += [
            FilaLote(indice=i, ok=True, resultado=_respuesta(servido, float(p)))
            for (i, _), p in zip(validas, probas, strict=True)
        ]
    salida.sort(key=lambda f: f.indice)
    n_ok = len(validas)
    return {"n_filas": len(filas), "n_ok": n_ok, "n_error": len(filas) - n_ok, "filas": salida}


FilasLote = Annotated[
    list[Any],
    Body(
        min_length=1,
        max_length=MAX_FILAS_LOTE,
        description=f"Lista JSON de filas (máximo {MAX_FILAS_LOTE}). Cada fila, con el esquema del endpoint individual.",
    ),
]


# =======================================================
# Aplicación
# =======================================================
def create_app(cargar_artefactos: bool = True) -> FastAPI:
    """Fábrica. ``cargar_artefactos=False`` arranca sin modelos (tests de contrato)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if cargar_artefactos:
            config = Config.desde_entorno()
            registro = cargar_registro(config.manifest_path, config.model_dir)
            for s in registro.modelos.values():
                log.info("modelo cargado: %s (%s) sha256=%s…", s.experimento, s.algoritmo, s.sha256[:16])
            app.state.registro = registro
        yield

    app = FastAPI(
        title="API de riesgo cardiovascular e hipertensión (DT-1)",
        version=API_VERSION,
        description=(
            "Probabilidades estimadas por dos modelos entrenados sobre el *Cardiovascular Disease "
            "Dataset* (Kaggle, 68.678 filas tras limpieza).\n\n"
            "* **A′ — riesgo cardiovascular** (con presión arterial): modelo principal.\n"
            "* **B1 — hipertensión sin presión arterial**: experimental; ordena riesgo, no clasifica.\n\n"
            f"⚠ {AVISO}"
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def error_422(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Mismo formato que los errores por fila del lote. Sin eco de la entrada:
        # un lote de 1.001 filas no debe volver entero en el mensaje de error.
        detalle = [
            ErrorCampo(loc=list(e["loc"]), msg=e["msg"], type=e["type"]).model_dump() for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": detalle})

    @app.get("/", include_in_schema=False)
    def raiz() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health", tags=["servicio"])
    def health(registro: RegistroDep) -> dict[str, Any]:
        """Estado, versión y huella de cada modelo cargado."""
        return {
            "estado": "ok",
            "version_api": API_VERSION,
            "modelos": [s.info().model_dump() for s in registro.modelos.values()],
            "entorno": entorno_en_ejecucion(),
        }

    @app.get("/v1/modelos", tags=["servicio"])
    def modelos(registro: RegistroDep) -> dict[str, Any]:
        """Metadatos del manifiesto: entrada esperada, prevalencia y métricas de test."""
        catalogo: list[dict[str, Any]] = []
        for servido, esquema, endpoint, rol in (
            (registro.modelos[RIESGO_CV], EntradaRiesgoCV, "/v1/riesgo-cardiovascular", "principal"),
            (registro.modelos[HTA_B1], EntradaHTASinPA, "/v1/hipertension-sin-pa", "experimental"),
        ):
            schema = esquema.model_json_schema()
            bloque = servido.bloque
            catalogo.append(
                {
                    "experimento": servido.experimento,
                    "rol": rol,
                    "endpoint": endpoint,
                    "endpoint_lote": f"{endpoint}/lote",
                    "target": bloque["target"],
                    "definicion_target": bloque.get("umbral", {}).get(
                        "definicion", "cardio: enfermedad cardiovascular, etiqueta original del dataset"
                    ),
                    "algoritmo": servido.algoritmo,
                    "sha256": servido.sha256,
                    "entrada": {"campos": schema["properties"], "requeridos": schema["required"]},
                    "derivadas_en_servidor": {
                        "bmi": f"weight / (height/100)^2; debe quedar en [{BMI_RANGE[0]:g}, {BMI_RANGE[1]:g}] kg/m²"
                    },
                    "orden_features_modelo": servido.features,
                    "prevalencia_base": servido.prevalencia_base,
                    "metricas_test": metricas_test(servido),
                    "devuelve_clase": servido.experimento == RIESGO_CV,
                    "umbral_clase": UMBRAL_CLASE if servido.experimento == RIESGO_CV else None,
                    "advertencia": advertencia_b1(servido) if servido.experimento == HTA_B1 else None,
                }
            )
        return {
            "modelos": catalogo,
            "max_filas_lote": MAX_FILAS_LOTE,
            "nota_metricas": (
                "Métricas del ganador sobre el test de DT-1. La probabilidad no lleva calibración "
                "post-hoc; su calibración se mide con ECE (10 bins uniformes) en ese test. El mismo "
                "test eligió al ganador entre 5 algoritmos (DT-4 abierto): las cifras están "
                "sesgadas al alza por selección."
            ),
            "aviso": AVISO,
        }

    @app.post(
        "/v1/riesgo-cardiovascular", response_model=RespuestaRiesgoCV, tags=["A′ riesgo cardiovascular"]
    )
    def riesgo_cardiovascular(entrada: EntradaRiesgoCV, registro: RegistroDep) -> Any:
        """Probabilidad de enfermedad cardiovascular (A′, con presión arterial)."""
        return _predecir_uno(registro.modelos[RIESGO_CV], entrada)

    @app.post(
        "/v1/riesgo-cardiovascular/lote",
        response_model=RespuestaLote[RespuestaRiesgoCV],
        tags=["A′ riesgo cardiovascular"],
    )
    def riesgo_cardiovascular_lote(filas: FilasLote, registro: RegistroDep) -> Any:
        """Lote de hasta 1.000 filas; las inválidas devuelven su error por índice."""
        return _predecir_lote(registro.modelos[RIESGO_CV], EntradaRiesgoCV, filas)

    @app.post("/v1/hipertension-sin-pa", response_model=RespuestaHTASinPA, tags=["B1 experimental"])
    def hipertension_sin_pa(entrada: EntradaHTASinPA, registro: RegistroDep) -> Any:
        """Probabilidad de hipertensión SIN presión arterial (B1, experimental; sin clase)."""
        return _predecir_uno(registro.modelos[HTA_B1], entrada)

    @app.post(
        "/v1/hipertension-sin-pa/lote",
        response_model=RespuestaLote[RespuestaHTASinPA],
        tags=["B1 experimental"],
    )
    def hipertension_sin_pa_lote(filas: FilasLote, registro: RegistroDep) -> Any:
        """Lote de hasta 1.000 filas; las inválidas devuelven su error por índice."""
        return _predecir_lote(registro.modelos[HTA_B1], EntradaHTASinPA, filas)

    return app


app = create_app()
