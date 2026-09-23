"""Esquemas Pydantic del servicio: entrada en unidades humanas, dominio = entrenamiento.

Los nombres de campo son los del dataset (``gender``, ``height``…) salvo la edad,
que llega en años (``age_years``, la feature exacta) y no en días. ``bmi`` no es
un campo: lo calcula el servidor y, como todo campo no declarado, enviarlo
devuelve 422 (``extra="forbid"``). En B1 eso mismo convierte ``ap_hi``/``ap_lo``
en 422: el contrato anti-leakage aplicado en la API.

Todas las cotas vienen de ``src/cardio_features.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.cardio_features import (
    AGE_YEARS_RANGE,
    AP_HI_PLAUSIBLE,
    AP_LO_PLAUSIBLE,
    BMI_RANGE,
    HEIGHT_CM_RANGE,
    WEIGHT_KG_RANGE,
    bmi,
    en_rango,
)

AVISO: str = (
    "Demostración de ingeniería de ML, no herramienta clínica. Esta probabilidad no es un "
    "diagnóstico ni sustituye la medición de la presión arterial ni la valoración de un "
    "profesional de salud."
)

MAX_FILAS_LOTE: int = 1000
UMBRAL_CLASE: float = 0.5

# Cita literal de dataset.codificacion_gender del manifiesto. Un test verifica que
# coincide con el manifiesto servido.
CODIFICACION_GENDER_MANIFIESTO: str = "1/2 crudo; 2 = hombre (inferido: talla media 169.9 vs 161.4 cm)"


def _rango(r: tuple[float, float]) -> str:
    return f"[{r[0]:g}, {r[1]:g}]"


# Etiquetas legibles de cada código. Fuente única: generan la description y se
# publican como ``x-etiquetas`` en el JSON Schema (/v1/modelos, /openapi.json), que es
# lo que usa el dashboard. Las de gender son inferidas (ver CODIFICACION_GENDER_MANIFIESTO).
ETIQUETAS: dict[str, dict[int, str]] = {
    "gender": {1: "mujer (inferido)", 2: "hombre (inferido)"},
    "cholesterol": {1: "normal", 2: "elevado", 3: "muy elevado"},
    "gluc": {1: "normal", 2: "elevada", 3: "muy elevada"},
    "smoke": {0: "no", 1: "sí"},
    "alco": {0: "no", 1: "sí"},
    "active": {0: "no", 1: "sí"},
}


def _codigos(campo: str) -> str:
    return ", ".join(f"{k} = {v}" for k, v in ETIQUETAS[campo].items())


def _extra(campo: str) -> dict[str, dict[str, dict[str, str]]]:
    return {"x-etiquetas": {str(k): v for k, v in ETIQUETAS[campo].items()}}


# =======================================================
# Entrada
# =======================================================
class EntradaBase(BaseModel):
    """Features comunes a A′ y B1."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    age_years: float = Field(
        title="Edad (años)",
        ge=AGE_YEARS_RANGE[0],
        le=AGE_YEARS_RANGE[1],
        description=(
            f"Edad en años (admite decimales). Rango {_rango(AGE_YEARS_RANGE)}: el de entrenamiento "
            "(29,56–64,92 años) redondeado hacia afuera. Fuera de él el modelo no extrapola y se rechaza."
        ),
        examples=[52.0],
    )
    gender: Literal[1, 2] = Field(
        title="Sexo (código)",
        description=(
            "Sexo, código crudo del dataset (1 | 2). Manifiesto, dataset.codificacion_gender: "
            f"«{CODIFICACION_GENDER_MANIFIESTO}». La correspondencia es inferida, no documentada "
            "por la fuente."
        ),
        examples=[1],
        json_schema_extra=_extra("gender"),
    )
    height: float = Field(
        title="Talla (cm)",
        ge=HEIGHT_CM_RANGE[0],
        le=HEIGHT_CM_RANGE[1],
        description=f"Talla en cm. Cota de limpieza del entrenamiento {_rango(HEIGHT_CM_RANGE)} cm.",
        examples=[165.0],
    )
    weight: float = Field(
        title="Peso (kg)",
        ge=WEIGHT_KG_RANGE[0],
        le=WEIGHT_KG_RANGE[1],
        description=(
            f"Peso en kg. Cota de limpieza del entrenamiento {_rango(WEIGHT_KG_RANGE)} kg. "
            f"El IMC resultante (calculado en el servidor) debe estar en {_rango(BMI_RANGE)} kg/m²."
        ),
        examples=[72.0],
    )
    cholesterol: Literal[1, 2, 3] = Field(
        title="Colesterol",
        description=f"Colesterol, ordinal del dataset: {_codigos('cholesterol')}.",
        examples=[1],
        json_schema_extra=_extra("cholesterol"),
    )
    gluc: Literal[1, 2, 3] = Field(
        title="Glucosa",
        description=f"Glucosa, ordinal del dataset: {_codigos('gluc')}.",
        examples=[1],
        json_schema_extra=_extra("gluc"),
    )
    smoke: Literal[0, 1] = Field(
        title="Fuma",
        description=f"Fuma: {_codigos('smoke')} (autorreportado).",
        examples=[0],
        json_schema_extra=_extra("smoke"),
    )
    alco: Literal[0, 1] = Field(
        title="Consume alcohol",
        description=f"Consume alcohol: {_codigos('alco')} (autorreportado).",
        examples=[0],
        json_schema_extra=_extra("alco"),
    )
    active: Literal[0, 1] = Field(
        title="Físicamente activo",
        description=f"Físicamente activo: {_codigos('active')} (autorreportado).",
        examples=[1],
        json_schema_extra=_extra("active"),
    )

    @property
    def imc(self) -> float:
        """IMC calculado en el servidor, con la misma fórmula que vio el modelo."""
        return bmi(self.weight, self.height)

    @model_validator(mode="after")
    def _imc_en_dominio(self) -> EntradaBase:
        if not en_rango(self.imc, BMI_RANGE):
            raise ValueError(
                f"IMC calculado {self.imc:.2f} kg/m² (weight / (height/100)²) fuera de la cota de "
                f"entrenamiento {_rango(BMI_RANGE)} kg/m²"
            )
        return self


class EntradaRiesgoCV(EntradaBase):
    """A′: riesgo cardiovascular, con presión arterial."""

    ap_hi: float = Field(
        title="PAS (mmHg)",
        ge=AP_HI_PLAUSIBLE[0],
        le=AP_HI_PLAUSIBLE[1],
        description=f"Presión sistólica (PAS) en mmHg. Rango plausible {_rango(AP_HI_PLAUSIBLE)}.",
        examples=[130.0],
    )
    ap_lo: float = Field(
        title="PAD (mmHg)",
        ge=AP_LO_PLAUSIBLE[0],
        le=AP_LO_PLAUSIBLE[1],
        description=(
            f"Presión diastólica (PAD) en mmHg. Rango plausible {_rango(AP_LO_PLAUSIBLE)}; "
            "debe ser menor que ap_hi."
        ),
        examples=[85.0],
    )

    @model_validator(mode="after")
    def _presion_ordenada(self) -> EntradaRiesgoCV:
        if not self.ap_lo < self.ap_hi:
            raise ValueError(
                f"ap_lo ({self.ap_lo:g}) debe ser menor que ap_hi ({self.ap_hi:g}); el entrenamiento "
                "descartó las filas con presión invertida"
            )
        return self


class EntradaHTASinPA(EntradaBase):
    """B1: hipertensión estimada SIN presión arterial. Enviar ap_hi/ap_lo es un 422."""


# =======================================================
# Salida
# =======================================================
class InfoModelo(BaseModel):
    experimento: str = Field(examples=["riesgo_cv_con_pa"])
    algoritmo: str = Field(examples=["RandomForest"])
    sha256: str = Field(
        description="sha256 del .pkl del modelo, verificado contra el manifiesto al arrancar."
    )


class RespuestaHTASinPA(BaseModel):
    probabilidad: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidad estimada de la clase positiva. No es un diagnóstico.",
    )
    prevalencia_base: float = Field(
        description="Prevalencia de la clase positiva en el train del modelo: referencia sin información."
    )
    modelo: InfoModelo
    aviso: str = Field(default=AVISO)


class RespuestaRiesgoCV(RespuestaHTASinPA):
    clase: Literal[0, 1] = Field(description="1 si probabilidad >= umbral.")
    umbral: float = Field(default=UMBRAL_CLASE, description="Umbral de decisión, fijo en 0,5.")


class ErrorCampo(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class FilaLote[R: RespuestaHTASinPA](BaseModel):
    indice: int = Field(description="Posición de la fila en la lista enviada (desde 0).")
    ok: bool
    resultado: R | None = None
    errores: list[ErrorCampo] | None = None


class RespuestaLote[R: RespuestaHTASinPA](BaseModel):
    n_filas: int
    n_ok: int
    n_error: int
    filas: list[FilaLote[R]]
    aviso: str = Field(default=AVISO)
