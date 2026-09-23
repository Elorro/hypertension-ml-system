"""Fuente única de verdad de las features y el dominio de DT-1.

Lo importan ``src/train_cardio_real.py`` (entrenamiento) y ``api/`` (servicio):
el orden de features, su derivación y las cotas del dominio de entrenamiento
viven aquí y en ningún otro sitio del camino entrenamiento → servicio.

Excepción deliberada: ``scripts/audit_cardio_leakage.py`` reimplementa la
limpieza a propósito, para que la auditoría sea una verificación independiente
y no herede un error de este módulo. La coincidencia se comprueba contra el
manifiesto.

Solo usa la biblioteca estándar: las funciones son genéricas sobre ``float``,
``numpy.ndarray`` y ``pandas.Series`` (aritmética y comparaciones elemento a
elemento), así que el servicio no necesita pandas para derivar features.
"""

from __future__ import annotations

from typing import Any

# =======================================================
# Cotas de limpieza (aplicadas al entrenar y exigidas por la API)
# =======================================================
# Criterio: imposibilidad fisiológica en adultos de 30-65 años, no recorte de
# outliers. Límites inclusivos (equivalentes a ``Series.between``).
HEIGHT_CM_RANGE: tuple[float, float] = (120.0, 220.0)
WEIGHT_KG_RANGE: tuple[float, float] = (30.0, 200.0)
BMI_RANGE: tuple[float, float] = (12.0, 70.0)

# Rangos de presión plausibles. En el entrenamiento NO limpian (solo alimentan el
# análisis de sensibilidad sobre test); la API los exige como dominio de entrada.
AP_HI_PLAUSIBLE: tuple[float, float] = (70.0, 250.0)
AP_LO_PLAUSIBLE: tuple[float, float] = (40.0, 200.0)

# Rango de edad de servicio. No es cota de limpieza: el CSV limpio va de 29,56 a
# 64,92 años (age / 365.25, 68.678 filas). Se redondea hacia afuera para no
# excluir edades vistas. Fuera de él un árbol no extrapola: aplana la predicción
# al valor del extremo, así que la API rechaza en vez de avisar.
AGE_YEARS_RANGE: tuple[float, float] = (29.0, 65.0)

DIAS_POR_ANIO: float = 365.25

# =======================================================
# Codificaciones
# =======================================================
# Valores que toma cada columna categórica en el CSV limpio.
CODIGOS: dict[str, tuple[int, ...]] = {
    "gender": (1, 2),
    "cholesterol": (1, 2, 3),
    "gluc": (1, 2, 3),
    "smoke": (0, 1),
    "alco": (0, 1),
    "active": (0, 1),
}

# =======================================================
# Features
# =======================================================
PRESSURE_COLUMNS: frozenset[str] = frozenset({"ap_hi", "ap_lo"})
FORBIDDEN_B1: frozenset[str] = PRESSURE_COLUMNS | {"cardio", "id", "hta", "hta_gt"}

# Orden de features: contrato con el manifiesto y con la API.
FEATURES_BASE: list[str] = [
    "age_years",
    "gender",
    "height",
    "weight",
    "bmi",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]
FEATURES_A_CON_PA: list[str] = FEATURES_BASE + ["ap_hi", "ap_lo"]
FEATURES_A_SIN_PA: list[str] = list(FEATURES_BASE)
FEATURES_B1: list[str] = list(FEATURES_BASE)

# Linaje de cada feature: columnas crudas de las que se calcula. Es lo que permite
# verificar estructuralmente que ninguna feature de B1 deriva de la presión.
FEATURE_LINEAGE: dict[str, set[str]] = {
    "age_years": {"age"},
    "gender": {"gender"},
    "height": {"height"},
    "weight": {"weight"},
    "bmi": {"weight", "height"},
    "cholesterol": {"cholesterol"},
    "gluc": {"gluc"},
    "smoke": {"smoke"},
    "alco": {"alco"},
    "active": {"active"},
    "ap_hi": {"ap_hi"},
    "ap_lo": {"ap_lo"},
}


# =======================================================
# Derivación
# =======================================================
def age_years_from_days(age_days: Any) -> Any:
    """Edad en años desde días, tal como la vio el modelo: ``age / 365.25``."""
    return age_days / DIAS_POR_ANIO


def bmi(weight_kg: Any, height_cm: Any) -> Any:
    """IMC en kg/m². No cambiar el orden de operaciones: fija los bits del float."""
    return weight_kg / (height_cm / 100.0) ** 2


def en_rango(valor: Any, rango: tuple[float, float]) -> Any:
    """``lo <= valor <= hi``; con Series/ndarray devuelve la máscara elemento a elemento."""
    lo, hi = rango
    return (valor >= lo) & (valor <= hi)
