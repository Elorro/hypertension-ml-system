"""Pruebas de la explicación de probabilidad sin importar Streamlit."""

import pytest
from explicacion import explicar_probabilidad


@pytest.mark.parametrize(
    ("p", "banda", "puntos"),
    [
        (0.544, "cercana al promedio", "+4 puntos"),
        (0.546, "por encima del promedio", "+5 puntos"),
        (0.456, "cercana al promedio", "-4 puntos"),
        (0.454, "por debajo del promedio", "-5 puntos"),
    ],
)
def test_bandas_y_bordes(p, banda, puntos):
    texto = explicar_probabilidad(p, 0.5, "riesgo_cv_con_pa")
    assert banda in texto
    assert puntos in texto


@pytest.mark.parametrize(
    ("experimento", "espera"),
    [
        ("riesgo_cv_con_pa", "etiqueta de enfermedad cardiovascular del dataset"),
        ("hta_b1", "≥140 sistólica o ≥90 diastólica"),
    ],
)
def test_texto_de_cada_experimento_y_porcentajes(experimento, espera):
    texto = explicar_probabilidad(0.734, 0.412, experimento)
    assert "73%" in texto and "41%" in texto
    assert espera in texto
    assert "+32 puntos" in texto
    if experimento == "hta_b1":
        assert "de que la presión de esta persona alcance" in texto


@pytest.mark.parametrize("experimento", ["riesgo_cv_con_pa", "hta_b1"])
def test_no_incluye_terminos_prohibidos(experimento):
    texto = explicar_probabilidad(0.6, 0.4, experimento).casefold()
    for prohibida in (
        "propenso",
        "padece",
        "sano",
        "enfermo",
        "tiene la enfermedad",
        "no tiene la enfermedad",
    ):
        assert prohibida not in texto


def test_rechaza_experimento_desconocido():
    with pytest.raises(ValueError):
        explicar_probabilidad(0.5, 0.4, "otro")
