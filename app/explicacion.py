"""Texto de explicación de las probabilidades del dashboard."""


def explicar_probabilidad(p: float, base: float, experimento: str) -> str:
    """Explica una probabilidad frente al promedio del dataset de entrenamiento."""
    pp = round(p * 100)
    bp = round(base * 100)
    d = pp - bp
    if abs(d) < 5:
        banda = "cercana al promedio"
    elif d >= 5:
        banda = "por encima del promedio"
    else:
        banda = "por debajo del promedio"

    if experimento == "riesgo_cv_con_pa":
        return (
            f"El modelo estima una probabilidad de {pp}% de que esta persona tenga la etiqueta de "
            f"enfermedad cardiovascular del dataset. Entre las personas del dataset de entrenamiento "
            f"el promedio es {bp}%; esta estimación está {banda} ({d:+d} puntos). Es una "
            "estimación estadística, no un diagnóstico."
        )
    if experimento == "hta_b1":
        return (
            f"El modelo estima una probabilidad de {pp}% de que la presión de esta persona alcance el umbral de "
            "hipertensión (≥140 sistólica o ≥90 diastólica), sin medirla. "
            f"Promedio del dataset: {bp}%; está {banda} ({d:+d} puntos). "
            "Esta cifra sirve para ordenar el riesgo, no para decidir: el modelo no detecta de forma "
            "fiable quién tiene hipertensión."
        )
    raise ValueError(f"Experimento no reconocido: {experimento}")
