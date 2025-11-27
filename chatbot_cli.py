from typing import Dict

from data_pipeline import FEATURE_COLUMNS
from model_utils import LABEL_MAP, load_model_bundle, predict_single


def ask(prompt: str, cast=float, default=None):
    raw = input(f"{prompt} [{default}]: ") or default
    if raw is None:
        raise ValueError("Se requieren datos para continuar.")
    try:
        return cast(raw)
    except Exception:
        print("Entrada inválida, intenta de nuevo.")
        return ask(prompt, cast, default)


def collect_inputs() -> Dict:
    print("Responderé con valores por defecto si presionas Enter.")
    datos = {
        "Edad": ask("Edad (años)", int, 45),
        "Sexo": ask("Sexo (0=F,1=M)", int, 1),
        "Peso": ask("Peso (kg)", float, 78.0),
        "Talla": ask("Talla (m)", float, 1.72),
        "PAS": ask("Presión sistólica PAS (mmHg)", float, 135.0),
        "PAD": ask("Presión diastólica PAD (mmHg)", float, 85.0),
        "Frec_Card": ask("Frecuencia cardiaca (lpm)", float, 76.0),
        "Colesterol": ask("Colesterol (mg/dL)", float, 200.0),
        "Glucosa": ask("Glucosa (mg/dL)", float, 105.0),
        "Tabaquismo": ask("Tabaquismo (0/1)", int, 0),
        "Ejercicio": ask("Ejercicio regular (0/1)", int, 1),
        "Estrés": ask("Nivel de estrés 0-10", float, 5.0),
        "Herencia_HTA": ask("Herencia HTA (0/1)", int, 0),
    }
    datos["IMC"] = round(datos["Peso"] / (datos["Talla"] ** 2), 1)
    datos["PAM"] = round((2 * datos["PAD"] + datos["PAS"]) / 3, 1)
    return datos


def main():
    print("=== Chatbot HTA ===")
    bundle = load_model_bundle()
    user_data = collect_inputs()

    # Reordenamos y completamos cualquier falta
    payload = {f: user_data.get(f) for f in FEATURE_COLUMNS}
    result = predict_single(bundle, payload)

    print(f"Diagnóstico: {result['clase']} (clase={result['prediccion']})")
    print("Probabilidades:")
    for clase, prob in result["probabilidades"].items():
        print(f" - {clase}: {prob:.3f}")


if __name__ == "__main__":
    main()
