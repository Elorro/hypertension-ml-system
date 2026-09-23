"""Dashboard Streamlit: cliente HTTP puro de la API de DT-1.

No importa scikit-learn ni carga modelos. Todo lo que muestra —campos, límites,
etiquetas de los códigos, métricas— viene de ``GET /v1/modelos`` y de las
respuestas de la API; nada del dominio se repite aquí.

Configuración por variables de entorno:

* ``API_URL`` — base de la API (por defecto ``http://127.0.0.1:8000``).
* ``API_WAIT_SECONDS`` — cuánto esperar a que la API responda al arrancar
  (por defecto 75 s: un plan gratuito puede tardar ~1 min en despertar).

Uso::

    API_URL=http://127.0.0.1:8000 streamlit run app/dashboard.py

AVISO: demostración de ingeniería de ML, no herramienta clínica.
"""

from __future__ import annotations

import io
import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL: str = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
ESPERA_MAX_S: float = float(os.environ.get("API_WAIT_SECONDS", "75"))
TIMEOUT_PING_S: float = 5.0
INTERVALO_S: float = 3.0
TIMEOUT_PRED_S: float = 60.0

AVISO_LOCAL: str = (
    "⚠ Demostración de ingeniería de ML, **no herramienta clínica**. Ninguna cifra de esta "
    "página es un diagnóstico ni sustituye la medición de la presión arterial ni la valoración "
    "de un profesional de salud."
)


# =======================================================
# Cliente HTTP
# =======================================================
class APINoDisponible(Exception):
    """La API no respondió (caída, dormida o URL incorrecta)."""


def _get(ruta: str, timeout: float) -> dict[str, Any]:
    try:
        r = requests.get(f"{API_URL}{ruta}", timeout=timeout)
    except requests.RequestException as exc:
        raise APINoDisponible(f"{type(exc).__name__}: {exc}") from exc
    r.raise_for_status()
    return r.json()


def _post(ruta: str, cuerpo: Any) -> requests.Response:
    try:
        return requests.post(f"{API_URL}{ruta}", json=cuerpo, timeout=TIMEOUT_PRED_S)
    except requests.RequestException as exc:
        raise APINoDisponible(f"{type(exc).__name__}: {exc}") from exc


def esperar_api() -> tuple[dict[str, Any] | None, str]:
    """Hace ping a /health con reintentos hasta ESPERA_MAX_S. Devuelve (health, último error)."""
    limite = time.monotonic() + ESPERA_MAX_S
    intento, ultimo_error = 0, ""
    with st.status("Despertando la API, puede tardar ~1 min en el plan gratuito…", expanded=True) as estado:
        while True:
            intento += 1
            try:
                health = _get("/health", TIMEOUT_PING_S)
                estado.update(label="API disponible", state="complete", expanded=False)
                return health, ""
            except (APINoDisponible, requests.HTTPError) as exc:
                ultimo_error = str(exc)
                restante = limite - time.monotonic()
                if restante <= 0:
                    st.write(f"Intento {intento}: sin respuesta de {API_URL}. Sin más reintentos.")
                    estado.update(label="La API no respondió", state="error", expanded=False)
                    return None, ultimo_error
                st.write(f"Intento {intento}: sin respuesta de {API_URL}. Reintentando…")
                time.sleep(min(INTERVALO_S, restante))


@st.cache_data(ttl=600, show_spinner=False)
def catalogo_modelos(url: str) -> dict[str, Any]:
    """/v1/modelos, cacheado por URL. ``url`` es la clave de la caché."""
    return _get("/v1/modelos", TIMEOUT_PING_S)


# =======================================================
# Presentación
# =======================================================
def _pct(x: float) -> str:
    return f"{x:.1%}".replace(".", ",")


def _dec(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def _errores_legibles(errores: list[dict[str, Any]], campos: dict[str, Any]) -> list[str]:
    """``loc`` vacío = regla sobre varios campos (IMC, ap_lo < ap_hi)."""
    salida = []
    for e in errores:
        loc = [p for p in e.get("loc", []) if p != "body"]
        campo = campos.get(str(loc[-1]), {}).get("title", loc[-1]) if loc else "entrada"
        salida.append(f"{campo}: {e['msg'].removeprefix('Value error, ')}")
    return salida


def formulario(modelo: dict[str, Any], prefijo: str) -> dict[str, Any]:
    """Un widget por campo requerido, con límites, ejemplo y ayuda del JSON Schema."""
    campos = modelo["entrada"]["campos"]
    valores: dict[str, Any] = {}
    # Magnitudes a la izquierda, códigos a la derecha; orden interno = el del esquema.
    col_num, col_cod = st.columns(2)
    for nombre in modelo["entrada"]["requeridos"]:
        spec = campos[nombre]
        titulo = spec.get("title", nombre)
        ejemplo = spec.get("examples", [None])[0]
        with col_cod if "enum" in spec else col_num:
            if "enum" in spec:
                etiquetas = spec.get("x-etiquetas", {})
                opciones = list(spec["enum"])
                valores[nombre] = st.selectbox(
                    titulo,
                    opciones,
                    index=opciones.index(ejemplo) if ejemplo in opciones else 0,
                    format_func=lambda v, et=etiquetas: f"{v} — {et[str(v)]}" if str(v) in et else str(v),
                    help=spec.get("description"),
                    key=f"{prefijo}_{nombre}",
                )
                if any("inferid" in e for e in etiquetas.values()):
                    st.caption(spec.get("description", ""))
            else:
                valores[nombre] = st.number_input(
                    titulo,
                    min_value=float(spec["minimum"]),
                    max_value=float(spec["maximum"]),
                    value=float(ejemplo if ejemplo is not None else spec["minimum"]),
                    step=1.0,
                    help=spec.get("description"),
                    key=f"{prefijo}_{nombre}",
                )
    for derivada, regla in modelo.get("derivadas_en_servidor", {}).items():
        st.caption(f"`{derivada}` se calcula en el servidor: {regla}.")
    return valores


def mostrar_resultado(resultado: dict[str, Any]) -> None:
    p, base = resultado["probabilidad"], resultado["prevalencia_base"]
    c1, c2 = st.columns([2, 1])
    c1.metric(
        "Probabilidad estimada",
        _pct(p),
        delta=f"{(p - base) * 100:+.1f} pp frente a la prevalencia base".replace(".", ","),
        delta_color="off",
    )
    c2.metric("Prevalencia base (train)", _pct(base))
    st.progress(min(max(p, 0.0), 1.0))
    if "clase" in resultado:
        st.write(
            f"Clase con umbral {_dec(resultado['umbral'], 1)} explícito: **{resultado['clase']}** "
            "(secundaria; la probabilidad es la salida principal)."
        )
    m = resultado["modelo"]
    st.caption(f"Modelo {m['experimento']} · {m['algoritmo']} · sha256 {m['sha256'][:12]}…")
    st.info(resultado["aviso"])


def predecir_individual(modelo: dict[str, Any], prefijo: str) -> None:
    with st.form(f"form_{prefijo}"):
        valores = formulario(modelo, prefijo)
        enviado = st.form_submit_button("Calcular probabilidad")
    if not enviado:
        return
    try:
        r = _post(modelo["endpoint"], valores)
    except APINoDisponible as exc:
        st.session_state.pop("health", None)
        st.error(f"La API dejó de responder en {API_URL}. Recarga la página para reintentar. ({exc})")
        return
    if r.status_code == 422:
        st.error("La entrada está fuera del dominio de entrenamiento:")
        for linea in _errores_legibles(r.json()["detail"], modelo["entrada"]["campos"]):
            st.write(f"- {linea}")
        return
    r.raise_for_status()
    mostrar_resultado(r.json())


# =======================================================
# Lote por CSV
# =======================================================
def plantilla_csv(modelo: dict[str, Any]) -> bytes:
    campos = modelo["entrada"]["campos"]
    fila = {c: campos[c].get("examples", [""])[0] for c in modelo["entrada"]["requeridos"]}
    return pd.DataFrame([fila]).to_csv(index=False).encode()


def filas_json(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Filas del CSV como dicts JSON nativos; las celdas vacías se omiten (la API dirá qué falta)."""
    registros = df.astype(object).where(df.notna(), None).to_dict("records")
    return [{str(k): v for k, v in r.items() if v is not None} for r in registros]


def predecir_lote(modelo: dict[str, Any], df: pd.DataFrame, max_filas: int) -> pd.DataFrame:
    filas = filas_json(df)
    probabilidades: list[float | None] = [None] * len(filas)
    clases: list[int | None] = [None] * len(filas)
    errores: list[str] = [""] * len(filas)
    campos = modelo["entrada"]["campos"]
    for inicio in range(0, len(filas), max_filas):
        r = _post(f"{modelo['endpoint_lote']}", filas[inicio : inicio + max_filas])
        r.raise_for_status()
        for f in r.json()["filas"]:
            i = inicio + f["indice"]
            if f["ok"]:
                probabilidades[i] = f["resultado"]["probabilidad"]
                clases[i] = f["resultado"].get("clase")
            else:
                errores[i] = "; ".join(_errores_legibles(f["errores"], campos))
    salida = df.copy()
    salida["probabilidad"] = probabilidades
    if modelo["devuelve_clase"]:
        salida["clase"] = pd.array(clases, dtype="Int64")
    salida["error"] = errores
    return salida


def seccion_lote(por_exp: dict[str, dict[str, Any]], max_filas: int) -> None:
    st.header("Lote por CSV")
    nombres = {
        "riesgo_cv_con_pa": "A′ — riesgo cardiovascular (con presión)",
        "hta_b1": "B1 — hipertensión sin presión (experimental)",
    }
    exp = st.radio("Modelo", list(por_exp), format_func=lambda e: nombres.get(e, e), horizontal=True)
    modelo = por_exp[exp]
    columnas = modelo["entrada"]["requeridos"]
    st.write(
        f"Columnas, en unidades humanas: `{'`, `'.join(columnas)}`. Máximo {max_filas} filas por "
        "llamada (el dashboard parte archivos mayores). Una fila inválida muestra su error y no "
        "afecta al resto; una columna no prevista (p. ej. `ap_hi` en B1) invalida la fila."
    )
    st.download_button(
        "Descargar plantilla CSV",
        plantilla_csv(modelo),
        file_name=f"plantilla_{exp}.csv",
        mime="text/csv",
    )
    archivo = st.file_uploader("Subir CSV", type=["csv"], key=f"csv_{exp}")
    if archivo is None:
        return
    df = pd.read_csv(archivo)
    if df.empty:
        st.warning("El CSV no tiene filas.")
        return
    try:
        resultado = predecir_lote(modelo, df, max_filas)
    except APINoDisponible as exc:
        st.session_state.pop("health", None)
        st.error(f"La API dejó de responder en {API_URL}. Recarga la página para reintentar. ({exc})")
        return
    n_err = int((resultado["error"] != "").sum())
    st.write(f"**{len(resultado) - n_err}** filas con probabilidad · **{n_err}** con error")
    st.dataframe(resultado, width="stretch")
    buffer = io.StringIO()
    resultado.to_csv(buffer, index=False)
    st.download_button(
        "Descargar resultados", buffer.getvalue().encode(), f"resultados_{exp}.csv", "text/csv"
    )


# =======================================================
# Métricas
# =======================================================
def seccion_metricas(catalogo: dict[str, Any]) -> None:
    st.header("Métricas de test (desde /v1/modelos)")
    st.warning(catalogo["nota_metricas"])
    filas = []
    for m in catalogo["modelos"]:
        t = m["metricas_test"]
        lo, hi = t["roc_auc_ic95_bootstrap"]
        filas.append(
            {
                "modelo": f"{m['experimento']} ({m['rol']})",
                "algoritmo": m["algoritmo"],
                "AUC [IC 95 %]": f"{_dec(t['roc_auc'])} [{_dec(lo)} · {_dec(hi)}]",
                "Brier (baseline)": f"{_dec(t['brier'])} ({_dec(t['brier_baseline_prevalencia'])})",
                "probabilidad": f"ECE medido en test: {_dec(t['ece_uniform_10'], 4)}",
                "prevalencia test": _pct(t["prevalencia_test"]),
                "n test": t["n_test"],
            }
        )
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")


# =======================================================
# Página
# =======================================================
def main() -> None:
    st.set_page_config(page_title="Riesgo cardiovascular — DT-1", page_icon="🩺", layout="wide")
    st.title("🩺 Riesgo cardiovascular e hipertensión — demostración")
    st.markdown(AVISO_LOCAL)

    if "health" not in st.session_state:
        health, error = esperar_api()
        if health is None:
            st.error(
                f"No se pudo contactar la API en `{API_URL}` tras {ESPERA_MAX_S:.0f} s de reintentos. "
                "Puede estar apagada, dormida o la URL (variable API_URL) ser incorrecta.\n\n"
                f"Último error: `{error}`"
            )
            st.button("Reintentar")
            st.stop()
        st.session_state["health"] = health
    health = st.session_state["health"]

    try:
        catalogo = catalogo_modelos(API_URL)
    except (APINoDisponible, requests.HTTPError) as exc:
        st.session_state.pop("health", None)
        st.error(f"No se pudo leer /v1/modelos de `{API_URL}`: {exc}")
        st.button("Reintentar")
        st.stop()
    por_exp = {m["experimento"]: m for m in catalogo["modelos"]}

    with st.sidebar:
        st.markdown(f"**API** `{API_URL}` · versión {health['version_api']}")
        for m in health["modelos"]:
            st.caption(f"{m['experimento']}: {m['algoritmo']} · sha256 {m['sha256'][:12]}…")

    tab_a, tab_b1, tab_lote, tab_metricas = st.tabs(
        ["A′ Riesgo cardiovascular", "B1 Hipertensión sin PA (experimental)", "Lote CSV", "Métricas"]
    )

    with tab_a:
        a = por_exp["riesgo_cv_con_pa"]
        st.header("A′ — Riesgo cardiovascular (modelo principal)")
        auc = a["metricas_test"]["roc_auc"]
        st.write(
            f"Probabilidad de enfermedad cardiovascular (`{a['definicion_target']}`), con presión "
            f"arterial. AUC en test ≈ {_dec(auc, 2)}; probabilidad (ECE medido en test: "
            f"{_dec(a['metricas_test']['ece_uniform_10'], 4)})."
        )
        predecir_individual(a, "a")

    with tab_b1:
        b1 = por_exp["hta_b1"]
        t = b1["metricas_test"]
        st.header("B1 — Hipertensión sin presión arterial · EXPERIMENTAL")
        st.warning(
            f"Sin presión arterial; AUC ≈ {_dec(t['roc_auc'], 2)} "
            f"[{_dec(t['roc_auc_ic95_bootstrap'][0], 2)} · {_dec(t['roc_auc_ic95_bootstrap'][1], 2)}]. "
            "La probabilidad ordena riesgo, no diagnostica."
        )
        with st.expander("Por qué no hay clase"):
            st.write(b1["advertencia"])
        predecir_individual(b1, "b1")

    with tab_lote:
        seccion_lote(por_exp, catalogo["max_filas_lote"])

    with tab_metricas:
        seccion_metricas(catalogo)


# streamlit run (y AppTest) ejecutan el script como __main__; importarlo no dibuja nada.
if __name__ == "__main__":
    main()
