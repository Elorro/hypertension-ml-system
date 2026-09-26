"""DT-4 — Selección por validación cruzada y una sola evaluación en test.

Implementa ``docs/DT4_PROTOCOL.md`` (preregistrado, commit ``0d8d4c9``). Resumen:

* Misma partición 80/20 que DT-1 (``make_split`` de ``src/train_cardio_real.py``).
* CV estratificada de 5 folds **solo sobre train**, con
  ``Pipeline(StandardScaler, modelo)`` ajustado por fold con sus filas de
  entrenamiento.
* Selección por log-loss media en CV con la regla de un error estándar y un orden
  de simplicidad declarado (la lista de :func:`grilla` ya está en ese orden).
* El seleccionado se reajusta una vez sobre todo el train y se evalúa **una vez**
  en test. Las ablaciones de A′ y la robustez de B1 son evaluaciones preregistradas
  que no intervienen en la selección.

``src/train_cardio_real.py`` y los artefactos ``dt1_*`` no se tocan: son el
registro de DT-1. De ese script se reutilizan la limpieza, la partición y las
métricas, sin copiarlas.

Uso (desde la raíz, con el árbol limpio; la corrida es larga)::

    make train-dt4        # nohup python -m src.train_cardio_dt4 > logs/dt4_run.log

Opciones: ``--desviacion "texto"`` (repetible) registra una desviación del
protocolo en el manifiesto; ``--permitir-arbol-sucio`` solo para depurar: el
manifiesto lo registra y la corrida no vale como preregistrada.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.cardio_features import FEATURES_A_CON_PA, FEATURES_A_SIN_PA, FEATURES_B1
from src.train_cardio_real import (
    DATA_PATH,
    SEED,
    Split,
    assert_no_pressure_lineage,
    bootstrap_auc_ci,
    compute_metrics,
    load_and_clean,
    make_split,
    paired_bootstrap_auc_delta,
    sha256_file,
)

# =======================================================
# Configuración (fijada por el protocolo)
# =======================================================
N_FOLDS: int = 5
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "dt4_manifest.json"
DT1_MANIFEST_PATH = MODELS_DIR / "dt1_manifest.json"
PROTOCOL_PATH = ROOT / "docs" / "DT4_PROTOCOL.md"
# Rutas que deben estar commiteadas para que el commit registrado describa la corrida.
RUTAS_LIMPIAS: tuple[str, ...] = ("src", "docs/DT4_PROTOCOL.md", "requirements.txt", "requirements.lock.txt")
TOL_SPLIT: float = 1e-9

# Hashes de los índices de la partición de DT-1 (protocolo §2). Copia de los
# congelados en tests/test_equivalencia_features.py; un test verifica que coinciden.
HASHES_SPLIT_DT1: dict[str, dict[str, str]] = {
    "cardio": {
        "train_idx": "76fb8ffd6f1134b20ebfc99fca7e0d4723fbac5c4dfa78ad37870caee6b4542a",
        "test_idx": "dc8c2155ab32442dfba55132bcdf4a8a17b55eeaa8c9d766b28233de1b024e7f",
    },
    "hta": {
        "train_idx": "18afc3072cbbe96012bec8676517d59035916fbec58c0c78e325b43392be5d37",
        "test_idx": "1930b665c90eb30a48b77eca7f1b1a55437f761454ddd97e25876e791449c276",
    },
}

FAMILIAS: tuple[str, ...] = ("LogisticRegression", "DecisionTree", "RandomForest", "XGBoost")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =======================================================
# Candidatos
# =======================================================
@dataclass(frozen=True)
class Candidato:
    familia: str
    hiperparametros: dict[str, Any] = field(hash=False)

    @property
    def id(self) -> str:
        hp = ",".join(f"{k}={v}" for k, v in self.hiperparametros.items())
        return f"{self.familia}({hp})"

    def estimador(self) -> ClassifierMixin:
        hp = self.hiperparametros
        if self.familia == "LogisticRegression":
            return LogisticRegression(max_iter=2000, random_state=SEED, **hp)
        if self.familia == "DecisionTree":
            return DecisionTreeClassifier(random_state=SEED, **hp)
        if self.familia == "RandomForest":
            return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, **hp)
        if self.familia == "XGBoost":
            return XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=SEED,
                n_jobs=-1,
                **hp,
            )
        raise ValueError(f"familia desconocida: {self.familia}")


def grilla() -> list[Candidato]:
    """Los 25 candidatos del protocolo (§5), **ya en orden de simplicidad** (§8).

    Por familia: LR < árbol < RF < XGBoost. Dentro: LR por C ascendente; árbol y RF por
    max_depth ascendente y, a igual profundidad, min_samples_leaf descendente; XGBoost
    por max_depth ascendente y, a igual profundidad, min_child_weight descendente.
    """
    lr = [Candidato("LogisticRegression", {"C": c}) for c in (0.01, 0.1, 1.0, 10.0)]
    arbol = [
        Candidato("DecisionTree", {"max_depth": d, "min_samples_leaf": hoja})
        for d in (4, 6, 8)
        for hoja in (200, 50, 1)
    ]
    rf = [
        Candidato("RandomForest", {"max_depth": d, "min_samples_leaf": hoja})
        for d in (8, 12)
        for hoja in (100, 20, 1)
    ]
    xgb = [Candidato("XGBoost", {"max_depth": d, "min_child_weight": w}) for d in (3, 5, 8) for w in (20, 1)]
    return lr + arbol + rf + xgb


# =======================================================
# Validación cruzada (solo filas de train)
# =======================================================
def folds_estratificados(y_train: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y_train)), y_train)]


def ajustar_fold(cand: Candidato, X: np.ndarray, y: np.ndarray, idx_tr: np.ndarray) -> Pipeline:
    """Pipeline(StandardScaler, modelo) ajustado SOLO con las filas ``idx_tr`` del fold."""
    pipe = Pipeline([("scaler", StandardScaler()), ("modelo", cand.estimador())])
    return pipe.fit(X[idx_tr], y[idx_tr])


def evaluar_candidato_cv(
    cand: Candidato, X_train: np.ndarray, y_train: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, Any]:
    """Métricas por fold sobre la parte de validación. Excluido si algún fold falla."""
    por_fold: list[dict[str, Any]] = []
    avisos: list[str] = []
    for k, (tr, va) in enumerate(folds):
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as capturados:
                warnings.simplefilter("always")
                pipe = ajustar_fold(cand, X_train, y_train, tr)
            fit_s = time.perf_counter() - t0
            p_va = pipe.predict_proba(X_train[va])[:, 1]
            p_tr = pipe.predict_proba(X_train[tr])[:, 1]
        except Exception as exc:  # noqa: BLE001 - el protocolo excluye y reporta
            return {"id": cand.id, "excluido": True, "motivo": f"fold {k}: {type(exc).__name__}: {exc}"}
        avisos += [f"fold {k}: {type(w.message).__name__}: {w.message}" for w in capturados]
        m = compute_metrics(y_train[va], p_va, float(y_train[tr].mean()))
        fila = {
            "fold": k,
            "log_loss": m["log_loss"],
            "brier": m["brier"],
            "roc_auc": m["roc_auc"],
            "ece_uniform_10": m["ece_uniform_10"],
            "log_loss_train_fold": float(log_loss(y_train[tr], p_tr, labels=[0, 1])),
            "fit_segundos": round(fit_s, 2),
        }
        if not math.isfinite(fila["log_loss"]):
            return {"id": cand.id, "excluido": True, "motivo": f"fold {k}: log-loss no finita"}
        por_fold.append(fila)

    ll = np.array([f["log_loss"] for f in por_fold])
    sd = float(ll.std(ddof=1))
    return {
        "id": cand.id,
        "familia": cand.familia,
        "hiperparametros": cand.hiperparametros,
        "excluido": False,
        "folds": por_fold,
        "log_loss_media": float(ll.mean()),
        "log_loss_sd": sd,
        "log_loss_ee": sd / math.sqrt(N_FOLDS),
        "avisos": avisos,
    }


def regla_un_ee(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """Regla de 1 EE (§7). ``resultados`` debe venir en orden de simplicidad."""
    validos = [r for r in resultados if not r["excluido"]]
    if not validos:
        raise RuntimeError("todos los candidatos quedaron excluidos")
    mejor = min(validos, key=lambda r: r["log_loss_media"])
    umbral = mejor["log_loss_media"] + mejor["log_loss_ee"]
    elegibles = [r["id"] for r in validos if r["log_loss_media"] <= umbral]
    seleccionado = elegibles[0]  # primero en orden de simplicidad
    return {
        "regla": "1 EE sobre log-loss media en CV; EE aproximado (folds no independientes)",
        "mejor": mejor["id"],
        "mejor_log_loss_media": mejor["log_loss_media"],
        "ee": mejor["log_loss_ee"],
        "umbral": umbral,
        "elegibles": elegibles,
        "seleccionado": seleccionado,
        "motivo": (
            f"{seleccionado} es el más simple de {len(elegibles)} candidatos con log-loss media "
            f"<= {umbral:.6f} (mejor {mejor['id']} = {mejor['log_loss_media']:.6f}, EE {mejor['log_loss_ee']:.6f})"
        ),
    }


# =======================================================
# Reajuste y evaluación en test
# =======================================================
@dataclass(frozen=True)
class ModeloSeleccionado:
    """Solo lo produce :func:`reajustar`: el test se toca con esto y nada más."""

    experimento: str
    candidato: Candidato
    features: list[str]
    scaler: StandardScaler
    modelo: ClassifierMixin
    prevalencia_train: float
    auc_train: float
    log_loss_train: float


def reajustar(
    experimento: str, cand: Candidato, X_train: np.ndarray, y_train: np.ndarray, features: list[str]
) -> ModeloSeleccionado:
    """Reajuste final (§10): el único punto en que un scaler ve todo el train."""
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    modelo = cand.estimador().fit(Xs, y_train)
    p = modelo.predict_proba(Xs)[:, 1]
    return ModeloSeleccionado(
        experimento=experimento,
        candidato=cand,
        features=list(features),
        scaler=scaler,
        modelo=modelo,
        prevalencia_train=float(y_train.mean()),
        auc_train=float(roc_auc_score(y_train, p)),
        log_loss_train=float(log_loss(y_train, p, labels=[0, 1])),
    )


def evaluar_en_test(
    sel: ModeloSeleccionado, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """La única función que usa filas de test. Una llamada por evaluación preregistrada."""
    if not isinstance(sel, ModeloSeleccionado):
        raise TypeError("evaluar_en_test solo acepta un ModeloSeleccionado ya reajustado")
    proba = sel.modelo.predict_proba(sel.scaler.transform(X_test))[:, 1]
    m = compute_metrics(y_test, proba, sel.prevalencia_train)
    m["roc_auc_ic95_bootstrap"] = bootstrap_auc_ci(y_test, proba)
    m["roc_auc_train"] = sel.auc_train
    m["log_loss_train"] = sel.log_loss_train
    return m, proba


# =======================================================
# Orquestación (sin E/S: la usan main() y los tests)
# =======================================================
@dataclass
class Registro:
    """Traza de la corrida: orden de selecciones y evaluaciones en test."""

    eventos: list[tuple[str, str]] = field(default_factory=list)


def filas(
    df: pd.DataFrame, idx: np.ndarray, features: list[str], target: str
) -> tuple[np.ndarray, np.ndarray]:
    sub = df.iloc[idx]
    return sub[features].to_numpy(dtype=float), sub[target].to_numpy()


def seleccionar_experimento(
    nombre: str,
    df: pd.DataFrame,
    split: Split,
    features: list[str],
    target: str,
    candidatos: list[Candidato],
    registro: Registro,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[ModeloSeleccionado, dict[str, Any], list[tuple[np.ndarray, np.ndarray]]]:
    """CV + regla de 1 EE + reajuste. Solo lee las filas de ``split.train_idx``."""
    X_train, y_train = filas(df, split.train_idx, features, target)
    folds = folds if folds is not None else folds_estratificados(y_train)
    log(f"{nombre}: CV de {len(candidatos)} candidatos × {N_FOLDS} folds sobre {len(y_train)} filas")
    resultados = []
    for cand in candidatos:
        r = evaluar_candidato_cv(cand, X_train, y_train, folds)
        resultados.append(r)
        if r["excluido"]:
            log(f"  EXCLUIDO {cand.id}: {r['motivo']}")
        else:
            log(f"  {cand.id:<62} log-loss {r['log_loss_media']:.5f} ± {r['log_loss_sd']:.5f}")
    seleccion = regla_un_ee(resultados)
    registro.eventos.append(("seleccion", nombre))
    log(f"  -> {seleccion['motivo']}")
    cand = next(c for c in candidatos if c.id == seleccion["seleccionado"])
    sel = reajustar(nombre, cand, X_train, y_train, features)
    cv = {
        "n_splits": N_FOLDS,
        "shuffle": True,
        "random_state": SEED,
        "estratificado_por": target,
        "tamanos_validacion": [int(len(va)) for _, va in folds],
        "candidatos": resultados,
    }
    return sel, {"cv": cv, "seleccion": seleccion}, folds


def evaluar(
    etiqueta: str, sel: ModeloSeleccionado, df: pd.DataFrame, split: Split, target: str, registro: Registro
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    X_test, y_test = filas(df, split.test_idx, sel.features, target)
    registro.eventos.append(("test", etiqueta))
    m, proba = evaluar_en_test(sel, X_test, y_test)
    log(
        f"  test [{etiqueta}]: AUC {m['roc_auc']:.4f} {m['roc_auc_ic95_bootstrap']}  log-loss {m['log_loss']:.4f}"
    )
    return m, proba, y_test


def sensibilidad(
    df: pd.DataFrame, split: Split, y_test: np.ndarray, proba: np.ndarray, prev: float
) -> dict[str, Any]:
    plausible = df.iloc[split.test_idx]["pa_plausible"].to_numpy()
    s = compute_metrics(y_test[plausible], proba[plausible], prev)
    return {
        "filas_excluidas_de_test": int((~plausible).sum()),
        "roc_auc": s["roc_auc"],
        "log_loss": s["log_loss"],
        "brier": s["brier"],
    }


def bloque(
    nombre: str,
    target: str,
    sel: ModeloSeleccionado,
    split: Split,
    cv_sel: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    return {
        "nombre": nombre,
        "target": target,
        "features_orden": sel.features,
        "n_features": len(sel.features),
        "n_train": int(len(split.train_idx)),
        "n_test": int(len(split.test_idx)),
        "prevalencia_train": sel.prevalencia_train,
        "ganador": sel.candidato.familia,
        "candidato_seleccionado": sel.candidato.id,
        "hiperparametros": sel.candidato.hiperparametros,
        **cv_sel,
        "test": test,
        "scaler": {
            "tipo": "StandardScaler",
            "ajustado_sobre": "train completo, solo en el reajuste final",
            "mean_": [float(v) for v in sel.scaler.mean_],
            "scale_": [float(v) for v in sel.scaler.scale_],
        },
    }


def correr(df: pd.DataFrame, candidatos: list[Candidato], registro: Registro) -> dict[str, Any]:
    """Toda la corrida sobre ``df`` limpio. Devuelve bloques del manifiesto y modelos."""
    assert_no_pressure_lineage(FEATURES_B1)

    # ---- A′: misma partición y mismos folds para con/sin PA ----
    split_a = make_split(df, "cardio")
    sel_con, cv_con, folds_a = seleccionar_experimento(
        "riesgo_cv_con_pa", df, split_a, FEATURES_A_CON_PA, "cardio", candidatos, registro
    )
    sel_sin, cv_sin, _ = seleccionar_experimento(
        "riesgo_cv_sin_pa", df, split_a, FEATURES_A_SIN_PA, "cardio", candidatos, registro, folds=folds_a
    )
    # Selecciones fijadas: a partir de aquí se toca el test.
    t_con, p_con, y_a = evaluar("riesgo_cv_con_pa", sel_con, df, split_a, "cardio", registro)
    t_con["sensibilidad_test_sin_presion_implausible"] = sensibilidad(
        df, split_a, y_a, p_con, sel_con.prevalencia_train
    )
    t_sin, p_sin, _ = evaluar("riesgo_cv_sin_pa", sel_sin, df, split_a, "cardio", registro)
    t_sin["sensibilidad_test_sin_presion_implausible"] = sensibilidad(
        df, split_a, y_a, p_sin, sel_sin.prevalencia_train
    )

    # Ablación principal (controlada): misma configuración que con PA, sin ap_hi/ap_lo.
    X_tr_sin, y_tr_a = filas(df, split_a.train_idx, FEATURES_A_SIN_PA, "cardio")
    sel_ctrl = reajustar("ablacion_controlada_sin_pa", sel_con.candidato, X_tr_sin, y_tr_a, FEATURES_A_SIN_PA)
    t_ctrl, p_ctrl, _ = evaluar("ablacion_controlada_sin_pa", sel_ctrl, df, split_a, "cardio", registro)
    ablacion = {
        "principal_controlada": {
            "descripcion": "configuración del seleccionado con PA reajustada sin ap_hi/ap_lo",
            "configuracion": sel_con.candidato.id,
            "roc_auc_con_pa": t_con["roc_auc"],
            "roc_auc_sin_pa": t_ctrl["roc_auc"],
            "log_loss_sin_pa": t_ctrl["log_loss"],
            **paired_bootstrap_auc_delta(y_a, p_con, p_ctrl),
        },
        "secundaria_practica": {
            "descripcion": "seleccionado con PA frente a seleccionado sin PA (cada uno por su CV)",
            "configuracion_con_pa": sel_con.candidato.id,
            "configuracion_sin_pa": sel_sin.candidato.id,
            "roc_auc_con_pa": t_con["roc_auc"],
            "roc_auc_sin_pa": t_sin["roc_auc"],
            **paired_bootstrap_auc_delta(y_a, p_con, p_sin),
        },
        "no_interviene_en_la_seleccion": True,
    }

    # ---- B1 ----
    split_b = make_split(df, "hta")
    sel_b1, cv_b1, _ = seleccionar_experimento(
        "hta_b1", df, split_b, FEATURES_B1, "hta", candidatos, registro
    )
    t_b1, p_b1, y_b = evaluar("hta_b1", sel_b1, df, split_b, "hta", registro)
    t_b1["sensibilidad_test_sin_presion_implausible"] = sensibilidad(
        df, split_b, y_b, p_b1, sel_b1.prevalencia_train
    )

    # Robustez: misma configuración, etiqueta estricta, misma partición. Sin artefactos.
    X_tr_b, y_tr_gt = filas(df, split_b.train_idx, FEATURES_B1, "hta_gt")
    sel_gt = reajustar("hta_b1_robustez_gt", sel_b1.candidato, X_tr_b, y_tr_gt, FEATURES_B1)
    t_gt, _, _ = evaluar("hta_b1_robustez_gt", sel_gt, df, split_b, "hta_gt", registro)

    return {
        "splits": {"A": split_a, "B": split_b},
        "seleccionados": {"riesgo_cv_con_pa": sel_con, "riesgo_cv_sin_pa": sel_sin, "hta_b1": sel_b1},
        "experimentos": {
            "A_prima_principal": {
                "con_pa": bloque("riesgo_cv_con_pa", "cardio", sel_con, split_a, cv_con, t_con),
                "sin_pa": bloque("riesgo_cv_sin_pa", "cardio", sel_sin, split_a, cv_sin, t_sin),
                "ablacion": ablacion,
            },
            "B1_experimento": {
                "principal": bloque("hta_b1", "hta", sel_b1, split_b, cv_b1, t_b1)
                | {
                    "umbral": {
                        "definicion": "hta = (ap_hi >= 140) | (ap_lo >= 90)",
                        "fuente": "JNC7 / ESC, fijado a priori",
                    }
                },
                "robustez_umbral_estricto": {
                    "nota": "Misma configuración que hta_b1, reajustada con hta_gt; no es re-selección; sin artefactos.",
                    "configuracion": sel_b1.candidato.id,
                    "umbral": {"definicion": "hta_gt = (ap_hi > 140) | (ap_lo > 90)"},
                    "test": t_gt,
                },
            },
        },
    }


# =======================================================
# Entorno, trazabilidad y E/S
# =======================================================
def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def estado_git() -> dict[str, Any]:
    sucio = _git("status", "--porcelain", "--", *RUTAS_LIMPIAS)
    return {
        "commit": _git("rev-parse", "HEAD"),
        "arbol_limpio": not sucio,
        "cambios_sin_commitear": sucio.splitlines(),
    }


def pip_freeze() -> list[str]:
    # set(): con .venv/lib64 -> lib, importlib.metadata ve cada distribución dos veces.
    return sorted({f"{d.metadata['Name']}=={d.version}" for d in md.distributions()})


def hash_indices(a: np.ndarray) -> str:
    """sha256 de dtype + forma + bytes; mismo esquema que tests/test_equivalencia_features.py."""
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def verificar_split_dt1(df: pd.DataFrame, dt1: dict[str, Any]) -> None:
    """La partición debe ser la de DT-1: hashes de índices, n y medias/escalas del train."""
    for bloque_dt1, target, features in (
        (dt1["experimentos"]["A_prima_principal"]["con_pa"], "cardio", FEATURES_A_CON_PA),
        (dt1["experimentos"]["B1_experimento"]["principal"], "hta", FEATURES_B1),
    ):
        split = make_split(df, target)
        for parte in ("train_idx", "test_idx"):
            if hash_indices(getattr(split, parte)) != HASHES_SPLIT_DT1[target][parte]:
                raise SystemExit(
                    f"partición distinta a DT-1 en {bloque_dt1['nombre']} ({parte}: hash distinto)"
                )
        X_tr = df.iloc[split.train_idx][features].to_numpy(dtype=float)
        ref = StandardScaler().fit(X_tr)
        if len(split.train_idx) != bloque_dt1["n_train"] or len(split.test_idx) != bloque_dt1["n_test"]:
            raise SystemExit(f"partición distinta a DT-1 en {bloque_dt1['nombre']}")
        for attr in ("mean_", "scale_"):
            delta = float(np.max(np.abs(getattr(ref, attr) - np.asarray(bloque_dt1["scaler"][attr]))))
            if delta > TOL_SPLIT:
                raise SystemExit(
                    f"train distinto al de DT-1 en {bloque_dt1['nombre']} ({attr}, |Δ| = {delta:.2e})"
                )


def guardar_artefactos(sel: ModeloSeleccionado) -> dict[str, Any]:
    MODELS_DIR.mkdir(exist_ok=True)
    rutas = {
        "modelo": MODELS_DIR / f"dt4_{sel.experimento}__modelo.pkl",
        "scaler": MODELS_DIR / f"dt4_{sel.experimento}__scaler.pkl",
    }
    joblib.dump(sel.modelo, rutas["modelo"])
    joblib.dump(sel.scaler, rutas["scaler"])
    return {
        k: {"ruta": str(r.relative_to(ROOT)), "sha256": sha256_file(r), "bytes": r.stat().st_size}
        for k, r in rutas.items()
    }


def sha256_dt1() -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(MODELS_DIR.glob("dt1_*"))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desviacion", action="append", default=[], help="desviación del protocolo a registrar")
    ap.add_argument("--permitir-arbol-sucio", action="store_true", help="solo depuración; queda registrado")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    git = estado_git()
    if not git["arbol_limpio"] and not args.permitir_arbol_sucio:
        raise SystemExit(
            f"árbol sucio en {RUTAS_LIMPIAS}: commitea antes de correr\n"
            + "\n".join(git["cambios_sin_commitear"])
        )
    desviaciones = list(args.desviacion)
    if not git["arbol_limpio"]:
        desviaciones.append("corrida con árbol sucio (--permitir-arbol-sucio): no vale como preregistrada")

    dt1 = json.loads(DT1_MANIFEST_PATH.read_text())
    sha_csv = sha256_file(DATA_PATH)
    if sha_csv != dt1["dataset"]["sha256"]:
        raise SystemExit("el CSV no es el de DT-1 (sha256 distinto)")
    dt1_antes = sha256_dt1()

    df, limpieza = load_and_clean()
    verificar_split_dt1(df, dt1)
    log(f"datos: {len(df)} filas; partición idéntica a DT-1 verificada")

    registro = Registro()
    candidatos = grilla()
    res = correr(df, candidatos, registro)

    exps = res["experimentos"]
    for sel in res["seleccionados"].values():
        destino = (
            exps["B1_experimento"]["principal"]
            if sel.experimento == "hta_b1"
            else (
                exps["A_prima_principal"]["con_pa"]
                if sel.experimento == "riesgo_cv_con_pa"
                else exps["A_prima_principal"]["sin_pa"]
            )
        )
        destino["artefactos"] = guardar_artefactos(sel)

    dt1_despues = sha256_dt1()
    if dt1_antes != dt1_despues:
        desviaciones.append("los archivos dt1_* cambiaron durante la corrida")

    protocolo_commit = _git("log", "-n", "1", "--format=%H", "--", str(PROTOCOL_PATH.relative_to(ROOT)))
    manifest = {
        "hito": "DT-4",
        "version_manifiesto": 2,
        "generado_por": "src/train_cardio_dt4.py",
        "protocolo": {
            "ruta": str(PROTOCOL_PATH.relative_to(ROOT)),
            "sha256": sha256_file(PROTOCOL_PATH),
            "commit": protocolo_commit,
        },
        "codigo": git,
        "semilla": SEED,
        "test_size": 0.2,
        "entorno": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "pip_freeze": pip_freeze(),
        },
        "dataset": {"ruta": str(DATA_PATH.relative_to(ROOT)), "sha256": sha_csv, "separador": ";"},
        "limpieza": limpieza,
        "particion": "idéntica a DT-1 (make_split, random_state=42); verificada contra dt1_manifest.json",
        "grilla": [
            {"id": c.id, "familia": c.familia, "hiperparametros": c.hiperparametros} for c in candidatos
        ],
        "orden_simplicidad": [c.id for c in candidatos],
        "svm_rbf": "excluida a priori (protocolo §5)",
        "experimentos": exps,
        "traza": [f"{tipo}:{nombre}" for tipo, nombre in registro.eventos],
        "dt1_sha256_antes_y_despues_iguales": dt1_antes == dt1_despues,
        "desviaciones": desviaciones,
        "duracion_segundos": round(time.perf_counter() - t_start, 1),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    log(f"manifiesto: {MANIFEST_PATH.relative_to(ROOT)} ({manifest['duracion_segundos']} s)")


if __name__ == "__main__":
    sys.exit(main())
