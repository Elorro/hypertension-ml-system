"""Garantías estructurales del protocolo de DT-4 (docs/DT4_PROTOCOL.md).

Corren en CI sin datos reales: un DataFrame sintético con el esquema limpio y una
grilla mínima. Lo que se verifica es el procedimiento, no las cifras.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import src.train_cardio_real as real
from src import train_cardio_dt4 as dt4
from src.cardio_features import FEATURES_A_CON_PA, FEATURES_B1, bmi

EVALUACIONES_PREREGISTRADAS = {
    "riesgo_cv_con_pa": "riesgo_cv_con_pa",
    "riesgo_cv_sin_pa": "riesgo_cv_sin_pa",
    "ablacion_controlada_sin_pa": "riesgo_cv_con_pa",  # usa la configuración de con PA
    "hta_b1": "hta_b1",
    "hta_b1_robustez_gt": "hta_b1",
}


def df_sintetico(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    height = rng.normal(165, 8, n).round()
    weight = rng.normal(74, 12, n).clip(40, 150)
    ap_hi = rng.normal(128, 16, n).round()
    ap_lo = (ap_hi - rng.normal(45, 8, n)).round()
    age = rng.uniform(30, 65, n)
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "age_years": age,
            "gender": rng.integers(1, 3, n),
            "height": height,
            "weight": weight,
            "bmi": bmi(weight, height),
            "cholesterol": rng.integers(1, 4, n),
            "gluc": rng.integers(1, 4, n),
            "smoke": rng.integers(0, 2, n),
            "alco": rng.integers(0, 2, n),
            "active": rng.integers(0, 2, n),
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
        }
    )
    logit = 0.06 * (ap_hi - 128) + 0.05 * (age - 50) + rng.normal(0, 1, n)
    df["cardio"] = (logit > 0).astype(int)
    df["hta"] = ((ap_hi >= 140) | (ap_lo >= 90)).astype(int)
    df["hta_gt"] = ((ap_hi > 140) | (ap_lo > 90)).astype(int)
    df["pa_plausible"] = df["ap_hi"].between(70, 250) & df["ap_lo"].between(40, 200)
    return df


MINI_GRILLA = [dt4.Candidato("LogisticRegression", {"C": c}) for c in (0.1, 1.0)]


@pytest.fixture(autouse=True)
def bootstrap_corto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(real, "N_BOOTSTRAP", 20)
    monkeypatch.setattr(dt4, "log", lambda msg: None)


# =======================================================
# Grilla y regla de selección
# =======================================================
def test_grilla_del_protocolo():
    g = dt4.grilla()
    assert len(g) == 25
    assert {c.familia for c in g} == set(dt4.FAMILIAS)  # sin SVM
    # Orden de simplicidad: familias en bloque y en el orden declarado.
    familias = [c.familia for c in g]
    assert familias == sorted(familias, key=dt4.FAMILIAS.index)
    # Las configuraciones de DT-1 están dentro de la grilla.
    ids = {c.id for c in g}
    for dt1 in (
        "LogisticRegression(C=1.0)",
        "DecisionTree(max_depth=8,min_samples_leaf=1)",
        "RandomForest(max_depth=12,min_samples_leaf=1)",
        "XGBoost(max_depth=8,min_child_weight=1)",
    ):
        assert dt1 in ids
    # Dentro de familia: más regularización primero.
    lr = [c.hiperparametros["C"] for c in g if c.familia == "LogisticRegression"]
    assert lr == sorted(lr)
    for fam, reg in (
        ("DecisionTree", "min_samples_leaf"),
        ("RandomForest", "min_samples_leaf"),
        ("XGBoost", "min_child_weight"),
    ):
        claves = [(c.hiperparametros["max_depth"], -c.hiperparametros[reg]) for c in g if c.familia == fam]
        assert claves == sorted(claves)


def _res(id_: str, losses: list[float], excluido: bool = False) -> dict:
    if excluido:
        return {"id": id_, "excluido": True, "motivo": "x"}
    ll = np.array(losses)
    sd = float(ll.std(ddof=1))
    return {
        "id": id_,
        "excluido": False,
        "log_loss_media": float(ll.mean()),
        "log_loss_sd": sd,
        "log_loss_ee": sd / math.sqrt(5),
    }


def test_regla_un_ee_elige_el_mas_simple_dentro_de_un_ee():
    mejor = [0.50, 0.52, 0.48, 0.51, 0.49]  # media 0.50, sd 0.0158, EE 0.00707
    resultados = [
        _res("simple_fuera", [0.52] * 5),  # 0.52 > 0.50707: fuera
        _res("excluido", [], excluido=True),
        _res("simple_dentro", [0.505] * 5),  # dentro
        _res("mejor", mejor),
        _res("complejo_dentro", [0.501] * 5),
    ]
    sel = dt4.regla_un_ee(resultados)
    assert sel["mejor"] == "mejor"
    assert sel["ee"] == pytest.approx(np.std(mejor, ddof=1) / math.sqrt(5))
    assert sel["elegibles"] == ["simple_dentro", "mejor", "complejo_dentro"]
    assert sel["seleccionado"] == "simple_dentro"


def test_candidato_que_falla_queda_excluido():
    df = df_sintetico()
    X, y = df[FEATURES_B1].to_numpy(dtype=float), df["hta"].to_numpy()
    malo = dt4.Candidato("LogisticRegression", {"C": -1.0})  # C inválido: fit lanza
    r = dt4.evaluar_candidato_cv(malo, X, y, dt4.folds_estratificados(y))
    assert r["excluido"] and "fold 0" in r["motivo"]


# =======================================================
# Escalado dentro de la CV
# =======================================================
def test_scaler_de_cada_fold_solo_ve_sus_filas_de_entrenamiento():
    df = df_sintetico()
    X, y = df[FEATURES_A_CON_PA].to_numpy(dtype=float), df["cardio"].to_numpy()
    for tr, va in dt4.folds_estratificados(y):
        pipe = dt4.ajustar_fold(MINI_GRILLA[0], X, y, tr)
        scaler = pipe.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, X[tr].mean(axis=0))
        np.testing.assert_allclose(scaler.scale_, X[tr].std(axis=0))
        # Control: con las filas de validación dentro, la media sería otra.
        assert not np.allclose(scaler.mean_, X.mean(axis=0))
        assert not np.allclose(scaler.mean_, X[np.concatenate([tr, va[:50]])].mean(axis=0))


# =======================================================
# El test se toca solo después de seleccionar, una vez
# =======================================================
def test_cv_y_reajuste_no_leen_filas_de_test():
    """Con NaN en todas las filas de test, la selección debe funcionar igual."""
    df = df_sintetico()
    split = real.make_split(df, "hta")
    df.loc[df.index[split.test_idx], FEATURES_B1] = np.nan  # LR falla con NaN si los lee
    reg = dt4.Registro()
    sel, cv_sel, _ = dt4.seleccionar_experimento("hta_b1", df, split, FEATURES_B1, "hta", MINI_GRILLA, reg)
    assert not any(r["excluido"] for r in cv_sel["cv"]["candidatos"])
    X_train = df.iloc[split.train_idx][FEATURES_B1].to_numpy(dtype=float)
    np.testing.assert_allclose(sel.scaler.mean_, X_train.mean(axis=0))
    assert reg.eventos == [("seleccion", "hta_b1")]


def test_evaluar_en_test_solo_acepta_un_seleccionado():
    with pytest.raises(TypeError):
        dt4.evaluar_en_test(object(), np.zeros((2, 10)), np.array([0, 1]))  # type: ignore[arg-type]


def test_cada_evaluacion_en_test_ocurre_una_vez_y_despues_de_su_seleccion(monkeypatch: pytest.MonkeyPatch):
    llamadas: list[str] = []
    original = dt4.evaluar_en_test

    def espia(sel, X, y):
        llamadas.append(sel.experimento)
        return original(sel, X, y)

    monkeypatch.setattr(dt4, "evaluar_en_test", espia)
    reg = dt4.Registro()
    res = dt4.correr(df_sintetico(), MINI_GRILLA, reg)

    assert sorted(llamadas) == sorted(EVALUACIONES_PREREGISTRADAS)
    tests = [n for tipo, n in reg.eventos if tipo == "test"]
    assert sorted(tests) == sorted(EVALUACIONES_PREREGISTRADAS)  # cada una exactamente una vez
    posicion = {ev: i for i, ev in enumerate(reg.eventos)}
    for evaluacion, seleccion in EVALUACIONES_PREREGISTRADAS.items():
        assert posicion[("seleccion", seleccion)] < posicion[("test", evaluacion)]
    # Las dos selecciones de A′ se fijan antes de tocar el test de A′.
    assert posicion[("seleccion", "riesgo_cv_sin_pa")] < posicion[("test", "riesgo_cv_con_pa")]

    abl = res["experimentos"]["A_prima_principal"]["ablacion"]
    assert {"principal_controlada", "secundaria_practica"} <= set(abl)
    assert (
        abl["principal_controlada"]["configuracion"] == res["seleccionados"]["riesgo_cv_con_pa"].candidato.id
    )


def test_manifiesto_por_experimento_registra_folds_y_regla():
    res = dt4.correr(df_sintetico(), MINI_GRILLA, dt4.Registro())
    b = res["experimentos"]["B1_experimento"]["principal"]
    assert b["cv"]["n_splits"] == 5
    for cand in b["cv"]["candidatos"]:
        assert len(cand["folds"]) == 5
        assert {"log_loss", "brier", "roc_auc", "ece_uniform_10", "log_loss_train_fold"} <= set(
            cand["folds"][0]
        )
    assert {"mejor", "ee", "umbral", "elegibles", "seleccionado", "motivo"} <= set(b["seleccion"])
    assert {"roc_auc", "roc_auc_ic95_bootstrap", "log_loss", "brier", "ece_uniform_10"} <= set(b["test"])
    assert b["ganador"] == "LogisticRegression"


@pytest.mark.requires_data
def test_particion_identica_a_dt1(entrenamiento, manifiesto):
    """La verificación de arranque de la corrida acepta el CSV real con la partición de DT-1."""
    _, df = entrenamiento
    dt4.verificar_split_dt1(df, manifiesto)


def test_verificar_split_detecta_una_particion_distinta(manifiesto):
    df = df_sintetico()
    with pytest.raises(SystemExit, match="partición distinta"):
        dt4.verificar_split_dt1(df, manifiesto)


def test_hashes_de_particion_coinciden_con_los_congelados():
    from tests.test_equivalencia_features import REFERENCIA, _h

    assert dt4.HASHES_SPLIT_DT1["cardio"] == {
        k: REFERENCIA["riesgo_cv_con_pa"][k] for k in ("train_idx", "test_idx")
    }
    assert dt4.HASHES_SPLIT_DT1["hta"] == {k: REFERENCIA["hta_b1"][k] for k in ("train_idx", "test_idx")}
    a = np.arange(10)
    assert dt4.hash_indices(a) == _h(a)
