import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# =======================================================
# Función para evaluar modelos
# =======================================================
def evaluar_modelo(nombre, modelo, X_test, y_test, y_pred, y_proba):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    # ROC-AUC: solo si el modelo tiene predict_proba
    try:
        roc = roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr"
        )
    except:
        roc = None

    print(f"\n===== {nombre} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"ROC-AUC OVR: {roc}")

    return acc, f1, roc


# =======================================================
# Entrenamiento principal
# =======================================================
def main():
    print("Cargando dataset...")
    df = pd.read_csv("data/raw/dataset_hipertension_sintetico.csv")

    # -------------------------
    # Separación X - y
    # -------------------------
    X = df.drop("Diagnostico", axis=1)
    y = df["Diagnostico"]

    # -------------------------
    # Escalado de features
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # -------------------------
    # Train / Test
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    resultados = {}

    # =======================================================
    # 1. REGRESIÓN LOGÍSTICA
    # =======================================================
    log_reg = LogisticRegression(max_iter=2000, multi_class="multinomial")
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    y_proba = log_reg.predict_proba(X_test)

    acc, f1, roc = evaluar_modelo("Logistic Regression", log_reg, X_test, y_test, y_pred, y_proba)
    resultados["LogisticRegression"] = f1
    joblib.dump(log_reg, "models/modelo_logreg.pkl")

    # =======================================================
    # 2. SVM RBF
    # =======================================================
    svm = SVC(kernel="rbf", C=2, probability=True)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    y_proba = svm.predict_proba(X_test)

    acc, f1, roc = evaluar_modelo("SVM RBF", svm, X_test, y_test, y_pred, y_proba)
    resultados["SVM_RBF"] = f1
    joblib.dump(svm, "models/modelo_svm.pkl")

    # =======================================================
    # 3. ÁRBOL DE DECISIÓN
    # =======================================================
    tree = DecisionTreeClassifier(max_depth=8)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    y_proba = tree.predict_proba(X_test)

    acc, f1, roc = evaluar_modelo("Decision Tree", tree, X_test, y_test, y_pred, y_proba)
    resultados["DecisionTree"] = f1
    joblib.dump(tree, "models/modelo_tree.pkl")

    # =======================================================
    # 4. RANDOM FOREST
    # =======================================================
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    acc, f1, roc = evaluar_modelo("Random Forest", rf, X_test, y_test, y_pred, y_proba)
    resultados["RandomForest"] = f1
    joblib.dump(rf, "models/modelo_rf.pkl")

    # =======================================================
    # 5. XGBOOST
    # =======================================================
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss"
    )

    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)

    acc, f1, roc = evaluar_modelo("XGBoost", xgb, X_test, y_test, y_pred, y_proba)
    resultados["XGBoost"] = f1
    joblib.dump(xgb, "models/modelo_xgb.pkl")

    # =======================================================
    # Selección del mejor modelo
    # =======================================================
    mejor = max(resultados, key=resultados.get)
    print(f"\n🔥 Mejor modelo según F1-score: {mejor}")

    with open("models/mejor_modelo.txt", "w") as f:
        f.write(mejor)


if __name__ == "__main__":
    main()
