r"""Banka Müşteri Terki (Churn) Erken Uyarı Sistemi - model eğitimi.

Çalıştırma:  .\.venv\Scripts\python.exe train.py
Çıktılar artifacts/ klasörüne yazılır; app.py yalnızca bunları okur.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             fbeta_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "Bank Customer Churn Prediction.csv"
OUT = ROOT / "artifacts"

MODEL_NAMES = {"lr": "Lojistik Regresyon", "rf": "Random Forest", "xgb": "XGBoost"}


def prepare_features(df):
    """Ham veriden model girdisini üretir (app.py de aynı fonksiyonu kullanır)."""
    X = df.drop(columns=["customer_id", "churn"], errors="ignore").copy()
    # Yeni özellikler
    X["has_balance"] = (X["balance"] > 0).astype(int)
    X["balance_salary_ratio"] = X["balance"] / X["estimated_salary"]
    # One-Hot Encoding (referans kategori: France, Female)
    X["country_Germany"] = (X["country"] == "Germany").astype(int)
    X["country_Spain"] = (X["country"] == "Spain").astype(int)
    X["gender_Male"] = (X["gender"] == "Male").astype(int)
    return X.drop(columns=["country", "gender"])


# Kesikli (0/1 ve ürün sayısı) sütunlar: SMOTE bunları 0.37 gibi var olmayan
# değerlere çevirmesin diye SMOTE-NC ile "kategorik" olarak işaretlenir.
DISCRETE = ["products_number", "credit_card", "active_member", "has_balance",
            "country_Germany", "country_Spain", "gender_Male"]


def build_models(feature_names):
    """Üç model. Her pipeline: ölçekleme -> SMOTE-NC -> model.
    imblearn Pipeline'da SMOTE yalnızca fit sırasında (eğitim verisine) çalışır;
    validation/test tahmininde atlanır. Ölçekleme SMOTE'un komşuluk mesafesi için
    gerekli; ağaç modelleri ölçekten etkilenmez."""
    cat_idx = [feature_names.index(c) for c in DISCRETE]

    def pipe(model):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTENC(categorical_features=cat_idx,
                              sampling_strategy=0.5, random_state=SEED)),
            ("model", model),
        ])

    return {
        "lr": pipe(LogisticRegression(max_iter=1000)),
        "rf": pipe(RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1)),
        "xgb": pipe(XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED, n_jobs=-1)),
    }


def metrics_at(y_true, proba, threshold):
    y_pred = (proba >= threshold).astype(int)
    return {
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred),
        "F2": fbeta_score(y_true, y_pred, beta=2),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


def best_f2_threshold(y_true, proba):
    """Validation setinde F2'yi maksimize eden eşik (0.05-0.95 arası, 0.01 adım)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [fbeta_score(y_true, (proba >= t).astype(int), beta=2) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    X = prepare_features(df)
    y = df["churn"]

    # 1) Test ayrımı: %80 eğitim+validation, %20 test (stratified)
    X_trval, X_test, y_trval, y_test, id_trval, id_test = train_test_split(
        X, y, df["customer_id"], test_size=0.20, stratify=y, random_state=SEED)
    # 2) Eşik seçimi için eğitim içinden validation: %60 eğitim, %20 validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=SEED)
    print(f"Eğitim: {len(X_train)}  Validation: {len(X_val)}  Test: {len(X_test)}")

    models = build_models(list(X.columns))
    thresholds, rows, test_proba, val_proba = {}, [], {}, {}
    for key, pipe in models.items():
        pipe.fit(X_train, y_train)  # SMOTE yalnızca burada, eğitim verisine uygulanır
        val_proba[key] = pipe.predict_proba(X_val)[:, 1]
        thr, val_f2 = best_f2_threshold(y_val, val_proba[key])
        thresholds[key] = {"threshold": thr, "val_f2": val_f2}
        proba = pipe.predict_proba(X_test)[:, 1]
        test_proba[key] = proba
        for label, t in [("0.50", 0.50), ("Optimize", thr)]:
            rows.append({"Model": MODEL_NAMES[key], "Eşik türü": label, "Eşik": t,
                         **metrics_at(y_test, proba, t)})
        print(f"{MODEL_NAMES[key]}: optimize eşik = {thr:.2f} (validation F2 = {val_f2:.3f})")

    metrics = pd.DataFrame(rows)
    print("\nTest metrikleri:\n", metrics.round(3).to_string(index=False))

    # 3) McNemar testi: optimize eşikle tahminlerin doğru/yanlış olmasına göre
    y_arr = y_test.to_numpy()
    correct = {k: ((test_proba[k] >= thresholds[k]["threshold"]).astype(int) == y_arr)
               for k in models}
    mc_rows = []
    for a, b in [("lr", "rf"), ("lr", "xgb"), ("rf", "xgb")]:
        table = [[np.sum(correct[a] & correct[b]), np.sum(correct[a] & ~correct[b])],
                 [np.sum(~correct[a] & correct[b]), np.sum(~correct[a] & ~correct[b])]]
        n_disc = table[0][1] + table[1][0]
        # Uyumsuz çift sayısı azsa kesin (binom) test, değilse süreklilik düzeltmeli ki-kare
        res = mcnemar(table, exact=n_disc < 25, correction=True)
        mc_rows.append({"Karşılaştırma": f"{MODEL_NAMES[a]} vs {MODEL_NAMES[b]}",
                        "Yalnız 1. doğru": int(table[0][1]), "Yalnız 2. doğru": int(table[1][0]),
                        "İstatistik": float(res.statistic), "p": float(res.pvalue)})
    mc = pd.DataFrame(mc_rows)
    reject, p_holm, _, _ = multipletests(mc["p"], alpha=0.05, method="holm")
    mc["p (Holm)"] = p_holm
    mc["Sonuç (α=0.05)"] = np.where(reject, "Anlamlı fark", "Anlamlı değil")
    print("\nMcNemar + Holm:\n", mc.round(4).to_string(index=False))

    # 4) SHAP (ağaç modelleri, test seti) - log-odds (XGB) / olasılık (RF) ölçeğinde.
    # Model ölçeklenmiş veriyle eğitildiği için SHAP de ölçeklenmiş girdiyle hesaplanır;
    # grafiklerde ham değerler gösterilir (ölçekleme monoton, yorum değişmez).
    shap_out = {}
    for key in ["xgb", "rf"]:
        explainer = shap.TreeExplainer(models[key].named_steps["model"])
        values = explainer.shap_values(models[key].named_steps["scaler"].transform(X_test))
        base = explainer.expected_value
        if key == "rf":  # RF: sınıf 1 (churn) değerlerini al
            values = values[:, :, 1] if np.ndim(values) == 3 else values[1]
            base = np.ravel(base)[1]
        shap_out[key] = {"values": np.asarray(values), "base": float(np.ravel(base)[0])}
    mean_abs = pd.Series(np.abs(shap_out["xgb"]["values"]).mean(axis=0),
                         index=X_test.columns).sort_values(ascending=False)
    print("\nXGBoost SHAP ortalama |değer| (ilk 5):\n", mean_abs.head(5).round(4).to_string())

    # 5) Kaydet
    joblib.dump(models, OUT / "models.joblib")
    joblib.dump(shap_out, OUT / "shap_test.joblib")
    # LR için SHAP LinearExplainer arka plan verisi (ham eğitim örneği; app ölçekler)
    joblib.dump(X_train.sample(200, random_state=SEED), OUT / "background.joblib")
    metrics.to_csv(OUT / "metrics.csv", index=False)
    mc.to_csv(OUT / "mcnemar.csv", index=False)
    preds = X_test.copy()
    preds.insert(0, "customer_id", id_test.to_numpy())
    preds["churn"] = y_arr
    for k in models:
        preds[f"proba_{k}"] = test_proba[k]
    preds.to_csv(OUT / "test_predictions.csv", index=False)
    # Validation tahminleri: maliyet-optimal eşik test setine dokunmadan bunlarla seçilir
    val_preds = pd.DataFrame({"churn": y_val.to_numpy(),
                              **{f"proba_{k}": val_proba[k] for k in models}})
    val_preds.to_csv(OUT / "val_predictions.csv", index=False)
    with open(OUT / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "feature_names": list(X.columns), **thresholds}, f, indent=2)
    print(f"\nArtifacts kaydedildi: {OUT}")


if __name__ == "__main__":
    main()
