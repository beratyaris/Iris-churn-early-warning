r"""Arayüz için hızlı duman testi: tüm sekmeleri hata vermeden çalıştırır.

Çalıştırma: .\.venv\Scripts\python.exe smoke_test.py
"""
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import analysis
from train import DATA


def run(at):
    at.run(timeout=180)
    assert not at.exception, at.exception
    return at


# ---------- Analiz modülü: küçük tutarlılık kontrolleri ----------
df = pd.read_csv(DATA)
counts, row_pct, chi = analysis.country_balance_table(df)
assert counts.to_numpy().sum() == len(df)
assert np.allclose(row_pct.sum(axis=1), 100)
strat = analysis.stratified_churn(df)
assert strat.groupby("Tabaka")["n"].sum()["Tümü"] == len(df)
germany, change, _, _ = analysis.germany_comparison(df)
assert len(germany) == 2 and (germany["GA alt"] < germany["OR"]).all()

y = np.array([0, 0, 1, 1])
p = np.array([0.1, 0.6, 0.4, 0.9])
assert analysis.evaluate_at(y, p, 0.5, 5, 1) == {"Recall": 0.5, "Precision": 0.5,
                                                 "FN": 1, "FP": 1, "Toplam maliyet": 6}
thr, curve = analysis.cost_optimal_threshold(y, p, 5, 1)
assert curve["Maliyet"].min() == analysis.evaluate_at(y, p, thr, 5, 1)["Toplam maliyet"]
assert abs(analysis.bayes_threshold(5, 1) - 1 / 6) < 1e-12
print("Analiz modülü OK")

# ---------- Arayüz ----------
at = run(AppTest.from_file("app.py", default_timeout=180))
labels = [t.label for t in at.tabs]
print("Sekmeler:", labels)
assert "Maliyet Analizi" in labels and "Özgünlük: Confounding Kontrolü" in labels

# Her model için risk sekmesi (test setinden müşteri) çalışmalı
for model in ["lr", "rf", "xgb"]:
    at.selectbox[0].set_value(model)
    run(at)
    print(model, "->", [f"{m.label}: {m.value}" for m in at.metric][-2:])

# Elle giriş modu
at.radio[0].set_value("Elle değer gir")
run(at)
print("Elle giriş ->", [f"{m.label}: {m.value}" for m in at.metric][-2:])

# Açıklanabilirlik sekmesinde Random Forest
at.radio[1].set_value("rf")
run(at)

# Maliyet sekmesi: farklı maliyetlerle yeniden çalışmalı
at.number_input(key="cost_fn").set_value(10.0)
at.number_input(key="cost_fp").set_value(2.0)
run(at)
print("Maliyet (FN=10, FP=2) OK")

# Confounding sekmesi: otomatik yorum üretilmiş olmalı
assert any("odds ratio" in i.value for i in at.info), "Confounding yorumu bulunamadı"
print("Confounding yorumu OK")
print("SMOKE TEST OK")
