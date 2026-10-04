r"""Arayüz için hızlı duman testi: tüm sayfaları hata vermeden çalıştırır.

Çalıştırma: .\.venv\Scripts\python.exe smoke_test.py
"""
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import analysis
import iris_ui as ui
from train import DATA


def run(at):
    at.run(timeout=180)
    assert not at.exception, at.exception
    return at


def open_page(at, path):
    at.switch_page(path)
    return run(at)


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

# ---------- Tasarım modülü ----------
assert ui.fmt_num(10000) == "10.000" and ui.fmt_pct(0.2037, 2) == "%20,37"
assert ui.risk_color(0) == ui.TURQ.upper() and ui.risk_color(1) == ui.CORAL.upper()
print("Tasarım modülü OK")

# ---------- Arayüz ----------
at = run(AppTest.from_file("app.py", default_timeout=180))
# Giriş sayfası: 6 sayfa düğmesi ve gerçek veriden sayılar
html = " ".join(str(h.proto.body) for h in at.get("html"))
assert "Müşteri Terki" in html and 'data-countup="10000' in html, "Giriş özeti eksik"
assert "Portföy risk seviyesi" in html
print("Giriş sayfası OK")

# Genel Bakış: varsayılan + filtreli
at = open_page(at, "sayfalar/genel_bakis.py")
html = " ".join(str(h.proto.body) for h in at.get("html"))
assert "iris-kpi" in html, "KPI kartları çizilmedi"
at.pills(key="f_country").set_value(["Germany"])
at.segmented_control(key="f_member").set_value("Pasif")
run(at)
at.pills(key="f_products").set_value([4])
at.segmented_control(key="f_balance").set_value("= 0")  # Almanya'da bakiye=0 yok -> boş
run(at)
assert any("müşteri yok" in i.value for i in at.info), "Boş filtre mesajı yok"
print("Genel Bakış (filtreler) OK")

# Müşteri Risk Skoru: her model için (kenar çubuğundaki ortak model seçimi)
at = open_page(at, "sayfalar/risk_skoru.py")
for model in ["lr", "rf", "xgb"]:
    at.selectbox(key="_w_model").set_value(model)
    run(at)
    print(model, "->", [f"{m.label}: {m.value}" for m in at.metric][-2:])
at.slider(key="_w_thr_xgb").set_value(0.5)
run(at)
assert at.session_state["iris_thr"]["xgb"] == 0.5, "Ortak eşik kaydedilmedi"
at.radio(key="risk_mode").set_value("Elle değer gir")
run(at)
print("Elle giriş ->", [f"{m.label}: {m.value}" for m in at.metric][-2:])

# Ortak ayar sayfalar arasında korunmalı (giriş sayfası widget çizmez)
at = open_page(at, "sayfalar/giris.py")
at = open_page(at, "sayfalar/genel_bakis.py")
assert at.session_state["iris_model"] == "xgb" and at.session_state["iris_thr"]["xgb"] == 0.5
print("Ortak model/eşik ayarı OK")

# Açıklanabilirlik: Random Forest
at = open_page(at, "sayfalar/aciklanabilirlik.py")
at.radio(key="shap_model").set_value("rf")
run(at)

# Model karşılaştırma
at = open_page(at, "sayfalar/model_karsilastirma.py")
assert at.dataframe, "Metrik tablosu yok"

# Maliyet: farklı maliyetlerle yeniden çalışmalı
at = open_page(at, "sayfalar/maliyet.py")
at.number_input(key="cost_fn").set_value(10.0)
at.number_input(key="cost_fp").set_value(2.0)
run(at)
print("Maliyet (FN=10, FP=2) OK")

# Confounding: otomatik yorum üretilmiş olmalı
at = open_page(at, "sayfalar/confounding.py")
assert any("odds ratio" in i.value for i in at.info), "Confounding yorumu bulunamadı"
print("Confounding yorumu OK")
print("SMOKE TEST OK")
