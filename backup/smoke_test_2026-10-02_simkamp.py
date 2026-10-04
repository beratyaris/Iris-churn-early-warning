r"""Arayüz için hızlı duman testi: analiz/kalibrasyon yardımcılarını ve tüm sayfaları çalıştırır.

Çalıştırma: .\.venv\Scripts\python.exe smoke_test.py
(Önce train.py ve calibrate.py çalışmış olmalı.)
"""
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import analysis
import iris_data as idata
import iris_ui as ui
from train import DATA, MODEL_NAMES


def run(at):
    at.run(timeout=180)
    assert not at.exception, at.exception
    return at


def open_page(at, path):
    at.switch_page(path)
    return run(at)


def page_html(at):
    return " ".join(str(h.proto.body) for h in at.get("html"))


# ---------- Analiz modülü: küçük tutarlılık kontrolleri ----------
df = pd.read_csv(DATA)
counts, row_pct, chi = analysis.country_balance_table(df)
assert counts.to_numpy().sum() == len(df)
assert np.allclose(row_pct.sum(axis=1), 100)
strat = analysis.stratified_churn(df)
assert strat.groupby("Tabaka")["n"].sum()["Tümü"] == len(df)
germany, change, _, _ = analysis.germany_comparison(df)
assert len(germany) == 2 and (germany["GA alt"] < germany["OR"]).all()
assert "," in analysis.interpret(strat, germany, change)[1], "Yorumda Türkçe ondalık yok"

y = np.array([0, 0, 1, 1])
p = np.array([0.1, 0.6, 0.4, 0.9])
assert analysis.evaluate_at(y, p, 0.5, 5, 1) == {"Recall": 0.5, "Precision": 0.5,
                                                 "FN": 1, "FP": 1, "Toplam maliyet": 6}
thr, curve = analysis.cost_optimal_threshold(y, p, 5, 1)
assert curve["Maliyet"].min() == analysis.evaluate_at(y, p, thr, 5, 1)["Toplam maliyet"]
assert abs(analysis.bayes_threshold(5, 1) - 1 / 6) < 1e-12
print("Analiz modülü OK")

# ---------- İstatistik yardımcıları ----------
lo, hi = idata.wilson(50, 100)
assert abs((lo + hi) / 2 - 0.5) < 1e-12 and 0.40 < lo < 0.41, (lo, hi)
lo0, hi0 = idata.wilson(0, 20)
assert abs(lo0) < 1e-12 and hi0 > 0.1
blo, bhi = idata.score_band(0.2, 2000)
assert abs((blo + bhi) / 2 - 0.2) < 1e-12 and 0.017 < bhi - 0.2 < 0.018
assert ui.rel_class(0.1, 0.15, 0.2) == "below" and ui.rel_class(0.18, 0.25, 0.2) == "near"
assert ui.rel_class(0.21, 0.3, 0.2) == "above"
print("İstatistik yardımcıları OK")

# ---------- Kalibrasyon ----------
art = idata.load_artifacts()
assert art["calibration"] is not None, "calibration.* yok: önce calibrate.py çalıştırın"
summ = art["calibration"]["summary"]
grid = np.linspace(0.001, 0.999, 500)
for k in MODEL_NAMES:
    info = summ["models"][k]
    raw = art["preds"][f"proba_{k}"].to_numpy()
    assert abs(info["mean_pred_test"] - raw.mean()) < 1e-9
    assert info["method"] in ("isotonic", "sigmoid")
    cal = idata.calibrate(art, k, grid)
    assert np.all(np.diff(cal) >= -1e-12), f"{k}: kalibratör monoton değil"
    # Eşik eşdeğerliği: skor ≥ eşik ⇔ kalibre olasılık ≥ eşiğin karşılığı (uyarı listesi değişmez)
    t = art["thresholds"][k]
    tp = idata.threshold_as_proba(art, k, t)
    same = (raw >= t) == (idata.calibrate(art, k, raw) >= tp)
    assert same.mean() > 0.999, f"{k}: eşik eşdeğerliği bozuk ({same.mean():.4f})"
    exp = idata.expected_churn(art, k, float(df["churn"].mean()))
    print(f"{MODEL_NAMES[k]}: ham ort. {exp['raw_mean']:.4f} → kalibre {exp['value']:.4f} "
          f"(gerçek {exp['actual']:.4f}), Brier {info['brier_test']:.4f} → {info['brier_test_cal']:.4f}")
print("Kalibrasyon OK")

# ---------- Tasarım modülü ----------
assert ui.fmt_num(10000) == "10.000" and ui.fmt_pct(0.2037, 2) == "%20,37"
assert ui.fmt_p(0.0004) == "< 0,001" and ui.fmt_p(0.0342) == "0,034"
assert ui.risk_color(0) == ui.TURQ.upper() and ui.risk_color(1) == ui.CORAL.upper()
svg = ui.gauge_svg(0.2, 0.4, [(0, .18, ui.TURQ), (.18, .22, ui.AMBER), (.22, .4, ui.CORAL)],
                   [(0, "%0"), (.2, "%20"), (.4, "%40")], marker=(.21, "eşik"))
assert svg.startswith("<img") and "base64" in svg
assert "Mini grafik" not in ui.kpi_card("x", "1", tip="açıklama")
print("Tasarım modülü OK")

# ---------- Arayüz ----------
at = run(AppTest.from_file("app.py", default_timeout=180))
html = page_html(at)
assert "Müşteri Terki" in html and 'data-countup="10000' in html, "Giriş özeti eksik"
assert "Beklenen churn oranı" in html and "Uyarı listesi" in html
assert "Portföy risk seviyesi" not in html and "Yüksek riskli" not in html
exp = idata.expected_churn(art, "xgb", float(df["churn"].mean()))
assert f'data-countup="{exp["value"] * 100:.6f}"' in html, "Gösterge değeri hesaplanan ile aynı değil"
n_alert = idata.alert_list(art, "xgb", art["thresholds"]["xgb"])["n_alert"]
assert f'data-countup="{n_alert:.6f}"' in html
print(f"Giriş sayfası OK (beklenen churn {exp['value']:.4f}, uyarı listesi {n_alert})")

# Genel Bakış: varsayılan (filtre yok → fark rozeti yok) + filtreli
at = open_page(at, "sayfalar/genel_bakis.py")
html = page_html(at)
assert "iris-kpi" in html, "KPI kartları çizilmedi"
assert 'class="k-cap"' not in html and 'class="iris-delta' not in html, \
    "Filtresizken fark rozeti ya da görünür yardım metni var"
assert 'class="k-info"' in html, "KPI ipuçları (tooltip) yok"
assert "Wilson" in html
at.pills(key="f_country").set_value(["Germany"])
at.segmented_control(key="f_member").set_value("Pasif")
run(at)
assert 'class="iris-delta' in page_html(at), "Filtreli iken fark rozeti yok"
at.pills(key="f_products").set_value([4])
at.segmented_control(key="f_balance").set_value("= 0")  # Almanya'da bakiye=0 yok -> boş
run(at)
assert any("müşteri yok" in i.value for i in at.info), "Boş filtre mesajı yok"
print("Genel Bakış (filtreler, renk kuralı) OK")

# Müşteri Risk Skoru: her model için (kenar çubuğundaki ortak model seçimi)
at = open_page(at, "sayfalar/risk_skoru.py")
for model in ["lr", "rf", "xgb"]:
    at.selectbox(key="_w_model").set_value(model)
    run(at)
    assert "Uyarı listesi" in page_html(at)
    print(model, "-> risk sayfası OK")
at.select_slider(key="_w_thr_xgb").set_value(0.5)
run(at)
assert at.session_state["iris_thr"]["xgb"] == 0.5, "Ortak eşik kaydedilmedi"
top_id = at.pills(key="risk_pick").options[1]
at.pills(key="risk_pick").set_value(int(top_id.split(" · ")[0]))
run(at)
assert str(at.session_state["risk_cid"]) == top_id.split(" · ")[0], "Hızlı seçim çalışmadı"
at.segmented_control(key="risk_mode").set_value("Elle değer gir")
run(at)
print("Hızlı seçim ve elle giriş OK")

# Ortak ayar sayfalar arasında korunmalı (giriş sayfası widget çizmez)
at = open_page(at, "sayfalar/giris.py")
at = open_page(at, "sayfalar/genel_bakis.py")
assert at.session_state["iris_model"] == "xgb" and at.session_state["iris_thr"]["xgb"] == 0.5
print("Ortak model/eşik ayarı OK")

# Tema seçimi: giriş sayfasındaki küçük denetim → Koyu; sayfa değişince korunmalı,
# yerleşik temayı değiştiren betik (localStorage + çerez + yenileme) yalnız bir kez gönderilmeli
at = open_page(at, "sayfalar/giris.py")
assert at.session_state["iris_tema"] == "system"
assert len(at.segmented_control(key="_w_tema").options) == 3  # simgeli: Sistem / Açık / Koyu
assert "#0A1522" not in page_html(at), "Varsayılan (sistem → AppTest'te açık) tema koyu görünüyor"
at.segmented_control(key="_w_tema").set_value("dark")
run(at)
html = page_html(at)
assert at.session_state["iris_tema"] == "dark", "Tema seçimi kaydedilmedi"
assert "stActiveTheme-" in html and "'Dark'" in html and "location.reload" in html, \
    "Yerleşik temayı değiştiren betik yok"
assert "iris_tema=dark" in html and "iris_ayar=xgb|" in html and "xgb:0.50" in html, \
    "Tema/ayar çerezleri yazılmıyor"
assert "#0A1522" in html, "Giriş sayfası koyu tasarıma geçmedi"
run(at)  # sonraki çalıştırmada betik yeniden gönderilmemeli, seçim kalmalı
assert "location.reload" not in page_html(at) and at.session_state["iris_tema"] == "dark"
at = open_page(at, "sayfalar/genel_bakis.py")
html = page_html(at)
assert at.segmented_control(key="_w_tema").value == "dark", "Kenar çubuğu denetimi seçimi göstermiyor"
assert "--iris-bg:#0D1929" in html, "İç sayfa CSS'i koyu temaya geçmedi"
assert at.session_state["iris_model"] == "xgb" and at.session_state["iris_thr"]["xgb"] == 0.5
at.segmented_control(key="_w_tema").set_value("light")
run(at)
assert "'Light'" in page_html(at) and "--iris-bg:#F7FAFC" in page_html(at)
at.segmented_control(key="_w_tema").set_value("system")
run(at)
assert at.session_state["iris_tema"] == "system" and "'System'" in page_html(at)
print("Tema seçimi (Sistem / Açık / Koyu) OK")

# Maliyet: farklı maliyetlerle yeniden çalışmalı, "Bu eşiği uygula" ortak eşiği değiştirmeli
at = open_page(at, "sayfalar/maliyet.py")
at.number_input(key="cost_fn").set_value(10)
at.number_input(key="cost_fp").set_value(2)
run(at)
opt, _ = analysis.cost_optimal_threshold(art["val_preds"]["churn"], art["val_preds"]["proba_xgb"], 10, 2)
at.button(key="apply_opt").click()
run(at)
assert abs(at.session_state["iris_thr"]["xgb"] - opt) < 1e-9, "Bu eşiği uygula çalışmadı"
print(f"Maliyet (FN=10, FP=2) OK · uygulanan eşik {opt:.2f}")

# Model karşılaştırma: metrik tablosu, McNemar, kalibrasyon
at = open_page(at, "sayfalar/model_karsilastirma.py")
html = page_html(at)
assert "iris-table" in html and "Holm" in html and "Brier" in html and "Kalibrasyon" in html
at.segmented_control(key="mk_thr").set_value("0,50")
at.segmented_control(key="cal_model").set_value("lr")
run(at)
print("Model karşılaştırma + kalibrasyon OK")

# Açıklanabilirlik: Random Forest + bağımlılık değişkeni
at = open_page(at, "sayfalar/aciklanabilirlik.py")
at.segmented_control(key="shap_model").set_value("rf")
run(at)
rf_top = pd.Series(np.abs(art["shap"]["rf"]["values"]).mean(axis=0),
                   index=art["features"]).sort_values(ascending=False).index
at.selectbox(key="shap_dep").set_value(rf_top[2])  # ham değer (select_index format_func'la bozuluyor)
run(at)
print("Açıklanabilirlik OK")

# Confounding: otomatik yorum üretilmiş olmalı
at = open_page(at, "sayfalar/confounding.py")
assert "odds ratio" in page_html(at), "Confounding yorumu bulunamadı"
print("Confounding yorumu OK")

# Duyarlılık analizi (SMOTE): artifacts/smote_karsilastirma.json'dan okunur
at = open_page(at, "sayfalar/smote_duyarlilik.py")
html = page_html(at)
assert "McNemar p = 0,73" in html and "iris-table" in html, "SMOTE bulgusu/tablosu eksik"
print("Duyarlılık (SMOTE) OK")

# Sunum modu: tüm adımlar Sonraki ile gezilir, son adımda başa dönülür
at = open_page(at, "sayfalar/sunum_modu.py")
n_steps = 0
while True:
    n_steps += 1
    assert f"Soru {n_steps} / " in page_html(at), f"Sunum adımı {n_steps} çizilmedi"
    last = at.button(key="sunum_next").label == "Başa dön"
    at.button(key="sunum_next").click()
    run(at)
    if last:
        break
assert at.session_state["sunum_adim"] == 0 and n_steps >= 6
at.button(key="sunum_next").click()
run(at)
at.button(key="sunum_prev").click()
run(at)
assert at.session_state["sunum_adim"] == 0
print(f"Sunum modu OK ({n_steps} adım)")
print("SMOKE TEST OK")
