r"""Arayüz için hızlı duman testi: analiz/kalibrasyon yardımcılarını ve tüm sayfaları çalıştırır.

Çalıştırma: .\.venv\Scripts\python.exe smoke_test.py
(Önce train.py ve calibrate.py çalışmış olmalı.)
"""
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import sys

import analysis
import iris_data as idata
import iris_karar as K
import iris_shap
import iris_ui as ui
from train import DATA, MODEL_NAMES

# Uygulamadaki tüm sayfalar (iki temada açılış kontrolü için)
ALL_PAGES = ["sayfalar/giris.py", "sayfalar/sunum_modu.py", "sayfalar/genel_bakis.py",
             "sayfalar/risk_skoru.py", "sayfalar/maliyet.py", "sayfalar/simulator.py",
             "sayfalar/kampanya.py", "sayfalar/model_karsilastirma.py", "sayfalar/smote_duyarlilik.py",
             "sayfalar/model_guven.py", "sayfalar/aciklanabilirlik.py", "sayfalar/confounding.py"]


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

# ---------- SHAP (shap paketi olmadan; iris_shap) ----------
X_test = art["preds"][art["features"]]
for k in ["xgb", "rf"]:  # kayıtlı (shap 0.52 TreeExplainer) değerlerle aynı olmalı
    vals, base = iris_shap.contribs(k, art["models"][k], X_test.iloc[:200], booster=idata.get_booster(k, art["models"][k]))
    d = float(np.abs(vals - np.asarray(art["shap"][k]["values"])[:200]).max())
    assert d < 1e-5 and abs(base[0] - art["shap"][k]["base"]) < 1e-5, f"{k}: SHAP farkı {d:.2e}"
    print(f"{MODEL_NAMES[k]}: SHAP kayıtlı değerlerle maks. fark {d:.1e}")
for k in MODEL_NAMES:  # taban + katkılar = modelin çıktısı (XGB/LR log-odds, RF olasılık)
    xr = X_test.iloc[[7]]
    e = idata.explain_row(k, art, xr)
    pr = float(art["models"][k].predict_proba(xr)[0, 1])
    out = pr if idata.SHAP_UNIT[k] == "olasılık" else np.log(pr / (1 - pr))
    assert abs(e.base_values + e.values.sum() - out) < 1e-4, f"{k}: SHAP toplamı model çıktısına eşit değil"
print("SHAP (XGBoost pred_contribs) OK")

# ---------- Tasarım modülü ----------
assert ui.fmt_num(10000) == "10.000" and ui.fmt_pct(0.2037, 2) == "%20,37"
assert ui.fmt_p(0.0004) == "< 0,001" and ui.fmt_p(0.0342) == "0,034"
assert ui.risk_color(0) == ui.TURQ.upper() and ui.risk_color(1) == ui.CORAL.upper()
svg = ui.gauge_svg(0.2, 0.4, [(0, .18, ui.TURQ), (.18, .22, ui.AMBER), (.22, .4, ui.CORAL)],
                   [(0, "%0"), (.2, "%20"), (.4, "%40")], marker=(.21, "eşik"))
assert svg.startswith("<img") and "base64" in svg
assert "Mini grafik" not in ui.kpi_card("x", "1", tip="açıklama")
print("Tasarım modülü OK")

# ---------- Karar Destek hesapları (iris_karar: nadirlik + kampanya) ----------
tr = idata.train_raw()
assert len(tr) == 6000 and not set(tr["customer_id"]) & set(art["preds"]["customer_id"]), \
    "Eğitim bölmesi yeniden üretilemedi (test müşterileriyle çakışma)"
bounds = idata.input_bounds()
assert bounds["age"] == (int(tr["age"].min()), int(tr["age"].max()))
assert bounds["balance"][1] <= tr["balance"].max() and bounds["estimated_salary"][0] >= tr["estimated_salary"].min()
assert bounds["country"] == ["France", "Germany", "Spain"]
ridx = idata.rarity_index()
prof = dict(credit_score=650, country="Germany", gender="Female", age=40, tenure=5, balance=0,
            products_number=1, estimated_salary=100000, credit_card=1, active_member=1)
r = K.rarity_check(ridx, prof)
assert r["level"] == "yok" and r["issues"][0]["cols"] == ("country", "balance") and r["issues"][0]["n"] == 0, \
    "Almanya × bakiye 0 'yok' olarak işaretlenmedi"
assert K.rarity_check(ridx, {**prof, "country": "France"})["level"] == "ok"
assert K.rarity_check(ridx, {**prof, "country": "France", "products_number": 4})["level"] == "nadir"
test_raw = idata.load_data().set_index("customer_id").loc[art["preds"]["customer_id"]].reset_index()
levels = pd.Series([K.rarity_check(ridx, p)["level"] for p in test_raw.to_dict("records")])
assert (levels != "ok").mean() < 0.05, "Nadirlik kontrolü gerçek test müşterilerinin çoğunu işaretliyor"
print(f"Nadirlik kontrolü OK (test setinde işaretlenen: {(levels != 'ok').mean():.2%})")
sc = art["preds"]["proba_xgb"].to_numpy()
pc = idata.calibrate(art, "xgb", sc)
t0 = art["thresholds"]["xgb"]
cur = K.campaign_curve(sc, pc, K.DEFAULT_COST_FP, K.DEFAULT_SUCCESS, K.DEFAULT_COST_FN)
n_al = int((sc >= t0).sum())
assert abs(cur["beklenen_terk"].iloc[n_al] - pc[sc >= t0].sum()) < 1e-6, "Kampanya: ilk N ≠ uyarı listesi"
assert abs(cur["beklenen_terk"].iloc[-1] - pc.sum()) < 1e-6 and cur["net"].iloc[0] == 0
row = cur.iloc[100]
assert abs(row["net"] - (row["beklenen_terk"] * K.DEFAULT_SUCCESS * K.DEFAULT_COST_FN - 100 * K.DEFAULT_COST_FP)) < 1e-9
assert K.reach_for_budget(n_al, 1, 2000) == n_al and K.reach_for_budget(10 ** 6, 1, 2000) == 2000
assert K.reach_for_budget(7, 2, 2000) == 3
print(f"Kampanya hesabı OK (uyarı listesi {n_al}: beklenen terk eden {pc[sc >= t0].sum():.1f}, "
      f"en iyi N {int(cur['net'].idxmax())})")

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
hub_links = [e.proto.label for e in at.get("page_link")]
for lab in ["Sunum Modu", "Model ne derdi?", "Kampanya Planlayıcı", "Modele güvenebilir miyiz?"]:
    assert lab in hub_links, f"Giriş sayfasında '{lab}' bağlantısı yok"
assert len(hub_links) == 10, hub_links
print(f"Giriş sayfası OK (beklenen churn {exp['value']:.4f}, uyarı listesi {n_alert}, {len(hub_links)} bağlantı)")

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
assert at.number_input(key="m_a").max == idata.input_bounds()["age"][1], "Elle giriş sınırı eğitim verisinden değil"
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

# "Model ne derdi?" simülatörü: sınırlar, sabit not, nadirlik uyarısı, başlangıç profili
at = open_page(at, "sayfalar/simulator.py")
html = page_html(at)
assert "müdahalenin etkisini göstermez" in html and "Uyarı listesinde" in html
assert "log-odds biriminde; artı riski artırır, eksi düşürür" in html, "SHAP okuma ipucu yok"
assert "temsil ediliyor" in html, "Varsayılan profil nadir görünüyor"
assert at.slider(key="s_a").min == bounds["age"][0] and at.slider(key="s_a").max == bounds["age"][1]
assert at.number_input(key="s_b").max == bounds["balance"][1], "Bakiye sınırı eğitim verisinden değil"
at.selectbox(key="s_c").set_value("Germany")
at.number_input(key="s_b").set_value(0)
run(at)
assert "eğitim verisinde yok; tahmin güvenilir değil" in page_html(at), "Almanya × bakiye 0 uyarısı yok"
cid = int(art["preds"].sort_values("proba_xgb", ascending=False)["customer_id"].iloc[3])
at.selectbox(key="sim_start").set_value(cid)
run(at)
crow = art["preds"].loc[art["preds"]["customer_id"] == cid].iloc[0]
assert at.session_state["s_a"] == int(crow["age"]) and str(cid) in at.session_state["sim_base"]["label"]
at.slider(key="s_a").set_value(bounds["age"][0])
run(at)
assert "Başlangıç (Test müşterisi" in page_html(at)
at.button(key="sim_reset").click()
run(at)
assert "Profil başlangıçla aynı" in page_html(at), "Başlangıca dön çalışmadı"
print("Simülatör OK")

# Kampanya planlayıcı: varsayılan bütçe = uyarı listesi, müşteri sayısı modu, sıfır başarı
at = open_page(at, "sayfalar/kampanya.py")
thr_now = at.session_state["iris_thr"]["xgb"]
n_now = int((sc >= thr_now).sum())
html = page_html(at)
assert f'data-countup="{n_now:.6f}"' in html and "Uyarı listesindeki" in html and "varsayım" in html
at.segmented_control(key="camp_mode").set_value("Müşteri sayısı")
run(at)
at.number_input(key="camp_n").set_value(360)
run(at)
assert 'data-countup="360.000000"' in page_html(at)
at.slider(key="camp_success").set_value(0)
run(at)
assert any("pozitif net fayda üretmiyor" in c.value for c in at.caption)
print("Kampanya planlayıcı OK")

# Modele güvenebilir miyiz?: bulgu, Brier, yasak kelime yok
at = open_page(at, "sayfalar/model_guven.py")
html = page_html(at)
assert "SMOTE olasılıkları yukarı şişirir" in html and "Sıralama ve uyarı listesi değişmez" in html
assert ui.fmt_num(art["calibration"]["summary"]["models"]["xgb"]["brier_test_cal"], 4) in html
assert "doğruluk" not in html.lower() and "accuracy" not in html.lower()
print("Modele güvenebilir miyiz OK")

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
assert "turkuaz = düşük, amber = orta, mercan = yüksek" in page_html(at), "Beeswarm renk açıklaması yok"
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

# Tüm sayfalar iki temada hatasız açılmalı
at = open_page(at, "sayfalar/genel_bakis.py")
for theme, bg in [("dark", "#0D1929"), ("light", "#F7FAFC")]:
    at.segmented_control(key="_w_tema").set_value(theme)
    run(at)
    for page in ALL_PAGES:
        at = open_page(at, page)
        if page != "sayfalar/giris.py":
            assert f"--iris-bg:{bg}" in page_html(at), f"{page}: {theme} teması uygulanmadı"
    at = open_page(at, "sayfalar/genel_bakis.py")
    print(f"Tüm sayfalar ({len(ALL_PAGES)}) {theme} temada OK")
# shap -> numba zinciri Akıllı Uygulama Denetimi'ne takılıyor: hiçbir modül yüklememeli
bad = sorted(m for m in sys.modules if m.split(".")[0] in ("shap", "numba", "llvmlite"))
assert not bad, f"shap/numba içe aktarıldı: {bad[:5]}"
print("shap/numba içe aktarılmadı OK")
print("SMOKE TEST OK")
