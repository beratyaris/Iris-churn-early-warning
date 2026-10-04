"""Iris - veri/artefakt yükleme, kalibrasyon ve tüm sayfaların paylaştığı model ayarları.

Bu modül eğitim yapmaz; yalnızca data/ ve artifacts/ içindeki gerçek çıktıları okur.
Gösterilen tüm sayılar buradan (ya da doğrudan veri setinden) hesaplanır.

Olasılık ölçekleri:
- "Skor" = modelin ham çıktısı (SMOTE-NC ile eğitildiği için yukarı kaymış). Karar eşikleri
  (thresholds.json) bu ölçekte seçildi; uyarı listesi / sınıflandırma HER ZAMAN skor ≥ eşik.
- "Olasılık" = calibrate.py'nin validation setinde eğittiği kalibratörden geçmiş değer.
  Yalnız beklenen churn oranı ve ekranda gösterilen müşteri olasılığı için kullanılır.
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

import iris_shap
from train import DATA, MODEL_NAMES, OUT, SEED

Z95 = float(norm.ppf(0.975))
THR_GRID = [round(float(x), 2) for x in np.arange(0.05, 0.955, 0.01)]

# Değişkenlerin Türkçe adları (grafik ve tablolarda)
FEATURE_TR = {
    "credit_score": "Kredi skoru", "age": "Yaş", "tenure": "Müşteri süresi (yıl)",
    "balance": "Bakiye", "products_number": "Ürün sayısı", "credit_card": "Kredi kartı",
    "active_member": "Aktif üye", "estimated_salary": "Tahmini maaş",
    "has_balance": "Bakiyesi var", "balance_salary_ratio": "Bakiye / maaş",
    "country_Germany": "Ülke: Almanya", "country_Spain": "Ülke: İspanya",
    "gender_Male": "Cinsiyet: Erkek",
}
BINARY = {"credit_card", "active_member", "has_balance", "country_Germany", "country_Spain",
          "gender_Male"}


def fmt_feature(name, value):
    """Değişken değerini Türkçe biçimde yazar (ör. Bakiye = 125.510, Aktif üye = Hayır)."""
    from iris_ui import fmt_num
    if name in BINARY:
        return "Evet" if value >= 0.5 else "Hayır"
    if name in ("balance", "estimated_salary"):
        return fmt_num(value, 0)
    if name == "balance_salary_ratio":
        return fmt_num(value, 2)
    return fmt_num(value, 0)


# ---------- İstatistik yardımcıları ----------
def wilson(k, n, z=Z95):
    """Oran için Wilson skor güven aralığı (alt, üst)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def score_band(p0, n, z=Z95):
    """Oranı p0 olan n kişilik bir grupta gözlenen oranın %95 aralığı (skor testinin kabul
    bölgesi; Wilson aralığı bu testin tersine çevrilmesidir): p0 ± z·√(p0(1−p0)/n)."""
    half = z * np.sqrt(p0 * (1 - p0) / n)
    return p0 - half, p0 + half


# ---------- Yükleme ----------
def artifacts_ready():
    return (OUT / "models.joblib").exists()


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


@st.cache_resource
def load_artifacts():
    with open(OUT / "thresholds.json", encoding="utf-8") as f:
        thr = json.load(f)
    cal = None
    if (OUT / "calibration.json").exists() and (OUT / "calibration.joblib").exists():
        with open(OUT / "calibration.json", encoding="utf-8") as f:
            cal = {"summary": json.load(f), "models": joblib.load(OUT / "calibration.joblib")}
    return {
        "models": joblib.load(OUT / "models.joblib"),
        "shap": joblib.load(OUT / "shap_test.joblib"),
        "background": joblib.load(OUT / "background.joblib"),
        "metrics": pd.read_csv(OUT / "metrics.csv"),
        "mcnemar": pd.read_csv(OUT / "mcnemar.csv"),
        "preds": pd.read_csv(OUT / "test_predictions.csv"),
        "val_preds": pd.read_csv(OUT / "val_predictions.csv"),
        "features": thr["feature_names"],
        "thresholds": {k: thr[k]["threshold"] for k in MODEL_NAMES},
        "calibration": cal,
    }


# ---------- Kalibrasyon ----------
def is_calibrated(art, key):
    cal = art.get("calibration")
    return bool(cal and cal["summary"]["models"][key]["applied"])


def calibrate(art, key, scores):
    """Ham skor → kalibre olasılık. Kalibrasyon yoksa ya da gerekmediyse skor aynen döner."""
    s = np.asarray(scores, dtype=float)
    if not is_calibrated(art, key):
        return s
    return np.clip(art["calibration"]["models"][key].predict(s), 0, 1)


def threshold_as_proba(art, key, thr):
    """Skor eşiğinin kalibre olasılık ölçeğindeki karşılığı. Kalibratör monoton artan olduğu
    için 'skor ≥ eşik' ile 'olasılık ≥ bu değer' aynı müşterileri seçer."""
    return float(calibrate(art, key, [thr])[0])


def expected_churn(art, key, base_rate):
    """Beklenen churn oranı = test setinde (2.000) kalibre olasılıkların ortalaması (eşikten
    bağımsız). Bantlar: tarihsel oran için n kişilik %95 skor aralığı."""
    preds = art["preds"]
    raw = preds[f"proba_{key}"].to_numpy()
    p = calibrate(art, key, raw)
    n = len(p)
    lo, hi = score_band(base_rate, n)
    value = float(p.mean())
    cls = "below" if value < lo else ("above" if value > hi else "near")
    return {"value": value, "raw_mean": float(raw.mean()), "actual": float(preds["churn"].mean()),
            "n": n, "lo": lo, "hi": hi, "cls": cls, "calibrated": is_calibrated(art, key)}


def alert_list(art, key, thr):
    """Uyarı listesi: test setinde ham skoru eşiğe eşit/üstünde olan müşteriler."""
    p = art["preds"][f"proba_{key}"]
    alert = p >= thr
    return {"n_alert": int(alert.sum()), "n_test": len(p), "share": float(alert.mean())}


# ---------- SHAP ----------
# shap paketi kullanılmaz (numba bağımlılığı Akıllı Uygulama Denetimi'ne takılıyor). Değerler
# iris_shap ile hesaplanır: ağaçlar XGBoost'un yerleşik TreeSHAP'i (pred_contribs), LR doğrusal
# formül; önceki TreeExplainer / LinearExplainer ayarlarıyla aynı sonuç (bkz. iris_shap.py).
@st.cache_resource
def get_booster(key, _pipe):
    """Ağaç modelleri için TreeSHAP'i çalıştıran XGBoost Booster (RF bir kez dönüştürülür)."""
    return iris_shap.booster_for(key, _pipe)


def explain_row(key, art, x_raw):
    """Tek müşteri için SHAP açıklaması. Model ölçeklenmiş veriyle çalışır,
    grafikte ise müşterinin ham değerleri gösterilir."""
    pipe = art["models"][key]
    booster = None if key == "lr" else get_booster(key, pipe)
    values, base = iris_shap.contribs(key, pipe, x_raw, background=art["background"], booster=booster)
    return iris_shap.Explanation(values=values[0], base_values=float(base[0]),
                                 data=x_raw.iloc[0].to_numpy(), feature_names=list(x_raw.columns))


SHAP_UNIT = {"lr": "log-odds", "xgb": "log-odds", "rf": "olasılık"}


@st.cache_data
def train_raw():
    """Modelin eğitildiği 6.000 müşteri (ham veri biçiminde). train.py'deki bölme birebir
    yeniden üretilir (aynı seed ve tabakalama); test sırası test_predictions.csv ile aynıdır."""
    from sklearn.model_selection import train_test_split
    df = load_data()
    trval, _ = train_test_split(df, test_size=0.20, stratify=df["churn"], random_state=SEED)
    train, _ = train_test_split(trval, test_size=0.25, stratify=trval["churn"], random_state=SEED)
    return train.reset_index(drop=True)


@st.cache_data
def input_bounds():
    """Simülatör/elle giriş sınırları: eğitim verisinde gerçekten görülen aralık. Para
    değerleri tam sayıya, aralığın İÇİNE yuvarlanır (ör. maaş 90,07 → 91)."""
    t = train_raw()
    b = {c: (int(t[c].min()), int(t[c].max())) for c in ("credit_score", "age", "tenure", "products_number")}
    for c in ("balance", "estimated_salary"):
        b[c] = (int(np.ceil(t[c].min())), int(np.floor(t[c].max())))
    b["country"] = sorted(t["country"].unique())
    b["gender"] = sorted(t["gender"].unique())
    return b


@st.cache_data
def rarity_index():
    """Nadirlik kontrolü için eğitim setindeki grup sayımları (iris_karar.build_rarity_index)."""
    from iris_karar import build_rarity_index
    return build_rarity_index(train_raw())


def test_frame(art):
    """Test tahminlerine ham veri benzeri kolonlar ekler (filtreleme için)."""
    p = art["preds"].copy()
    p["country"] = "France"
    p.loc[p["country_Germany"] == 1, "country"] = "Germany"
    p.loc[p["country_Spain"] == 1, "country"] = "Spain"
    p["gender"] = p["gender_Male"].map({1: "Male", 0: "Female"})
    return p


# ---------- Paylaşılan model / eşik ayarı ----------
# Widget'lar yalnız iç sayfalarda çizilir; değerler kalıcı anahtarlarda (iris_model,
# iris_thr) tutulur ki giriş sayfasına gidip dönünce ayar kaybolmasın.
def current_settings(art):
    ss = st.session_state
    ss.setdefault("iris_model", "xgb")
    ss.setdefault("iris_thr", dict(art["thresholds"]))
    for k, v in art["thresholds"].items():  # tema yenilemesinden geri yüklenen ayar eksikse tamamla
        ss["iris_thr"].setdefault(k, v)
    if ss["iris_model"] not in art["thresholds"]:
        ss["iris_model"] = "xgb"
    key = ss["iris_model"]
    return key, float(ss["iris_thr"][key])


def set_threshold(key, value):
    """Ortak eşiği değiştirir (ör. Maliyet sayfasındaki 'Bu eşiği uygula')."""
    st.session_state["iris_thr"][key] = round(float(value), 2)
    st.session_state.pop(f"_w_thr_{key}", None)  # kaydırıcı yeni değerden yeniden kurulur


def _copy_model():
    st.session_state["iris_model"] = st.session_state["_w_model"]


def _copy_thr(key):
    st.session_state["iris_thr"][key] = st.session_state[f"_w_thr_{key}"]


def settings_sidebar(art):
    """Kenar çubuğunda model ve karar eşiği; tüm iç sayfalar aynı değeri kullanır."""
    from iris_ui import fmt_num
    key, thr = current_settings(art)
    with st.sidebar:
        st.markdown('<div class="iris-side-label">Model ayarı</div>', unsafe_allow_html=True)
        st.selectbox("Model", list(MODEL_NAMES), index=list(MODEL_NAMES).index(key),
                     format_func=MODEL_NAMES.get, key="_w_model", on_change=_copy_model)
        key, thr = current_settings(art)
        opt = float(art["thresholds"][key])
        value = min(THR_GRID, key=lambda g: abs(g - thr))
        st.select_slider("Karar eşiği (skor ≥ eşik → uyarı listesi)", THR_GRID, value=value,
                         format_func=lambda v: fmt_num(v, 2), key=f"_w_thr_{key}",
                         on_change=_copy_thr, args=(key,),
                         help=f"Varsayılan: validation setinde F2'yi maksimize eden eşik "
                              f"({fmt_num(opt, 2)}). Eşik modelin ham skoru üzerindedir.")
        key, thr = current_settings(art)
        if abs(thr - opt) > 1e-9:
            st.button(f"F2-optimal eşiğe dön ({fmt_num(opt, 2)})", on_click=set_threshold,
                      args=(key, opt), width="stretch")
        al = alert_list(art, key, thr)
        st.html(f'<div class="iris-side-note">Uyarı listesi (test): <b>{fmt_num(al["n_alert"])}</b> '
                f'müşteri. Bu ayar tüm sayfalarda ve giriş ekranında ortaktır.</div>')
    return key, thr
