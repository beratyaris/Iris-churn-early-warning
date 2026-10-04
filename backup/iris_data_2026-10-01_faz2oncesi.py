"""Iris - veri/artefakt yükleme ve tüm sayfaların paylaştığı model ayarları.

Bu modül eğitim yapmaz; yalnızca data/ ve artifacts/ içindeki gerçek çıktıları okur.
Gösterilen tüm sayılar buradan (ya da doğrudan veri setinden) hesaplanır.
"""
import json

import joblib
import pandas as pd
import shap
import streamlit as st

from train import DATA, MODEL_NAMES, OUT

# Portföy risk bantları (giriş sayfasındaki gösterge): uyarı verilen müşteri payı
RISK_BANDS = [(1 / 3, "Düşük"), (2 / 3, "Orta"), (1.01, "Yüksek")]


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
    }


@st.cache_resource
def get_explainer(key, _pipe, _background):
    """Ağaç modelleri için TreeExplainer, Lojistik Regresyon için LinearExplainer."""
    model = _pipe.named_steps["model"]
    if key == "lr":
        bg = _pipe.named_steps["scaler"].transform(_background)
        return shap.LinearExplainer(model, shap.maskers.Independent(bg, max_samples=len(bg)))
    return shap.TreeExplainer(model)


def explain_row(key, art, x_raw):
    """Tek müşteri için SHAP açıklaması. Model ölçeklenmiş veriyle çalışır,
    grafikte ise müşterinin ham değerleri gösterilir."""
    pipe = art["models"][key]
    explainer = get_explainer(key, pipe, art["background"])
    x_scaled = pipe.named_steps["scaler"].transform(x_raw)
    sv = explainer(x_scaled)
    values, base = sv.values[0], sv.base_values[0]
    if values.ndim == 2:  # Random Forest: sınıf 1 (churn) sütunu
        values, base = values[:, 1], base[1]
    return shap.Explanation(values=values, base_values=float(base),
                            data=x_raw.iloc[0].to_numpy(), feature_names=list(x_raw.columns))


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
    key = ss["iris_model"]
    return key, float(ss["iris_thr"][key])


def _copy_model():
    st.session_state["iris_model"] = st.session_state["_w_model"]


def _copy_thr(key):
    st.session_state["iris_thr"][key] = st.session_state[f"_w_thr_{key}"]


def _reset_thr(key, value):
    st.session_state["iris_thr"][key] = value
    st.session_state.pop(f"_w_thr_{key}", None)  # slider varsayılandan yeniden kurulur


def settings_sidebar(art):
    """Kenar çubuğunda model ve karar eşiği; tüm iç sayfalar aynı değeri kullanır."""
    key, thr = current_settings(art)
    opt = float(art["thresholds"][key])
    with st.sidebar:
        st.markdown('<div class="iris-side-label">Model ayarı</div>', unsafe_allow_html=True)
        st.selectbox("Model", list(MODEL_NAMES), index=list(MODEL_NAMES).index(key),
                     format_func=MODEL_NAMES.get, key="_w_model", on_change=_copy_model)
        key, thr = current_settings(art)
        opt = float(art["thresholds"][key])
        st.slider("Karar eşiği (olasılık ≥ eşik → uyarı)", 0.05, 0.95, thr, 0.01,
                  key=f"_w_thr_{key}", on_change=_copy_thr, args=(key,),
                  help=f"Varsayılan: validation setinde F2'yi maksimize eden eşik ({opt:.2f})")
        key, thr = current_settings(art)
        if abs(thr - opt) > 1e-9:
            st.button(f"F2-optimal eşiğe dön ({opt:.2f})", on_click=_reset_thr,
                      args=(key, opt), width="stretch")
        st.caption("Bu ayar Genel Bakış, Müşteri Risk Skoru ve giriş göstergesinde ortaktır.")
    return key, thr


def portfolio_risk(art, key, thr):
    """Portföy risk seviyesi = seçili modelin test setinde (eğitimde görülmemiş 2.000
    müşteri) eşik üstünde işaretlediği müşteri payı."""
    p = art["preds"][f"proba_{key}"]
    alert = p >= thr
    share = float(alert.mean())
    band = next(label for limit, label in RISK_BANDS if share < limit)
    return {"share": share, "n_alert": int(alert.sum()), "n_test": len(p), "band": band,
            "mean_proba": float(p.mean())}
