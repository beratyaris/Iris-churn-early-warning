r"""Iris - Banka Müşteri Terki (Churn) Erken Uyarı Sistemi (Streamlit, çok sayfalı).

Önce modeller eğitilmeli:  .\.venv\Scripts\python.exe train.py
Sonra:                     .\.venv\Scripts\streamlit.exe run app.py
Bu dosya eğitim yapmaz; yalnızca artifacts/ klasöründeki çıktıları okur.

İsteğe bağlı:              .\.venv\Scripts\python.exe calibrate.py  (olasılık kalibrasyonu)

Giriş noktası: sayfaları (sayfalar/*.py) tanımlar, ortak temayı (açık/koyu), logoyu ve
kenar çubuğundaki model/eşik ayarını kurar, seçili sayfayı çalıştırır.
"""
import streamlit as st

import iris_ui as ui
from iris_data import artifacts_ready, load_artifacts, settings_sidebar

PAGES = {
    "": [st.Page("sayfalar/giris.py", title="Giriş", icon=":material/home:", default=True)],
    "Sunum": [
        st.Page("sayfalar/sunum_modu.py", title="Sunum Modu", icon=":material/co_present:"),
    ],
    "Karar Destek": [
        st.Page("sayfalar/genel_bakis.py", title="Genel Bakış", icon=":material/dashboard:"),
        st.Page("sayfalar/risk_skoru.py", title="Müşteri Risk Skoru",
                icon=":material/person_search:"),
        st.Page("sayfalar/maliyet.py", title="Maliyet ve Eşik", icon=":material/payments:"),
    ],
    "Analiz": [
        st.Page("sayfalar/model_karsilastirma.py", title="Model Karşılaştırma",
                icon=":material/leaderboard:"),
        st.Page("sayfalar/smote_duyarlilik.py", title="Duyarlılık: SMOTE",
                icon=":material/tune:"),
        st.Page("sayfalar/aciklanabilirlik.py", title="Açıklanabilirlik (SHAP)",
                icon=":material/psychology:"),
        st.Page("sayfalar/confounding.py", title="Confounding", icon=":material/account_tree:"),
    ],
}


def main():
    st.set_page_config(page_title="Iris · Churn Erken Uyarı", page_icon=str(ui.LOGO / "favicon-32.png"),
                       layout="wide", initial_sidebar_state="expanded")
    page = st.navigation(PAGES)
    ui.begin_run()
    ui.inject_base_css()
    if not artifacts_ready():
        st.error("Model dosyaları bulunamadı. Önce `train.py` çalıştırın.")
        st.stop()
    if page.title == "Giriş":
        ui.inject_landing_css()
    else:
        ui.sidebar_logo()  # açık/koyu temaya göre logo
        settings_sidebar(load_artifacts())
        with st.sidebar:  # kenar çubuğunun en altında tema seçimi
            st.html('<div class="iris-side-label" style="margin-top:.6rem">Görünüm</div>')
            ui.theme_switch()
    page.run()


main()
