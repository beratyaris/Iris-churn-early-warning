r"""Iris - Banka Müşteri Terki (Churn) Erken Uyarı Sistemi (Streamlit, çok sayfalı).

Önce modeller eğitilmeli:  .\.venv\Scripts\python.exe train.py
Sonra:                     .\.venv\Scripts\streamlit.exe run app.py
Bu dosya eğitim yapmaz; yalnızca artifacts/ klasöründeki çıktıları okur.

Giriş noktası: sayfaları (sayfalar/*.py) tanımlar, ortak temayı, logoyu ve kenar
çubuğundaki model/eşik ayarını kurar, seçili sayfayı çalıştırır.
"""
import streamlit as st

import iris_ui as ui
from iris_data import artifacts_ready, load_artifacts, settings_sidebar

PAGES = {
    "": [st.Page("sayfalar/giris.py", title="Giriş", icon=":material/home:", default=True)],
    "Karar Destek": [
        st.Page("sayfalar/genel_bakis.py", title="Genel Bakış", icon=":material/dashboard:"),
        st.Page("sayfalar/risk_skoru.py", title="Müşteri Risk Skoru",
                icon=":material/person_search:"),
        st.Page("sayfalar/maliyet.py", title="Maliyet ve Eşik", icon=":material/payments:"),
    ],
    "Analiz": [
        st.Page("sayfalar/model_karsilastirma.py", title="Model Karşılaştırma",
                icon=":material/leaderboard:"),
        st.Page("sayfalar/aciklanabilirlik.py", title="Açıklanabilirlik (SHAP)",
                icon=":material/psychology:"),
        st.Page("sayfalar/confounding.py", title="Confounding", icon=":material/account_tree:"),
    ],
}


def main():
    st.set_page_config(page_title="Iris · Churn Erken Uyarı", page_icon=str(ui.LOGO / "favicon-32.png"),
                       layout="wide", initial_sidebar_state="expanded")
    page = st.navigation(PAGES)
    ui.inject_base_css()
    if not artifacts_ready():
        st.error("Model dosyaları bulunamadı. Önce `train.py` çalıştırın.")
        st.stop()
    if page.title == "Giriş":
        ui.inject_landing_css()
    else:
        st.logo(str(ui.LOGO / "iris-logo-sidebar-light.png"),
                icon_image=str(ui.LOGO / "iris-icon-sidebar-light.png"), size="medium")
        settings_sidebar(load_artifacts())
    page.run()


main()
