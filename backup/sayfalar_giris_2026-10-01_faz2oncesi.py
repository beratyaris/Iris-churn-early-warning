"""Giriş / hub sayfası: solda portföy risk göstergesi, sağda özet ve sayfa düğmeleri."""
import streamlit as st

import iris_ui as ui
from iris_data import current_settings, load_artifacts, load_data, portfolio_risk
from train import MODEL_NAMES

df, art = load_data(), load_artifacts()
key, thr = current_settings(art)
risk = portfolio_risk(art, key, thr)
band_color = ui.risk_color(risk["share"])

hero, hub = st.columns([55, 45], gap=None)

with hero:
    with st.container(key="hero"):
        ui.html(f"""
<div class="iris-hero">
  {ui.gauge_svg(risk["share"])}
  <div class="h-kicker">Portföy risk seviyesi</div>
  <div class="h-value">{ui.count(risk["share"] * 100, 1, pre="%", dur=1500)}</div>
  <div class="h-band"><i style="background:{band_color}"></i>{risk["band"]} risk ·
    {ui.fmt_num(risk["n_alert"])} / {ui.fmt_num(risk["n_test"])} müşteri uyarıda</div>
  <div class="h-def">Gösterge, <b>{MODEL_NAMES[key]}</b> modelinin karar eşiği
    <b>{ui.fmt_num(thr, 2)}</b> ile test setindeki (eğitimde görülmemiş)
    {ui.fmt_num(risk["n_test"])} müşterinin ne kadarını yüksek riskli işaretlediğini gösterir.
    Bantlar: %33'ün altı düşük, %33–66 orta, üstü yüksek. Eşik F2 (recall ağırlıklı)
    ölçütüyle seçildiği için uyarı payı gerçek churn oranından yüksektir.</div>
</div>""", js=True)

with hub:
    with st.container(key="hub"):
        churn_rate = df["churn"].mean()
        ui.html(f"""
<div class="iris-hub-logo">{ui.logo_img("iris-logo-sidebar-dark.svg")}</div>
<div class="iris-hub-title">Müşteri Terki<br>Erken Uyarı Sistemi</div>
<div class="iris-hub-sub">Banka müşterilerinde terk riskini önceden görmek,
  açıklamak ve maliyete göre karar vermek için.</div>
<div class="iris-stats">
  <div class="iris-stat"><div class="s-val">{ui.count(len(df))}</div>
    <div class="s-lab">Müşteri</div></div>
  <div class="iris-stat"><div class="s-val">{ui.count(churn_rate * 100, 2, pre="%")}</div>
    <div class="s-lab"><span class="s-dot" style="background:{ui.rate_color(churn_rate)}"></span>Churn oranı</div></div>
  <div class="iris-stat"><div class="s-val">{ui.count(risk["n_alert"])}</div>
    <div class="s-lab"><span class="s-dot" style="background:{ui.CORAL}"></span>Yüksek riskli (test)</div></div>
</div>""", js=True)
        ui.html('<div class="iris-group">Karar Destek</div>')
        with st.container(key="hub-karar", gap="small"):
            st.page_link("sayfalar/genel_bakis.py", label="Genel Bakış",
                         icon=":material/dashboard:", width="stretch")
            st.page_link("sayfalar/risk_skoru.py", label="Müşteri Risk Skoru",
                         icon=":material/person_search:", width="stretch")
            st.page_link("sayfalar/maliyet.py", label="Maliyet ve Eşik",
                         icon=":material/payments:", width="stretch")
        ui.html('<div class="iris-group">Analiz</div>')
        with st.container(key="hub-analiz", gap="small"):
            st.page_link("sayfalar/model_karsilastirma.py", label="Model Karşılaştırma",
                         icon=":material/leaderboard:", width="stretch")
            st.page_link("sayfalar/aciklanabilirlik.py", label="Açıklanabilirlik (SHAP)",
                         icon=":material/psychology:", width="stretch")
            st.page_link("sayfalar/confounding.py", label="Confounding",
                         icon=":material/account_tree:", width="stretch")
        ui.html(f'<div class="iris-hub-foot">Veri: {ui.fmt_num(len(df))} banka müşterisi · '
                f'Modeller: {", ".join(MODEL_NAMES.values())}</div>')
