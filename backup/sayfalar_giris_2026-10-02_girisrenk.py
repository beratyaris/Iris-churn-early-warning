"""Giriş / hub sayfası: solda beklenen churn oranı göstergesi ve uyarı listesi, sağda özet
ve sayfa düğmeleri.

Gösterge eşikten bağımsızdır (kalibre olasılıkların ortalaması); uyarı listesi ise eşiğe
bağlı eylem listesidir. İkisi bilerek ayrı gösterilir.
"""
import streamlit as st

import iris_ui as ui
from iris_data import alert_list, current_settings, expected_churn, load_artifacts, load_data
from train import MODEL_NAMES

df, art = load_data(), load_artifacts()
key, thr = current_settings(art)
base_rate = float(df["churn"].mean())
exp = expected_churn(art, key, base_rate)
al = alert_list(art, key, thr)
dark = ui.mode() == "dark"

vmax = 2 * base_rate  # tarihsel oran yayın tepesinde
band_color = ui.REL_COLORS[exp["cls"]]
band_text = {"below": "Tarihsel oranın altında", "near": "Tarihsel orana yakın",
             "above": "Tarihsel oranın üstünde"}[exp["cls"]]
gauge = ui.gauge_svg(
    exp["value"], vmax,
    bands=[(0, exp["lo"], ui.TURQ), (exp["lo"], exp["hi"], ui.AMBER), (exp["hi"], vmax, ui.CORAL)],
    labels=[(0, "%0"), (base_rate, f"{ui.fmt_pct(base_rate)} · tarihsel"), (vmax, ui.fmt_pct(vmax))],
    alt=f"Beklenen churn oranı göstergesi: {ui.fmt_pct(exp['value'])}", on_dark=dark)
cal_note = ("kalibre edilmiş" if exp["calibrated"] else "ham (kalibrasyon dosyası yok)")

hero, hub = st.columns([55, 45], gap=None)

with hero:
    with st.container(key="hero"):
        ui.html(f"""
<div class="iris-hero">
  {gauge}
  <div class="h-kicker">Beklenen churn oranı</div>
  <div class="h-value">{ui.count(exp["value"] * 100, 1, pre="%", dur=900)}</div>
  <div class="h-band"><i style="background:{band_color}"></i>{band_text} ·
    aralık {ui.fmt_pct(exp["lo"])}–{ui.fmt_pct(exp["hi"])}</div>
  <div class="h-def"><b>{MODEL_NAMES[key]}</b> modelinin test setindeki (eğitimde görülmemiş)
    {ui.fmt_num(exp["n"])} müşteri için verdiği {cal_note} churn olasılıklarının ortalamasıdır;
    karar eşiğinden bağımsızdır. Renk kuralı: tarihsel oranı {ui.fmt_pct(base_rate)} olan
    {ui.fmt_num(exp["n"])} kişilik bir grupta gözlenecek oranın %95 aralığı amber, altı turkuaz,
    üstü mercan. Test setinde gerçekleşen oran: {ui.fmt_pct(exp["actual"])}.</div>
  <div class="iris-alert">
    <div><div class="a-lab">Uyarı listesi</div>
      <div class="a-val">{ui.count(al["n_alert"])} <span style="font-size:.95rem;font-weight:650">müşteri</span></div></div>
    <div class="a-txt">Eşik <b>{ui.fmt_num(thr, 2)}</b> (skor ≥ eşik) · test setindeki payı
      {ui.fmt_pct(al["share"])}. İletişime geçilecek müşteriler; eşik recall ağırlıklı (F2)
      seçildiği için liste beklenen orandan geniştir.</div>
  </div>
</div>""", js=True)

with hub:
    with st.container(key="hub"):
        with st.container(key="tema-mini"):  # sağ üstte küçük tema seçimi (Sistem/Açık/Koyu)
            ui.theme_switch(compact=True)
        ui.html(f"""
<div class="iris-hub-logo">{ui.logo_img("iris-logo-sidebar-dark.svg")}</div>
<div class="iris-hub-title">Müşteri Terki<br>Erken Uyarı Sistemi</div>
<div class="iris-hub-sub">Banka müşterilerinde terk riskini önceden görmek,
  açıklamak ve maliyete göre karar vermek için.</div>
<div class="iris-stats">
  <div class="iris-stat"><div class="s-val">{ui.count(len(df))}</div>
    <div class="s-lab">Müşteri</div></div>
  <div class="iris-stat"><div class="s-val">{ui.count(base_rate * 100, 2, pre="%")}</div>
    <div class="s-lab"><span class="s-dot" style="background:{ui.AMBER}"></span>Tarihsel churn oranı</div></div>
  <div class="iris-stat"><div class="s-val">{ui.count(al["n_alert"])}</div>
    <div class="s-lab"><span class="s-dot" style="background:{ui.CORAL}"></span>Uyarı listesi (eşik {ui.fmt_num(thr, 2)})</div></div>
</div>""", js=True)
        with st.container(key="hub-sunum"):  # vurgulu: sunum akışına tek tıkla giriş
            st.page_link("sayfalar/sunum_modu.py", label="Sunum Modu",
                         icon=":material/co_present:", width="stretch")
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
                f'Modeller: {", ".join(MODEL_NAMES.values())} · Model ve eşik iç sayfalarda '
                f'kenar çubuğundan değişir</div>')
