"""Müşteri Risk Skoru: tek müşterinin churn olasılığı, uyarı listesi durumu ve SHAP açıklaması.

- Uyarı listesi kararı: modelin ham skoru ≥ kenar çubuğundaki eşik (eşik bu ölçekte seçildi).
- Göstergede tek sayı: kalibre "gitme olasılığı" (calibrate.py); yayda eşik işareti yoktur.
  Uyarı durumu ayrı satırda kendi ölçeğinde (ham skor ≥ eşik) yazılır. Kalibratör monoton
  olduğu için iki ölçek aynı müşterileri seçer; uyarı listesi değişmez.
- SHAP: modelin kendi çıktı ölçeğinde (XGBoost/LR: log-odds, RF: olasılık).
Elle giriş formu, gösterge bloğu ve SHAP şelalesi iris_musteri.py'de (simülatörle ortak);
elle giriş sınırları eğitim verisindeki gerçek aralıktır.
"""
import streamlit as st

import iris_musteri as cm
import iris_ui as ui
from iris_data import SHAP_UNIT, current_settings, is_calibrated, load_artifacts
from iris_karar import COUNTRY_TR
from train import MODEL_NAMES

art = load_artifacts()
key, thr = current_settings(art)
preds, features = art["preds"], art["features"]
score_col = f"proba_{key}"

ui.page_header("Karar Destek", "Müşteri Risk Skoru",
               "Tek müşterinin churn olasılığı, uyarı listesi durumu ve kararın gerekçesi (SHAP)",
               chip=f"{MODEL_NAMES[key]} · eşik {ui.fmt_num(thr, 2)}")

# Uyarı listesinde en yüksek skorlu müşteriler (hızlı seçim)
alerts = preds.loc[preds[score_col] >= thr].sort_values(score_col, ascending=False)
top = alerts.head(6)
ids = preds.sort_values(score_col, ascending=False)["customer_id"].tolist()
st.session_state.setdefault("risk_cid", int(top["customer_id"].iloc[0]) if len(top) else ids[0])
st.session_state.setdefault("risk_mode", "Test setinden seç")


def _pick():
    v = st.session_state.get("risk_pick")
    if v is not None:
        st.session_state["risk_cid"] = v
    st.session_state["risk_pick"] = None


def score_label(c):
    s = float(preds.loc[preds["customer_id"] == c, score_col].iloc[0])
    return f"{c} · {ui.fmt_num(s, 2)}"


left, mid, right = st.columns([1.1, 1, 1], gap="medium")

with left, st.container(key="card-select"):
    ui.card_title("Müşteri seçimi", "Test setindeki (eğitimde görülmemiş) 2.000 müşteriden biri "
                                    "ya da elle girilen bir profil")
    mode = st.segmented_control("Kaynak", ["Test setinden seç", "Elle değer gir"],
                                required=True, key="risk_mode", label_visibility="collapsed")
    actual = None
    if mode == "Test setinden seç":
        st.selectbox("Müşteri no (yazarak arayın)", ids, key="risk_cid",
                     help="Liste model skoruna göre yüksekten düşüğe sıralıdır.")
        if len(top):
            st.pills("Uyarı listesinde en yüksek skorlu müşteriler (no · skor)",
                     top["customer_id"].tolist(), format_func=score_label,
                     key="risk_pick", on_change=_pick)
        row = preds.loc[preds["customer_id"] == st.session_state["risk_cid"]]
        x_raw = row[features]
        actual = int(row["churn"].iloc[0])
        country = ("Germany" if row["country_Germany"].iloc[0] else
                   "Spain" if row["country_Spain"].iloc[0] else "France")
        rank = ids.index(st.session_state["risk_cid"]) + 1
        ui.html(f'<div class="iris-card-sub">Skor sıralaması: <b>{ui.fmt_num(rank)}</b> / '
                f'{ui.fmt_num(len(ids))} · uyarı listesinde <b>{ui.fmt_num(len(alerts))}</b> müşteri</div>')
    else:
        profile = cm.customer_form("m")  # sınırlar eğitim verisindeki gerçek aralık
        x_raw = cm.features(art, profile)
        country = profile["country"]

score, prob = cm.score(art, key, x_raw)
calibrated = is_calibrated(art, key)
# Tek ölçek: göstergede yalnız kalibre olasılık (eşik işareti yok). Uyarı kararı ayrı satırda,
# kendi ölçeğinde (ham skor ≥ eşik) yazılır; ikisi aynı yayda karıştırılmaz.
ui.html(cm.ALERT_CSS)

with mid, st.container(key="card-gauge"):
    ui.card_title("Gitme olasılığı",
                  "Kalibre olasılık: bu profildeki 100 müşteriden kaçının gitmesi beklenir"
                  if calibrated else "Model skoru (kalibrasyon yok)")
    truth = ("" if actual is None else
             f'<div class="iris-status"><span class="iris-pill flat">Gerçek: '
             f'{"terk etti" if actual else "kaldı"}</span></div>')
    ui.html(cm.gauge_html(prob, score, thr, calibrated, extra=truth), js=True)

with right, st.container(key="card-facts"):
    ui.card_title("Müşteri bilgileri", "Modelin gördüğü ham değerler")
    x = x_raw.iloc[0]
    pairs = [("Ülke", COUNTRY_TR[country]), ("Cinsiyet", "Erkek" if x["gender_Male"] else "Kadın"),
             ("Yaş", ui.fmt_num(x["age"])), ("Müşteri süresi", f"{ui.fmt_num(x['tenure'])} yıl"),
             ("Kredi skoru", ui.fmt_num(x["credit_score"])), ("Ürün sayısı", ui.fmt_num(x["products_number"])),
             ("Bakiye", ui.fmt_num(x["balance"])), ("Tahmini maaş", ui.fmt_num(x["estimated_salary"])),
             ("Aktif üye", "Evet" if x["active_member"] else "Hayır"),
             ("Kredi kartı", "Evet" if x["credit_card"] else "Hayır")]
    ui.html(ui.facts(pairs))

# ---------- SHAP ----------
with st.container(key="card-shap"):
    ui.card_title("Bu skoru ne belirledi? (SHAP)",
                  "Başlangıç: modelin ortalama çıktısı; her çubuk bir değişkenin bu müşteride skoru "
                  "ne kadar <b>artırdığını</b> (mercan) ya da <b>azalttığını</b> (turkuaz) gösterir. "
                  f"Birim: {SHAP_UNIT[key]}.")
    w = cm.shap_waterfall(art, key, x_raw, chart_key="shap-wf")
    note = ("log-odds ölçeğinde; skor = 1 / (1 + e^(−çıktı))" if w["unit"] == "log-odds"
            else "olasılık ölçeğinde (ağaçların ortalaması)")
    st.caption(f"Bu müşteride model çıktısı {ui.fmt_num(w['fx'], w['dec'])} ({note}) → skor "
               f"{ui.fmt_num(score, 3)}. SHAP, kalibrasyon öncesi model çıktısını açıklar; kalibrasyon "
               "monoton olduğu için sıralama, yön ve önem yorumları değişmez.")
