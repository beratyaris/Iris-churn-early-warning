"""Müşteri Risk Skoru: tek müşterinin churn olasılığı, uyarı listesi durumu ve SHAP açıklaması.

- Uyarı listesi kararı: modelin ham skoru ≥ kenar çubuğundaki eşik (eşik bu ölçekte seçildi).
- Göstergede tek sayı: kalibre "gitme olasılığı" (calibrate.py); yayda eşik işareti yoktur.
  Uyarı durumu ayrı satırda kendi ölçeğinde (ham skor ≥ eşik) yazılır. Kalibratör monoton
  olduğu için iki ölçek aynı müşterileri seçer; uyarı listesi değişmez.
- SHAP: modelin kendi çıktı ölçeğinde (XGBoost/LR: log-odds, RF: olasılık).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import iris_ui as ui
from iris_data import (FEATURE_TR, SHAP_UNIT, calibrate, current_settings, explain_row,
                       fmt_feature, is_calibrated, load_artifacts)
from train import MODEL_NAMES, prepare_features

COUNTRY_TR = {"France": "Fransa", "Germany": "Almanya", "Spain": "İspanya"}
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


def manual_input_form():
    """Elle müşteri bilgisi girişi; ham veri formatında tek satırlık tablo döner."""
    c1, c2 = st.columns(2)
    row = {
        "credit_score": c1.number_input("Kredi skoru", 300, 900, 650, key="m_cs"),
        "country": c1.selectbox("Ülke", list(COUNTRY_TR), format_func=COUNTRY_TR.get, key="m_c"),
        "gender": c1.selectbox("Cinsiyet", ["Female", "Male"],
                               format_func=lambda g: "Kadın" if g == "Female" else "Erkek", key="m_g"),
        "age": c1.number_input("Yaş", 18, 100, 40, key="m_a"),
        "tenure": c1.number_input("Müşteri süresi (yıl)", 0, 10, 5, key="m_t"),
        "balance": c2.number_input("Bakiye", 0, 300000, 0, step=1000, key="m_b"),
        "products_number": c2.number_input("Ürün sayısı", 1, 4, 1, key="m_p"),
        "estimated_salary": c2.number_input("Tahmini maaş", 1, 250000, 100000, step=1000, key="m_s"),
        "credit_card": int(c2.checkbox("Kredi kartı var", value=True, key="m_cc")),
        "active_member": int(c2.checkbox("Aktif üye", value=True, key="m_am")),
    }
    return prepare_features(pd.DataFrame([row]))


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
        x_raw = manual_input_form()[features]
        country = st.session_state.get("m_c", "France")

score = float(art["models"][key].predict_proba(x_raw)[0, 1])
prob = float(calibrate(art, key, [score])[0])
in_list = score >= thr
# 0,209 gibi skorlar iki basamakta eşikle aynı görünmesin
score_dec = 3 if round(score, 2) == round(thr, 2) else 2

calibrated = is_calibrated(art, key)
# Tek ölçek: göstergede yalnız kalibre olasılık (eşik işareti yok). Uyarı kararı ayrı satırda,
# kendi ölçeğinde (ham skor ≥ eşik) yazılır; ikisi aynı yayda karıştırılmaz.
ALERT_TIP = ("Gitme olasılığı kalibre edilmiştir: gerçek churn oranıyla uyumlu, yüzde olarak "
             "okunabilir. Uyarı kararı ise modelin ham skoru ile maliyet ve F2'ye göre seçilen "
             f"eşik ({ui.fmt_num(thr, 2)}) karşılaştırılarak verilir. Kalibrasyon müşterilerin "
             "sıralamasını değiştirmediği için uyarı listesi iki ölçekte de aynıdır."
             if calibrated else
             "Bu model için kalibrasyon uygulanmadı; gösterilen değer modelin ham skorudur. Uyarı "
             f"kararı ham skor ≥ eşik ({ui.fmt_num(thr, 2)}) kuralıyla verilir.")
ui.html("""<style>
.iris-alertline { display:flex; align-items:center; justify-content:center; gap:.45rem; flex-wrap:wrap;
  margin:.65rem auto 0; padding:.5rem .8rem; border:1px solid var(--iris-border); border-radius:12px;
  background: var(--iris-soft); color: var(--iris-ink); font-size:.86rem; max-width: 360px; }
.iris-alertline .al-score { color: var(--iris-muted); font-size:.8rem; font-variant-numeric: tabular-nums; }
.iris-info { display:inline-flex; width:16px; height:16px; border-radius:50%; flex:0 0 auto;
  border:1px solid var(--iris-subtle); color: var(--iris-subtle); font-size:.64rem; font-weight:700;
  align-items:center; justify-content:center; cursor:help; font-style:normal; }
</style>""")

with mid, st.container(key="card-gauge"):
    ui.card_title("Gitme olasılığı",
                  "Kalibre olasılık: bu profildeki 100 müşteriden kaçının gitmesi beklenir"
                  if calibrated else "Model skoru (kalibrasyon yok)")
    g = ui.gauge_svg(prob, 1.0, bands=ui.risk_bands(), separators=False,
                     labels=[(0, "%0"), (0.5, "%50"), (1, "%100")],
                     on_dark=ui.mode() == "dark", cls="iris-gauge-sm",
                     alt=f"Gitme olasılığı {ui.fmt_pct(prob)}")
    truth = ("" if actual is None else
             f'<div class="iris-status"><span class="iris-pill flat">Gerçek: '
             f'{"terk etti" if actual else "kaldı"}</span></div>')
    alert_pill = ('<span class="iris-pill warn">Evet</span>' if in_list
                  else '<span class="iris-pill ok">Hayır</span>')
    ui.html(f'{g}<div style="text-align:center">'
            f'<div class="iris-big">{ui.count(prob * 100, 1, pre="%")}</div>'
            f'<div class="iris-card-sub">gitme olasılığı</div>'
            f'<div class="iris-alertline" title="{ALERT_TIP}"><b>Uyarı listesinde:</b> {alert_pill}'
            f'<span class="al-score">(model skoru {ui.fmt_num(score, score_dec)} {"≥" if in_list else "<"} '
            f'eşik {ui.fmt_num(thr, 2)})</span><i class="iris-info" title="{ALERT_TIP}">i</i></div>'
            f'{truth}</div>', js=True)

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
expl = explain_row(key, art, x_raw)
vals = np.asarray(expl.values, dtype=float)
order = np.argsort(-np.abs(vals))
k = 9
shown, rest = order[:k], order[k:]
labels = [f"{FEATURE_TR[features[i]]} = {fmt_feature(features[i], x_raw.iloc[0, i])}" for i in shown]
contrib = [float(v) for v in vals[shown]]
if len(rest):
    labels.append(f"Diğer {len(rest)} değişken")
    contrib.append(float(vals[rest].sum()))
unit = SHAP_UNIT[key]
dec = 3 if unit == "olasılık" else 2
base = float(expl.base_values)
fx = base + float(vals.sum())

with st.container(key="card-shap"):
    ui.card_title("Bu skoru ne belirledi? (SHAP)",
                  "Başlangıç: modelin ortalama çıktısı; her çubuk bir değişkenin bu müşteride skoru "
                  "ne kadar <b>artırdığını</b> (mercan) ya da <b>azalttığını</b> (turkuaz) gösterir. "
                  f"Birim: {unit}.")
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(contrib) + ["total"],
        y=["Ortalama (başlangıç)"] + labels + ["Bu müşteri"],
        x=[base] + contrib + [fx],
        text=[ui.fmt_num(base, dec)] + [ui.fmt_signed(c, dec) for c in contrib] + [ui.fmt_num(fx, dec)],
        textposition="outside", cliponaxis=False,
        increasing=dict(marker=dict(color=ui.CORAL)), decreasing=dict(marker=dict(color=ui.TURQ)),
        totals=dict(marker=dict(color=ui.tok("neutral"))),
        connector=dict(line=dict(color=ui.tok("border"), width=1)),
        hovertemplate="%{y}<br>Değer: %{x:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=False)
    fig.update_xaxes(showgrid=True, gridcolor=ui.tok("grid"), title=f"Model çıktısı ({unit})")
    # Uzun Türkçe etiketler kırpılmasın diye sol boşluk sabit
    fig.update_layout(showlegend=False, margin=dict(l=190, r=48, t=8, b=44))
    ui.plot(fig, height=430, key="shap-wf")
    note = ("log-odds ölçeğinde; skor = 1 / (1 + e^(−çıktı))" if unit == "log-odds"
            else "olasılık ölçeğinde (ağaçların ortalaması)")
    st.caption(f"Bu müşteride model çıktısı {ui.fmt_num(fx, dec)} ({note}) → skor "
               f"{ui.fmt_num(score, 3)}. SHAP, kalibrasyon öncesi model çıktısını açıklar; kalibrasyon "
               "monoton olduğu için sıralama, yön ve önem yorumları değişmez.")
