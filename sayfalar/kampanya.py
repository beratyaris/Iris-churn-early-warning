"""Kampanya planlayıcı: kalibre olasılıkları ("kaç kişi gider?") bir bütçe kararına bağlar.

Veri: test seti (eğitimde görülmemiş 2.000 müşteri, artifacts/test_predictions.csv) ve kenar
çubuğunda seçili model. Müşteriler model skoruna göre sıralanır; bütçenin yettiği ilk N kişiye
ulaşılır. Beklenen terk eden = bu N kişinin KALİBRE olasılıklarının toplamı (ham skor SMOTE-NC
nedeniyle şişiktir); beklenen kurtarılan = beklenen terk eden × başarı oranı; net fayda =
kurtarılan × müşteri değeri − N × temas maliyeti (iris_karar.campaign_curve).
Varsayılan maliyetler Maliyet ve Eşik sayfasıyla aynı kaynaktan (iris_karar.DEFAULT_COST_*).
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import iris_ui as ui
from iris_data import calibrate, current_settings, is_calibrated, load_artifacts
from iris_karar import (DEFAULT_COST_FN, DEFAULT_COST_FP, DEFAULT_SUCCESS, campaign_curve,
                        reach_for_budget)
from train import MODEL_NAMES

art = load_artifacts()
key, thr = current_settings(art)
preds = art["preds"]
scores = preds[f"proba_{key}"].to_numpy()
probs = calibrate(art, key, scores)
calibrated = is_calibrated(art, key)
n_test = len(scores)
n_alert = int((scores >= thr).sum())
ss = st.session_state

ui.page_header("Karar Destek", "Kampanya Planlayıcı",
               "Bütçe kaç müşteriye yeter, bunların kaçının gitmesi beklenir, kampanya net ne kazandırır? "
               "Kalibre olasılıklarla, test setinde.",
               chip=f"{MODEL_NAMES[key]} · test seti {ui.fmt_num(n_test)} müşteri")

# ---------- Girdiler (tamamı değiştirilebilir; maliyet ve başarı oranı VARSAYIM) ----------
ss.setdefault("camp_contact", DEFAULT_COST_FP)
ss.setdefault("camp_value", DEFAULT_COST_FN)
ss.setdefault("camp_success", int(round(DEFAULT_SUCCESS * 100)))
ss.setdefault("camp_mode", "Bütçe")
ss.setdefault("camp_budget", n_alert * DEFAULT_COST_FP)  # başlangıç: uyarı listesine yeten bütçe
ss.setdefault("camp_n", n_alert)

with st.container(key="filterbar"):
    # c0 geniş: "Müşteri sayısı" seçeneği 1440 px'de kırpılıyordu
    c0, c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.15, 1], vertical_alignment="bottom")
    mode = c0.segmented_control("Plan", ["Bütçe", "Müşteri sayısı"], required=True, key="camp_mode",
                                help="Bütçe girin ya da doğrudan kaç müşteriye ulaşılacağını seçin.")
    contact = c2.number_input("Temas maliyeti (kişi başı) · varsayım", 1, 1000, step=1, key="camp_contact",
                              help=f"Varsayılan {DEFAULT_COST_FP} birim = Maliyet ve Eşik sayfasındaki FP "
                                   "(yanlış alarm = bir kampanya teması) maliyeti. Varsayımsal göreli birim.")
    value = c3.number_input("Kaybedilen müşterinin maliyeti · varsayım", 1, 1000, step=1, key="camp_value",
                            help=f"Varsayılan {DEFAULT_COST_FN} birim = Maliyet ve Eşik sayfasındaki FN "
                                 "(kaçırılan churn) maliyeti. Varsayımsal göreli birim.")
    success = c4.slider("Kampanya başarı oranı · varsayım", 0, 100, step=5, key="camp_success",
                        format="%%%d",
                        help="Ulaşılan ve gerçekten ayrılacak olan bir müşteriyi tutma olasılığı. Projede bu "
                             "oran için veri ya da kaynak YOK; %50 yalnız başlangıç değeridir.") / 100
    if mode == "Bütçe":
        budget = c1.number_input("Kampanya bütçesi (birim)", 0, 1_000_000, step=50, key="camp_budget",
                                 help=f"Varsayılan: uyarı listesindeki {ui.fmt_num(n_alert)} kişiye yeten bütçe.")
        N = reach_for_budget(budget, contact, n_test)
    else:
        N = int(c1.number_input("Ulaşılacak müşteri sayısı", 0, n_test, step=10, key="camp_n"))
        budget = N * contact
    ui.html('<div class="iris-card-sub">Temas maliyeti, müşteri değeri ve başarı oranı <b>varsayımdır</b> '
            '(gerçek banka verisi değil; birimler göreli). Maliyetlerin varsayılanları Maliyet ve Eşik '
            f'sayfasındaki FN = {DEFAULT_COST_FN}, FP = {DEFAULT_COST_FP} ile aynıdır.</div>')

curve = campaign_curve(scores, probs, contact, success, value)
order = np.argsort(-scores, kind="stable")
actual_cum = np.concatenate([[0], np.cumsum(preds["churn"].to_numpy()[order])])
raw_cum = np.concatenate([[0.0], np.cumsum(scores[order])])
row = curve.iloc[N]
best = int(curve["net"].idxmax())
brow = curve.iloc[best]
arow = curve.iloc[n_alert]


def signed(v, dec=0):
    """Sayan sayı, işaretli (− / +)."""
    return ui.count(abs(v), dec, pre="−" if v < 0 else ("+" if v > 0 else ""))


# ---------- Sayı kartları ----------
last = (f" · son ulaşılan model skoru {ui.fmt_num(row['son_skor'], 2)}" if N > 0 else "")
cap = " (bütçe tüm test setine yetiyor)" if mode == "Bütçe" and N == n_test and budget > n_test * contact else ""
ui.kpi_row([
    ui.kpi_card("Ulaşılan müşteri", ui.count(N) + f"<small> / {ui.fmt_num(n_test)}</small>",
                f"Bütçe <b>{ui.fmt_num(budget)}</b> birim{last}{cap}",
                tip="Model skoru en yüksek müşteriden başlayarak bütçenin yettiği kişi sayısı.",
                tone=ui.tok("ink")),
    ui.kpi_card("Beklenen terk eden", ui.count(row["beklenen_terk"], 1),
                f"Kalibre olasılıklar toplamı · test setinde gerçekleşen: <b>{ui.fmt_num(actual_cum[N])}</b>",
                tip="Ulaşılan N kişinin kalibre gitme olasılıklarının toplamı (beklenen değer).",
                tone=ui.AMBER),
    ui.kpi_card("Beklenen kurtarılan", ui.count(row["kurtarilan"], 1),
                f"Beklenen terk eden × başarı oranı (%{ui.fmt_num(success * 100)}, varsayım)",
                tip="Kampanyanın tuttuğu beklenen müşteri sayısı; başarı oranı varsayımdır.", tone=ui.TURQ),
    ui.kpi_card("Kampanya maliyeti", ui.count(row["maliyet"]),
                f"{ui.fmt_num(N)} kişi × {ui.fmt_num(contact)} birim (varsayım)",
                tip="Ulaşılan kişi sayısı × kişi başı temas maliyeti.", tone=ui.tok("subtle")),
    ui.kpi_card("Net fayda", signed(row["net"], 1),
                f"Kurtarılan × {ui.fmt_num(value)} − maliyet (birim)",
                tip="Beklenen kurtarılan × kaybedilen müşterinin maliyeti − kampanya maliyeti.",
                tone=ui.TURQ if row["net"] >= 0 else ui.CORAL),
], cols=5)

# ---------- Net fayda eğrisi ----------
with st.container(key="card-camp-curve"):
    ui.card_title("Ulaşılan müşteri sayısına göre net fayda",
                  "Müşteriler model skoruna göre sıralı; her nokta ilk N kişiye ulaşan bir plan. "
                  "Halka: en yüksek net fayda · elmas: seçili plan · noktalı çizgi: uyarı listesi.")
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=ui.tok("neutral"), width=1))
    fig.add_trace(go.Scatter(
        x=curve["N"], y=curve["net"], mode="lines", name="Net fayda",
        line=dict(color=ui.tok("ink"), width=2.4),
        customdata=np.c_[curve["N"] * contact, curve["beklenen_terk"], curve["kurtarilan"]],
        hovertemplate="N = %{x}<br>Bütçe %{customdata[0]:,.0f}<br>Beklenen terk eden %{customdata[1]:,.1f}"
                      "<br>Beklenen kurtarılan %{customdata[2]:,.1f}<br>Net fayda %{y:,.1f}<extra></extra>"))
    fig.add_vline(x=n_alert, line=dict(color=ui.AMBER, width=1.4, dash="dot"))
    fig.add_annotation(x=n_alert, y=1, xref="x", yref="paper", yanchor="top", xanchor="left", xshift=5,
                       text=f"Uyarı listesi ({ui.fmt_num(n_alert)})", showarrow=False,
                       bgcolor=ui.tok("surface"), font=dict(color=ui.tok("muted"), size=11))
    fig.add_trace(go.Scatter(x=[best], y=[brow["net"]], mode="markers", name="En yüksek net fayda",
                             marker=dict(color=ui.TURQ, size=22, symbol="circle-open", line=dict(width=3)),
                             hovertemplate="En yüksek net fayda<br>N = %{x}<br>Net %{y:,.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[N], y=[row["net"]], mode="markers", name="Seçili plan",
                             marker=dict(color=ui.CORAL, size=10, symbol="diamond",
                                         line=dict(color=ui.tok("surface"), width=1.5)),
                             hovertemplate="Seçili plan<br>N = %{x}<br>Net %{y:,.1f}<extra></extra>"))
    fig.add_annotation(x=best, y=brow["net"], text=f"En iyi: N = {ui.fmt_num(best)}", showarrow=False,
                       yanchor="bottom", yshift=14, font=dict(color=ui.tok("accent_text"), size=12))
    fig.update_xaxes(title="Ulaşılan müşteri sayısı (N, skora göre sıralı)", range=[0, n_test],
                     showgrid=True, gridcolor=ui.tok("grid"), tickformat=",d")
    fig.update_yaxes(title="Net fayda (birim)", tickformat=",.0f")
    fig.update_layout(showlegend=False, hovermode="closest", margin=dict(l=4, r=12, t=28, b=4))
    ui.plot(fig, height=360, key="camp-curve")
    if success > 0 and brow["net"] > 0:
        p_last = float(probs[order][best - 1])
        st.caption(f"En yüksek net fayda N = {ui.fmt_num(best)} müşteride ({ui.fmt_num(best * contact)} birim "
                   f"bütçe): {ui.fmt_signed(brow['net'], 1)} birim. Bir müşteriyi listeye eklemek, kalibre "
                   "olasılığı × başarı oranı × müşteri değeri temas maliyetini aştığı sürece net faydayı "
                   f"artırır; bu sınır kalibre olasılık ≈ {ui.fmt_num(contact / (success * value), 3)} "
                   f"(en iyi noktadaki son müşteri: {ui.fmt_num(p_last, 3)}).")
    else:
        st.caption("Bu varsayımlarla hiçbir plan pozitif net fayda üretmiyor (en iyi seçenek kimseye "
                   "ulaşmamak). Başarı oranını ya da müşteri değerini kontrol edin.")

# ---------- Uyarı listesiyle karşılaştırma + yöntem ----------
c1, c2 = st.columns(2, gap="medium")
with c1:
    share = (f"Seçili plan <b>{ui.fmt_num(N)}</b> kişiye ulaşıyor; uyarı listesi büyüklüğüne oranı "
             f"<b>{ui.fmt_pct(N / n_alert)}</b>."
             if 0 < N <= n_alert and n_alert else
             f"Seçili plan uyarı listesinin ötesinde <b>{ui.fmt_num(N - n_alert)}</b> kişiye daha ulaşıyor."
             if N > n_alert else "Seçili plan kimseye ulaşmıyor.")
    ui.html(ui.callout(
        f"Uyarı listesiyle karşılaştırma (eşik {ui.fmt_num(thr, 2)}, model skoru)",
        [f"Uyarı listesindeki <b>{ui.fmt_num(n_alert)}</b> kişinin tamamına ulaşmak için gereken bütçe: "
         f"<b>{ui.fmt_num(n_alert * contact)}</b> birim ({ui.fmt_num(n_alert)} × {ui.fmt_num(contact)}).",
         f"Bu listede beklenen terk eden <b>{ui.fmt_num(arow['beklenen_terk'], 1)}</b> (test setinde "
         f"gerçekleşen {ui.fmt_num(actual_cum[n_alert])}), beklenen kurtarılan "
         f"{ui.fmt_num(arow['kurtarilan'], 1)}, net fayda <b>{ui.fmt_signed(arow['net'], 1)}</b> birim.",
         share],
        tone=ui.AMBER,
        note="Eşik recall ağırlıklı (F2) seçildiği için uyarı listesi, net faydanın en yüksek olduğu N'den "
             "farklı olabilir; ikisi farklı soruları yanıtlar."))
with c2:
    cal_line = (f"<b>Neden kalibre olasılık?</b> Ham skorlar SMOTE-NC nedeniyle yukarı kaymıştır. Uyarı "
                f"listesinde ham skorların toplamı {ui.fmt_num(raw_cum[n_alert], 1)} olurdu; kalibre toplam "
                f"{ui.fmt_num(arow['beklenen_terk'], 1)}, test setinde gerçekte terk eden "
                f"{ui.fmt_num(actual_cum[n_alert])}. Kalibratör monoton olduğu için sıralama ve uyarı "
                "listesi değişmez." if calibrated else
                "<b>Bu model için kalibrasyon uygulanmadı</b>; beklenen terk eden ham skorların toplamıdır.")
    ui.html(ui.callout(
        "Yöntem ve veri",
        [f"<b>Veri:</b> test seti, eğitimde görülmemiş {ui.fmt_num(n_test)} müşteri "
         f"(artifacts/test_predictions.csv), {MODEL_NAMES[key]} skorları. Sonuçlar bu "
         f"{ui.fmt_num(n_test)} kişilik ölçektedir; tüm portföye ölçeklemek ayrı bir varsayım gerektirir.",
         "<b>Hesap:</b> skora göre ilk N kişi; beklenen terk eden = kalibre olasılıklar toplamı; "
         "kurtarılan = terk eden × başarı oranı; net fayda = kurtarılan × müşteri değeri − N × temas maliyeti.",
         cal_line],
        tone=ui.TURQ,
        note="Beklenen değer hesabıdır. Başarı oranı ve maliyetler varsayımdır; kampanyanın gerçek etkisi "
             "bu veriden ölçülemez."))
