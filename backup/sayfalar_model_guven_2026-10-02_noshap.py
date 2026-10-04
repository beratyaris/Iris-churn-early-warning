"""Modele güvenebilir miyiz? XGBoost olasılıklarının gerçek terk oranıyla örtüşmesi.

Üç sürüm: SMOTE-NC'li XGBoost (ham skor, uygulamadaki ana model), SMOTE'suz XGBoost ve kalibre
edilmiş XGBoost (SMOTE-NC'li modelin skoru + validation setinde eğitilmiş Platt kalibratörü).
Burada hiçbir model eğitilmez:
- SMOTE'lu ve kalibre eğriler test tahminlerinden (test_predictions.csv + calibration.joblib)
  calibrate.py'deki AYNI kutulama fonksiyonuyla (reliability, N_BINS eşit genişlikli kutu) çizilir.
- SMOTE'suz sürüm smote_karsilastirma.json'dan okunur; o betik müşteri düzeyindeki olasılıkları
  kaydetmediği için bu sürüm yalnız genel ortalama (ortalama tahmin vs gerçek oran) ve Brier ile
  gösterilir.
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import iris_ui as ui
from calibrate import N_BINS, reliability
from iris_data import calibrate, current_settings, is_calibrated, load_artifacts, threshold_as_proba
from train import OUT

SMOTE_FILE = OUT / "smote_karsilastirma.json"
KEY = "xgb"  # sayfa XGBoost'a özeldir (kenar çubuğundaki model seçiminden bağımsız)


@st.cache_data
def load_smote(mtime):
    with open(SMOTE_FILE, encoding="utf-8") as f:
        return json.load(f)


art = load_artifacts()
current_settings(art)
thr = float(st.session_state["iris_thr"][KEY])  # ortak eşik (varsayılan 0,21)
preds = art["preds"]
y = preds["churn"].to_numpy()
raw = preds[f"proba_{KEY}"].to_numpy()

ui.page_header("Analiz", "Modele güvenebilir miyiz?",
               "XGBoost'un verdiği olasılıklar gerçek terk oranıyla ne kadar örtüşüyor? "
               f"Test seti ({ui.fmt_num(len(y))} müşteri).",
               chip=f"XGBoost · {N_BINS} eşit genişlikli kutu")

if not is_calibrated(art, KEY):
    st.info("Kalibrasyon dosyası yok ya da XGBoost için uygulanmadı. Önce "
            "`.\\.venv\\Scripts\\python.exe calibrate.py` çalıştırın.")
    st.stop()

cal = calibrate(art, KEY, raw)
summ = art["calibration"]["summary"]["models"][KEY]
actual = float(y.mean())
sm = load_smote(SMOTE_FILE.stat().st_mtime) if SMOTE_FILE.exists() else None
nos = sm["results"][KEY]["no_smote"] if sm else None

versions = [  # (ad, ortalama olasılık, Brier, renk)
    ("SMOTE'lu XGBoost (ham skor)", float(raw.mean()), float(summ["brier_test"]), ui.CORAL),
    ("Kalibre XGBoost", float(cal.mean()), float(summ["brier_test_cal"]), ui.TURQ),
]
if nos:
    versions.insert(1, ("SMOTE'suz XGBoost", float(nos["mean_pred"]), float(nos["brier"]), ui.tok("france")))

# Sıralama ve uyarı listesi kontrolü (hesaplanır, elle yazılmaz)
order = np.argsort(-raw, kind="stable")
rank_kept = bool(np.all(np.diff(cal[order]) <= 1e-12))
n_alert_raw = int((raw >= thr).sum())
n_alert_cal = int((cal >= threshold_as_proba(art, KEY, thr)).sum())
same_list = bool(((raw >= thr) == (cal >= threshold_as_proba(art, KEY, thr))).all())

# ---------- Bulgu ----------
gap = (raw.mean() - actual) * 100
nos_txt = (f" SMOTE'suz model zaten gerçek orana yakın: {ui.fmt_pct(nos['mean_pred'], 2)} "
           f"(Brier {ui.fmt_num(nos['brier'], 4)})." if nos else "")
keep_txt = (f"<b>Sıralama ve uyarı listesi değişmez:</b> eşik {ui.fmt_num(thr, 2)} orijinal skora "
            f"uygulanır; iki ölçekte de {ui.fmt_num(n_alert_raw)} müşteri, aynı sırayla."
            if rank_kept and same_list else
            f"<b>Dikkat:</b> kontrol sıralamada ya da uyarı listesinde fark buldu (ham {n_alert_raw}, "
            f"kalibre {n_alert_cal}).")
ui.html(ui.callout("Bulgu", [
    f"<b>SMOTE olasılıkları yukarı şişirir:</b> SMOTE'lu XGBoost'un ortalama olasılığı "
    f"{ui.fmt_pct(raw.mean(), 2)}, gerçek terk oranı {ui.fmt_pct(actual, 2)} "
    f"({ui.fmt_signed(gap, 1)} puan).",
    f"<b>Kalibrasyon olasılıkları gerçek orana hizalar:</b> ortalama {ui.fmt_pct(cal.mean(), 2)}; Brier "
    f"{ui.fmt_num(summ['brier_test'], 4)} → {ui.fmt_num(summ['brier_test_cal'], 4)}." + nos_txt,
    keep_txt], tone=ui.TURQ))

# ---------- Güvenilirlik grafiği + ortalamalar ----------
rel_raw, rel_cal = pd.DataFrame(reliability(y, raw)), pd.DataFrame(reliability(y, cal))
c1, c2 = st.columns([1.35, 1], gap="medium")
with c1, st.container(key="card-guven-rel"):
    ui.card_title(f"Güvenilirlik grafiği (test, {N_BINS} eşit genişlikli kutu)",
                  "Her nokta bir olasılık kutusu: yatayda kutudaki ortalama tahmin, dikeyde o kutudaki "
                  "müşterilerin gerçekte terk etme oranı. Köşegen = tahmin ile gerçek oran aynı. Nokta "
                  "boyutu kutudaki müşteri sayısı.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Tahmin = gerçek oran",
                             line=dict(color=ui.tok("neutral"), width=1, dash="dash"), hoverinfo="skip"))
    for (label, _, _, color), rel in [(versions[0], rel_raw), (versions[-1], rel_cal)]:
        fig.add_trace(go.Scatter(
            x=rel["mean_pred"], y=rel["frac_pos"], mode="lines+markers", name=label,
            line=dict(color=color, width=2.2),
            marker=dict(color=color, size=np.clip(np.sqrt(rel["n"]) * 1.1, 6, 26),
                        line=dict(color=ui.tok("surface"), width=1.5)),
            customdata=rel["n"],
            hovertemplate=f"{label}<br>Ort. tahmin %{{x:.3f}}<br>Gerçek oran %{{y:.3f}}"
                          "<br>n = %{customdata}<extra></extra>"))
    if nos:
        fig.add_trace(go.Scatter(
            x=[nos["mean_pred"]], y=[nos["actual_rate"]], mode="markers",
            name="SMOTE'suz XGBoost (yalnız genel ortalama)",
            marker=dict(color=ui.tok("france"), size=13, symbol="diamond",
                        line=dict(color=ui.tok("surface"), width=1.5)),
            hovertemplate="SMOTE'suz XGBoost (tüm test seti)<br>Ort. tahmin %{x:.3f}<br>"
                          "Gerçek oran %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Tahmin edilen olasılık (kutu ortalaması)", range=[0, 1],
                     showgrid=True, gridcolor=ui.tok("grid"), tickformat=".1f")
    fig.update_yaxes(title="Gerçekleşen terk oranı", range=[0, 1.02], tickformat=".1f")
    fig.update_layout(legend=dict(orientation="v", x=0.02, xanchor="left", y=0.98, yanchor="top",
                                  bgcolor=ui.tok("surface"), bordercolor=ui.tok("border"), borderwidth=1))
    ui.plot(fig, height=380, key="guven-rel")
    st.caption("SMOTE'suz modelin müşteri düzeyindeki olasılıkları kaydedilmediği için (smote_karsilastirma.py "
               "yalnız özet yazar) bu sürüm kutu eğrisi yerine tek noktayla gösterilir: tüm test setinde "
               "ortalama tahmin ve gerçek oran.")

with c2, st.container(key="card-guven-mean"):
    ui.card_title("Ortalama olasılık ve gerçek terk oranı",
                  f"Test setindeki {ui.fmt_num(len(y))} müşteri için ortalama tahmin. Kesikli çizgi: "
                  "gerçekleşen terk oranı.")
    fig = go.Figure(go.Bar(
        y=[v[0] for v in versions], x=[v[1] * 100 for v in versions], orientation="h",
        marker=dict(color=[v[3] for v in versions], cornerradius=5),
        text=[ui.fmt_pct(v[1], 2) for v in versions], textposition="outside", cliponaxis=False,
        textfont=dict(color=ui.tok("ink"), size=13),
        hovertemplate="%{y}<br>Ortalama olasılık %%{x:.2f}<extra></extra>"))
    fig.add_vline(x=actual * 100, line=dict(color=ui.tok("subtle"), width=1.5, dash="dash"))
    fig.add_annotation(x=actual * 100, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"Gerçek {ui.fmt_pct(actual, 2)}", font=dict(color=ui.tok("muted"), size=12))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(ticksuffix="", tickprefix="%", range=[0, max(v[1] for v in versions) * 100 * 1.22],
                     showgrid=True, gridcolor=ui.tok("grid"))
    fig.update_layout(showlegend=False, bargap=0.45, margin=dict(l=4, r=12, t=30, b=4))
    ui.plot(fig, height=300, key="guven-mean")

# ---------- Sürüm kartları: Brier + ortalama vs gerçek ----------
cards = []
for name, mean, brier, color in versions:
    diff = (mean - actual) * 100
    cards.append(ui.kpi_card(
        name, ui.count(brier, 4) + "<small> Brier</small>",
        f"Ortalama olasılık <b>{ui.fmt_pct(mean, 2)}</b> · gerçek {ui.fmt_pct(actual, 2)} "
        f"{ui.delta_badge(diff)}",
        tip="Brier = ortalama (olasılık − gerçek sonuç)²; düşük olan daha iyi. Rozet: ortalama olasılık ile "
            "gerçek terk oranı arasındaki fark (puan).", tone=color))
ui.kpi_row(cards, cols=len(cards))

ui.html(ui.callout("Nasıl hesaplandı?", [
    f"<b>Kutular:</b> {N_BINS} eşit genişlikli olasılık kutusu (0–0,1, 0,1–0,2, …; calibrate.py "
    "<code>reliability</code>). Boş kutu çizilmez.",
    "<b>Kalibrasyon:</b> Platt/isotonic seçimi ve kalibratör eğitimi <b>yalnız validation setinde</b> "
    "yapıldı (calibrate.py); bu sayfadaki tüm sayılar test setindendir. Hiçbir model yeniden eğitilmedi.",
    "<b>SMOTE'suz sürüm:</b> aynı bölme, özellikler ve hiperparametrelerle, yalnız SMOTE-NC adımı "
    "çıkarılarak eğitilmişti (smote_karsilastirma.py); değerler o çıktıdan okunur.",
    f"<b>Uyarı listesi:</b> orijinal skor ≥ {ui.fmt_num(thr, 2)}; kalibrasyon monoton olduğu için "
    f"kalibre ölçekte de aynı {ui.fmt_num(n_alert_cal)} müşteri seçilir."],
    tone=ui.AMBER,
    note="Grafik ve kartlar olasılıkların gerçek orana uyumunu gösterir; modelin müşterileri ayırma "
         "gücü (ROC/PR) için Model Karşılaştırma sayfasına bakın."))
