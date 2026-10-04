"""Maliyet ve Eşik: maliyete duyarlı karar eşiği (validation'da seçilir, test'te raporlanır).

Toplam maliyet = FN maliyeti × kaçırılan churn + FP maliyeti × yanlış alarm.
Eşik ekseni modelin ham skorudur (paylaşılan eşikle aynı ölçek).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis
import iris_ui as ui
from iris_data import calibrate, current_settings, is_calibrated, load_artifacts, set_threshold
from train import MODEL_NAMES

# Varsayılanlar - VARSAYIMSAL birimler, gerçek banka verisi değildir.
# Gerekçe: kaçırılan bir churn müşterisi (FN) gelecekteki gelirin tamamen kaybıdır;
# yanlış alarm (FP) yalnızca bir elde tutma kampanyasının maliyetidir. FN'nin FP'den
# birkaç kat pahalı olduğu varsayımıyla 5:1 seçildi; kullanıcı arayüzden değiştirebilir.
DEFAULT_COST_FN = 5
DEFAULT_COST_FP = 1

art = load_artifacts()
key, thr = current_settings(art)
val, test = art["val_preds"], art["preds"]

ui.page_header("Karar Destek", "Maliyet ve Eşik",
               "Hangi eşik toplam maliyeti en aza indirir? Eşik validation setinde seçilir, "
               "sonuçlar test setinde raporlanır.",
               chip=f"{MODEL_NAMES[key]} · eşik {ui.fmt_num(thr, 2)}")

with st.container(key="filterbar"):
    c1, c2, c3 = st.columns([1, 1, 2.2], vertical_alignment="bottom")
    cost_fn = c1.number_input("FN maliyeti (kaçırılan churn)", 1, 1000, DEFAULT_COST_FN, 1,
                              key="cost_fn", help="Varsayımsal göreli birim; yalnız FN/FP oranı önemlidir.")
    cost_fp = c2.number_input("FP maliyeti (yanlış alarm)", 1, 1000, DEFAULT_COST_FP, 1,
                              key="cost_fp", help="Varsayımsal göreli birim (ör. bir kampanya maliyeti).")
    c3.html(f'<div class="iris-card-sub">Varsayılan FN = {DEFAULT_COST_FN}, FP = {DEFAULT_COST_FP} '
            'varsayımsaldır: kaçırılan müşterinin kampanya maliyetinden birkaç kat pahalı olduğu kabul '
            'edildi. Gerçek banka maliyetleriyle değiştirilmelidir. Oran '
            f'<b>{ui.fmt_num(cost_fn / cost_fp, 1)} : 1</b>.</div>')

y_val, s_val = val["churn"], val[f"proba_{key}"]
y_test, s_test = test["churn"], test[f"proba_{key}"]
opt, curve = analysis.cost_optimal_threshold(y_val, s_val, cost_fn, cost_fp)
curve_test = analysis.cost_curve(y_test, s_test, cost_fn, cost_fp)

# Bayes eşiği kalibre olasılık ölçeğindedir; ham skor karşılığını kalibratörle buluruz.
bayes_p = analysis.bayes_threshold(cost_fn, cost_fp)
bayes_raw = None
if is_calibrated(art, key):
    fine = np.linspace(0.001, 0.999, 999)
    above = fine[calibrate(art, key, fine) >= bayes_p]
    bayes_raw = float(above[0]) if len(above) else None


def at(t, y, s):
    r = analysis.evaluate_at(y, s, t, cost_fn, cost_fp)
    r["Uyarı"] = int((np.asarray(s) >= t).sum())
    return r


cur, best = at(thr, y_test, s_test), at(opt, y_test, s_test)
same = abs(opt - thr) < 1e-9


def sub_opt(v):
    return f"Maliyet-optimal eşikte ({ui.fmt_num(opt, 2)}): <b>{v}</b>"


ui.html('<div class="iris-section"><div><div class="iris-card-title">Seçili eşikte test seti sonuçları</div>'
        f'<div class="iris-card-sub">{MODEL_NAMES[key]} · eşik {ui.fmt_num(thr, 2)} (kenar çubuğundaki '
        'ortak eşik)</div></div></div>')
ui.kpi_row([
    ui.kpi_card("Recall", ui.count(cur["Recall"] * 100, 1, pre="%"), sub_opt(ui.fmt_pct(best["Recall"])),
                tip="Gerçek churn müşterilerinin ne kadarı uyarı listesinde.", tone=ui.TURQ),
    ui.kpi_card("Precision", ui.count(cur["Precision"] * 100, 1, pre="%"),
                sub_opt(ui.fmt_pct(best["Precision"])),
                tip="Uyarı listesindekilerin ne kadarı gerçekten terk etti.", tone=ui.tok("ink")),
    ui.kpi_card("Uyarı listesi", ui.count(cur["Uyarı"]) + f"<small> / {ui.fmt_num(len(test))}</small>",
                sub_opt(ui.fmt_num(best["Uyarı"])), tip="Skoru eşiğe eşit/üstünde test müşterisi.",
                tone=ui.AMBER),
    ui.kpi_card("Toplam maliyet (test)", ui.count(cur["Toplam maliyet"]),
                sub_opt(ui.fmt_num(best["Toplam maliyet"])) +
                f" · FN {ui.fmt_num(cur['FN'])}, FP {ui.fmt_num(cur['FP'])}",
                tip="FN maliyeti × kaçırılan churn + FP maliyeti × yanlış alarm (seçili eşikte).",
                tone=ui.CORAL),
])

with st.container(key="card-cost"):
    h1, h2 = st.columns([3, 1.1], vertical_alignment="center")
    with h1:
        ui.card_title("Eşiğe göre toplam maliyet",
                      f"Düz çizgi: validation (eşik burada seçilir) · noktalı: test (yalnız bilgi). "
                      f"Maliyet-optimal eşik <b>{ui.fmt_num(opt, 2)}</b>, seçili eşik <b>{ui.fmt_num(thr, 2)}</b>.")
    h2.button(f"Bu eşiği uygula ({ui.fmt_num(opt, 2)})", key="apply_opt", type="primary",
              on_click=set_threshold, args=(key, opt), disabled=same, width="stretch",
              help="Kenar çubuğundaki ortak karar eşiğini maliyet-optimal değere ayarlar.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["Eşik"], y=curve["Maliyet"], name="Validation", mode="lines",
        line=dict(color=ui.tok("ink"), width=2.4),
        customdata=np.c_[curve["FN"], curve["FP"]],
        hovertemplate="Eşik %{x:.2f}<br>Maliyet %{y:,.0f}<br>FN %{customdata[0]} · FP %{customdata[1]}"
                      "<extra>Validation</extra>"))
    fig.add_trace(go.Scatter(
        x=curve_test["Eşik"], y=curve_test["Maliyet"], name="Test", mode="lines",
        line=dict(color=ui.tok("subtle"), width=1.6, dash="dot"),
        hovertemplate="Eşik %{x:.2f}<br>Maliyet %{y:,.0f}<extra>Test</extra>"))
    vmin = float(curve["Maliyet"].min())
    vcur = float(curve.loc[(curve["Eşik"] - thr).abs().idxmin(), "Maliyet"])
    fig.add_trace(go.Scatter(x=[opt], y=[vmin], mode="markers", name="Maliyet-optimal",
                             marker=dict(color=ui.TURQ, size=13, line=dict(color=ui.tok("surface"), width=2)),
                             hovertemplate="Maliyet-optimal eşik %{x:.2f}<br>Maliyet %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[thr], y=[vcur], mode="markers", name="Seçili eşik",
                             marker=dict(color=ui.CORAL, size=13, symbol="diamond",
                                         line=dict(color=ui.tok("surface"), width=2)),
                             hovertemplate="Seçili eşik %{x:.2f}<br>Maliyet %{y:,.0f}<extra></extra>"))
    fig.add_vline(x=opt, line=dict(color=ui.TURQ, width=1.2, dash="dash"))
    fig.add_vline(x=thr, line=dict(color=ui.CORAL, width=1.2, dash="dash"))
    if bayes_raw is not None:
        fig.add_vline(x=bayes_raw, line=dict(color=ui.AMBER, width=1.2, dash="dot"),
                      annotation_text=f"Bayes ({ui.fmt_num(bayes_raw, 2)})",
                      annotation_font=dict(color=ui.tok("muted"), size=11), annotation_position="bottom right")
    fig.update_xaxes(title="Karar eşiği (model skoru)", range=[0, 1], tickformat=".1f",
                     showgrid=True, gridcolor=ui.tok("grid"))
    fig.update_yaxes(title="Toplam maliyet (birim)", rangemode="tozero")
    fig.update_layout(hovermode="closest")
    ui.plot(fig, height=360, key="cost-curve")
    bayes_txt = (f"Kalibre olasılıklarla teorik Bayes eşiği FP/(FP+FN) = {ui.fmt_num(bayes_p, 3)}; "
                 f"bu, model skorunda ≈ {ui.fmt_num(bayes_raw, 2)} demektir (amber noktalı çizgi). "
                 f"Validation'daki ampirik optimum {ui.fmt_num(opt, 2)}; fark "
                 f"{ui.fmt_num(abs(opt - bayes_raw), 2)}."
                 if bayes_raw is not None else
                 f"Teorik Bayes eşiği FP/(FP+FN) = {ui.fmt_num(bayes_p, 3)} kalibre olasılık varsayar; "
                 "ham skorlar SMOTE-NC nedeniyle yukarı kaydığı için doğrudan karşılaştırılamaz.")
    st.caption(bayes_txt)

# ---------- Tüm modeller, üç eşik ----------
rows = []
for k, name in MODEL_NAMES.items():
    o, _ = analysis.cost_optimal_threshold(val["churn"], val[f"proba_{k}"], cost_fn, cost_fp)
    for label, t in [("Maliyet-optimal", o), ("F2-optimal", art["thresholds"][k]), ("0,50", 0.50)]:
        r = at(t, test["churn"], test[f"proba_{k}"])
        rows.append({"Model": name, "Eşik türü": label, "Eşik": t, "Recall": r["Recall"],
                     "Precision": r["Precision"], "Uyarı": r["Uyarı"], "FN": r["FN"], "FP": r["FP"],
                     "Toplam maliyet": r["Toplam maliyet"]})
table = pd.DataFrame(rows)
with st.container(key="card-table"):
    ui.card_title("Test seti: modeller ve eşik türleri",
                  "Aynı FN/FP maliyetleriyle; en düşük maliyet ve en yüksek recall vurgulu. "
                  "Seçili model satırları hafif renkli.")
    ui.html(ui.table_html(
        table, fmt={"Eşik": lambda v: ui.fmt_num(v, 2), "Recall": lambda v: ui.fmt_num(v, 3),
                    "Precision": lambda v: ui.fmt_num(v, 3), "Uyarı": ui.fmt_num, "FN": ui.fmt_num,
                    "FP": ui.fmt_num, "Toplam maliyet": ui.fmt_num},
        num=("Eşik", "Recall", "Precision", "Uyarı", "FN", "FP", "Toplam maliyet"),
        best={"Toplam maliyet": "min", "Recall": "max"},
        sel_rows=set(table.index[table["Model"] == MODEL_NAMES[key]])))
