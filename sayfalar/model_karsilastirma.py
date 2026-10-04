"""Model Karşılaştırma: test metrikleri, ROC/PR eğrileri, McNemar + Holm ve kalibrasyon.

Tüm sayılar artifacts/ içindeki gerçek çıktılardan okunur (metrics.csv, mcnemar.csv,
test_predictions.csv, calibration.json). Hiçbir model burada yeniden eğitilmez.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, precision_recall_curve, roc_curve

import iris_ui as ui
from iris_data import current_settings, load_artifacts
from train import MODEL_NAMES

art = load_artifacts()
key, thr = current_settings(art)
preds, y = art["preds"], art["preds"]["churn"]
MODEL_COLORS = {"lr": ui.tok("subtle"), "rf": ui.tok("ink"), "xgb": ui.TURQ}
NAME_TO_KEY = {v: k for k, v in MODEL_NAMES.items()}

ui.page_header("Analiz", "Model Karşılaştırma",
               "Test seti (2.000 müşteri) metrikleri, ROC/PR eğrileri, McNemar testi ve kalibrasyon",
               chip=f"Seçili: {MODEL_NAMES[key]}")

# ---------- 1) Metrik tablosu ----------
with st.container(key="card-metrics"):
    h1, h2 = st.columns([3, 1.2], vertical_alignment="center")
    with h1:
        ui.card_title("Test seti metrikleri",
                      "Ana ölçütler recall ve F2 (kaçırılan churn pahalı). PR-AUC ve ROC-AUC eşikten "
                      "bağımsızdır. Her sütunda en iyi değer vurgulu.")
    kind = h2.segmented_control("Eşik", ["F2-optimal", "0,50"], default="F2-optimal", required=True,
                                key="mk_thr", label_visibility="collapsed")
    m = art["metrics"].copy()
    m = m[m["Eşik türü"] == ("Optimize" if kind == "F2-optimal" else "0.50")].reset_index(drop=True)
    m = m[["Model", "Eşik", "Recall", "Precision", "F2", "PR-AUC", "ROC-AUC"]]
    m["Model"] = [f'<span class="iris-dot" style="background:{MODEL_COLORS[NAME_TO_KEY[n]]}"></span>{n}'
                  for n in m["Model"]]
    f3 = lambda v: ui.fmt_num(v, 3)  # noqa: E731
    ui.html(ui.table_html(m, fmt={"Eşik": lambda v: ui.fmt_num(v, 2), "Recall": f3, "Precision": f3,
                                  "F2": f3, "PR-AUC": f3, "ROC-AUC": f3},
                          num=("Eşik", "Recall", "Precision", "F2", "PR-AUC", "ROC-AUC"),
                          best={c: "max" for c in ("Recall", "Precision", "F2", "PR-AUC", "ROC-AUC")},
                          raw=("Model",)))

# ---------- 2) ROC ve PR eğrileri ----------
c1, c2 = st.columns(2)
roc_fig, pr_fig = go.Figure(), go.Figure()
for k, name in MODEL_NAMES.items():
    p = preds[f"proba_{k}"]
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    ap = float(art["metrics"].query("Model == @name")["PR-AUC"].iloc[0])
    width = 3 if k == key else 1.8
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({ui.fmt_num(auc(fpr, tpr), 3)})",
                                 line=dict(color=MODEL_COLORS[k], width=width),
                                 hovertemplate=f"{name}<br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}<extra></extra>"))
    pr_fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{name} ({ui.fmt_num(ap, 3)})",
                                line=dict(color=MODEL_COLORS[k], width=width, shape="hv"),
                                hovertemplate=f"{name}<br>Recall %{{x:.3f}}<br>Precision %{{y:.3f}}<extra></extra>"))
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Rastgele",
                             line=dict(color=ui.tok("neutral"), width=1, dash="dash"), hoverinfo="skip"))
pr_fig.add_hline(y=float(y.mean()), line=dict(color=ui.tok("neutral"), width=1, dash="dash"),
                 annotation_text=f"Rastgele (churn oranı {ui.fmt_pct(y.mean())})",
                 annotation_font=dict(color=ui.tok("muted"), size=11), annotation_position="bottom left")
for fig, xt, yt in [(roc_fig, "Yanlış pozitif oranı", "Doğru pozitif oranı (recall)"),
                    (pr_fig, "Recall", "Precision")]:
    fig.update_xaxes(title=xt, range=[0, 1], showgrid=True, gridcolor=ui.tok("grid"), tickformat=".1f")
    fig.update_yaxes(title=yt, range=[0, 1.02], tickformat=".1f")
# Açıklamalar grafiğin boş köşesinde, dikey (yatay dizilimde etiketler çakışıyordu)
_leg = dict(orientation="v", bgcolor=ui.tok("surface"), bordercolor=ui.tok("border"), borderwidth=1)
roc_fig.update_layout(legend=dict(_leg, x=0.98, xanchor="right", y=0.04, yanchor="bottom"))
pr_fig.update_layout(legend=dict(_leg, x=0.02, xanchor="left", y=0.28, yanchor="bottom"))
with c1, st.container(key="card-roc"):
    ui.card_title("ROC eğrisi", "Parantez içinde ROC-AUC; kalın çizgi seçili model")
    ui.plot(roc_fig, height=340, key="roc")
with c2, st.container(key="card-pr"):
    ui.card_title("Precision-Recall eğrisi", "Parantez içinde PR-AUC (ortalama precision); dengesiz "
                                             "veride ROC'tan daha bilgilendirici")
    ui.plot(pr_fig, height=340, key="pr")

# ---------- 3) McNemar + Holm ----------
mc = art["mcnemar"].copy()
res_col = [c for c in mc.columns if c.startswith("Sonuç")][0]
lines = []
for _, r in mc.iterrows():
    a, b = r["Karşılaştırma"].split(" vs ")
    n1, n2 = int(r["Yalnız 1. doğru"]), int(r["Yalnız 2. doğru"])
    better, worse, nb, nw = (a, b, n1, n2) if n1 >= n2 else (b, a, n2, n1)
    sig = r["p (Holm)"] < 0.05
    lines.append(f"<b>{better}</b>, {worse}'a göre {ui.fmt_num(nb)} müşteride tek başına doğru; tersi "
                 f"{ui.fmt_num(nw)} müşteri. Holm düzeltmeli p {ui.fmt_p(r['p (Holm)'])} → "
                 + ("hata oranları <b>anlamlı biçimde farklı</b>." if sig else
                    "fark <b>istatistiksel olarak anlamlı değil</b>."))
tab = pd.DataFrame({
    "Karşılaştırma": mc["Karşılaştırma"],
    "Yalnız 1. doğru": mc["Yalnız 1. doğru"], "Yalnız 2. doğru": mc["Yalnız 2. doğru"],
    "χ²": mc["İstatistik"], "p": mc["p"], "p (Holm)": mc["p (Holm)"],
    "Sonuç (α = 0,05)": [f'<span class="iris-pill {"warn" if s == "Anlamlı fark" else "flat"}">{s}</span>'
                         for s in mc[res_col]]})
with st.container(key="card-mcnemar"):
    ui.card_title("McNemar testi + Holm düzeltmesi",
                  "Aynı test müşterilerinde iki modelin doğru/yanlış sınıflamaları eşleştirilerek "
                  "karşılaştırılır; yalnız modellerin ayrıştığı müşteriler (uyumsuz çiftler) teste girer.")
    ui.html(ui.table_html(tab, fmt={"Yalnız 1. doğru": ui.fmt_num, "Yalnız 2. doğru": ui.fmt_num,
                                    "χ²": lambda v: ui.fmt_num(v, 2), "p": ui.fmt_p, "p (Holm)": ui.fmt_p},
                          num=("Yalnız 1. doğru", "Yalnız 2. doğru", "χ²", "p", "p (Holm)"),
                          raw=("Sonuç (α = 0,05)",)))
    ui.html(ui.callout("Yorum", lines, tone=ui.TURQ,
                       note="Modeller kendi F2-optimal eşiklerinde karşılaştırıldı (farklı eşikler). "
                            "Üç ikili karşılaştırma yapıldığı için p-değerleri Holm yöntemiyle "
                            "düzeltildi; α = 0,05."))

# ---------- 4) Kalibrasyon ----------
cal = art.get("calibration")
ui.html('<div class="iris-section"><div><div class="iris-card-title">Kalibrasyon</div>'
        '<div class="iris-card-sub">Model skorları gerçek olasılık gibi okunabilir mi? SMOTE-NC eğitimde '
        'churn payını yapay olarak artırdığı için skorların yukarı kayması beklenir.</div></div></div>')
if cal is None:
    st.info("Kalibrasyon dosyası yok. Önce `.\\.venv\\Scripts\\python.exe calibrate.py` çalıştırın.")
    st.stop()

s = cal["summary"]
rows = []
for k, name in MODEL_NAMES.items():
    i = s["models"][k]
    rows.append({"Model": f'<span class="iris-dot" style="background:{MODEL_COLORS[k]}"></span>{name}',
                 "Ort. skor (ham)": i["mean_pred_test"], "Ort. olasılık (kalibre)": i["mean_pred_test_cal"],
                 "Gerçek oran": s["actual_rate_test"], "Ham fark (puan)": i["gap_test"] * 100,
                 "Brier (ham)": i["brier_test"], "Brier (kalibre)": i["brier_test_cal"],
                 "Yöntem": "Platt" if i["method"] == "sigmoid" else "Isotonic",
                 "Uygulandı": "Evet" if i["applied"] else "Hayır"})
ct = pd.DataFrame(rows)
f3 = lambda v: ui.fmt_num(v, 4)  # noqa: E731
pc = lambda v: ui.fmt_pct(v, 1)  # noqa: E731

c1, c2 = st.columns([1.05, 1])
with c1, st.container(key="card-reliability"):
    h1, h2 = st.columns([1.4, 1], vertical_alignment="center")
    with h1:
        ui.card_title("Güvenilirlik eğrisi (test, 10 kutu)",
                      "Köşegen = mükemmel kalibrasyon. Nokta boyutu kutudaki müşteri sayısı.")
    ck = h2.segmented_control("Model", list(MODEL_NAMES), default=key, required=True,
                              format_func=lambda k: {"lr": "LR", "rf": "RF", "xgb": "XGB"}[k],
                              key="cal_model", label_visibility="collapsed")
    info = s["models"][ck]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Mükemmel kalibrasyon",
                             line=dict(color=ui.tok("neutral"), width=1, dash="dash"), hoverinfo="skip"))
    for label, rel, color in [("Ham skor", info["reliability_test"], ui.CORAL),
                              ("Kalibre", info["reliability_test_cal"], ui.TURQ)]:
        rel = pd.DataFrame(rel)
        fig.add_trace(go.Scatter(
            x=rel["mean_pred"], y=rel["frac_pos"], mode="lines+markers", name=label,
            line=dict(color=color, width=2.2),
            marker=dict(color=color, size=np.clip(np.sqrt(rel["n"]) * 1.1, 6, 26),
                        line=dict(color=ui.tok("surface"), width=1.5)),
            customdata=rel["n"],
            hovertemplate=f"{label}<br>Ort. tahmin %{{x:.3f}}<br>Gerçek oran %{{y:.3f}}"
                          "<br>n = %{customdata}<extra></extra>"))
    fig.update_xaxes(title="Tahmin edilen olasılık (kutu ortalaması)", range=[0, 1],
                     showgrid=True, gridcolor=ui.tok("grid"), tickformat=".1f")
    fig.update_yaxes(title="Gerçekleşen churn oranı", range=[0, 1.02], tickformat=".1f")
    fig.update_layout(legend=dict(orientation="v", x=0.02, xanchor="left", y=0.98, yanchor="top",
                                  bgcolor=ui.tok("surface"), bordercolor=ui.tok("border"), borderwidth=1))
    ui.plot(fig, height=340, key="reliability")

with c2, st.container(key="card-calib-method"):
    ui.card_title("Yöntem ve kullanım", f"Gerçekleşen churn oranı (test): <b>{pc(s['actual_rate_test'])}</b>")
    cv = s["models"][ck]["cv_brier_val"]
    chosen = s["models"][ck]["method"]
    mono = ("Platt (iki parametreli lojistik eğri) kesin monoton artan olduğu için müşteri sıralamasını, "
            "dolayısıyla ROC/PR eğrilerini değiştirmez." if chosen == "sigmoid" else
            "Isotonic monoton (azalmayan) olduğu için sıralamayı bozmaz; eşit değerli basamaklar oluşabilir.")
    ui.html(ui.callout(
        "Kalibrasyon nasıl yapıldı?",
        [f"Ham skorların ortalaması gerçek orandan {ui.fmt_num(s['material_gap'] * 100, 0)} puandan fazla "
         "saptığında kalibrasyon uygulanır. Kalibratör <b>yalnız validation setinde</b> eğitildi; test "
         "seti yalnız raporlama içindir.",
         "Isotonic ve Platt (sigmoid), validation içinde 5 katlı çapraz doğrulama Brier skoruyla "
         f"karşılaştırıldı ({MODEL_NAMES[ck]}: isotonic {ui.fmt_num(cv['isotonic'], 4)}, sigmoid "
         f"{ui.fmt_num(cv['sigmoid'], 4)}); düşük olan seçildi. " + mono,
         "<b>Uyarı listesi ve sınıflandırma</b> orijinal skor + orijinal eşikle yapılır (eşikler kalibre "
         "edilmemiş skorlar üzerinde seçildi). Kalibre olasılık yalnız <b>beklenen churn oranı</b> ve "
         "ekranda gösterilen müşteri olasılığı içindir."],
        tone=ui.AMBER))

with st.container(key="card-calib-table"):
    ui.card_title("Ortalama tahmin, gerçek oran ve Brier skoru (test seti)",
                  "Brier = ortalama (olasılık − gerçek)²; düşük olan iyi. Ham fark: ortalama skor − "
                  "gerçek oran (puan).")
    ui.html(ui.table_html(
        ct,
        fmt={"Ort. skor (ham)": pc, "Ort. olasılık (kalibre)": pc, "Gerçek oran": pc,
             "Ham fark (puan)": lambda v: ui.fmt_signed(v, 1), "Brier (ham)": f3, "Brier (kalibre)": f3},
        num=("Ort. skor (ham)", "Ort. olasılık (kalibre)", "Gerçek oran", "Ham fark (puan)",
             "Brier (ham)", "Brier (kalibre)"),
        best={"Brier (kalibre)": "min"}, raw=("Model",)))
