"""Açıklanabilirlik (SHAP): global değişken önemi, beeswarm ve bağımlılık grafiği.

SHAP değerleri train.py'nin test seti için kaydettiği değerlerdir (shap_test.joblib):
XGBoost log-odds, Random Forest olasılık ölçeğinde. Grafikte ham (ölçeklenmemiş) değişken
değerleri gösterilir; ölçekleme monoton olduğu için yorum değişmez.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import iris_ui as ui
from iris_data import BINARY, FEATURE_TR, SHAP_UNIT, current_settings, load_artifacts
from train import MODEL_NAMES

art = load_artifacts()
key, _ = current_settings(art)
features = art["features"]
X = art["preds"][features]

ui.page_header("Analiz", "Açıklanabilirlik (SHAP)",
               "Model hangi değişkenlere dayanıyor ve bu değişkenler churn skorunu hangi yöne itiyor?")

with st.container(key="filterbar"):
    c1, c2 = st.columns([1, 2.4], vertical_alignment="center")
    default = key if key in ("xgb", "rf") else "xgb"
    mk = c1.segmented_control("Model", ["xgb", "rf"], default=default, required=True,
                              format_func=MODEL_NAMES.get, key="shap_model")
    c2.html('<div class="iris-card-sub">SHAP değerleri test setindeki 2.000 müşteri için eğitim '
            f'sırasında hesaplandı (ağaç modelleri). Birim: <b>{SHAP_UNIT[mk]}</b>. Lojistik '
            'Regresyon için tek müşteri açıklaması Müşteri Risk Skoru sayfasındadır.</div>')

s = art["shap"][mk]
vals = np.asarray(s["values"])
imp = pd.Series(np.abs(vals).mean(axis=0), index=features).sort_values(ascending=False)
labels = {f: FEATURE_TR[f] for f in features}
dec = 3 if SHAP_UNIT[mk] == "olasılık" else 2

c1, c2 = st.columns([1, 1.25])
with c1, st.container(key="card-importance"):
    ui.card_title("Global önem", f"Ortalama |SHAP| ({SHAP_UNIT[mk]}): değişkenin skoru ortalamada ne "
                                 "kadar değiştirdiği (yönden bağımsız)")
    r = imp.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=r.values, y=[labels[f] for f in r.index], orientation="h",
        marker=dict(color=[ui.TURQ if i >= len(r) - 5 else ui.tok("neutral") for i in range(len(r))],
                    cornerradius=4),
        text=[ui.fmt_num(v, dec + 1) for v in r.values], textposition="outside", cliponaxis=False,
        textfont=dict(color=ui.tok("ink"), size=11),
        hovertemplate="%{y}<br>Ortalama |SHAP|: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(showgrid=True, gridcolor=ui.tok("grid"), title=f"Ortalama |SHAP| ({SHAP_UNIT[mk]})")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=4, r=40, t=8, b=4), bargap=0.3)
    ui.plot(fig, height=440, key="shap-imp")

# Beeswarm benzeri: her nokta bir müşteri; dikey yayılma yoğunluğa göre (deterministik)
TOP = 8
top_feats = list(imp.index[:TOP])
# Renk = değişkenin değeri, Iris'in risk renk diliyle (logo/gösterge ile aynı üç durak):
# düşük turkuaz → orta amber → yüksek mercan. İki temada da aynı (doygun renkler iki zeminde okunur).
scale = [[0, ui.TURQ], [0.5, ui.AMBER], [1, ui.CORAL]]
rng = np.random.default_rng(42)
xs, ys, cs, hv = [], [], [], []
for row, f in enumerate(top_feats):
    j = features.index(f)
    v = vals[:, j]
    raw = X[f].to_numpy(dtype=float)
    # renk: değişkenin kendi içindeki sırası; 0/1 değişkenlerde yalnız iki uç (Hayır turkuaz, Evet mercan)
    rank = (raw >= 0.5).astype(float) if f in BINARY else pd.Series(raw).rank(pct=True).to_numpy()
    counts, edges = np.histogram(v, bins=40)
    dens = counts[np.clip(np.digitize(v, edges) - 1, 0, len(counts) - 1)] / max(counts.max(), 1)
    xs.append(v)
    ys.append(TOP - 1 - row + rng.uniform(-1, 1, len(v)) * 0.38 * np.sqrt(dens))
    cs.append(rank)
    hv.append([f"{labels[f]} = {'Evet' if (f in BINARY and x >= .5) else 'Hayır' if f in BINARY else ui.fmt_num(x, 2 if f == 'balance_salary_ratio' else 0)}"
               for x in raw])
with c2, st.container(key="card-beeswarm"):
    ui.card_title("Beeswarm: etkinin yönü ve dağılımı",
                  "Her nokta bir test müşterisi. Sağ = skoru artırır, sol = azaltır. Renk, değişkenin "
                  "o müşterideki değerinin göreli düzeyi (turkuaz = düşük, amber = orta, mercan = yüksek; "
                  "evet/hayır değişkenlerinde turkuaz = Hayır, mercan = Evet).")
    fig = go.Figure(go.Scattergl(
        x=np.concatenate(xs), y=np.concatenate(ys), mode="markers",
        marker=dict(size=4.5, color=np.concatenate(cs), colorscale=scale, opacity=.8, cmin=0, cmax=1,
                    colorbar=dict(title=dict(text="Değer", side="right"), tickvals=[0, 1],
                                  ticktext=["Düşük", "Yüksek"], thickness=10, len=.7)),
        text=sum(hv, []), hovertemplate="%{text}<br>SHAP: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=ui.tok("subtle"), width=1))
    fig.update_yaxes(tickvals=list(range(TOP)), ticktext=[labels[f] for f in top_feats][::-1],
                     showgrid=False, range=[-0.7, TOP - 0.3])
    fig.update_xaxes(title=f"SHAP değeri ({SHAP_UNIT[mk]})", showgrid=True, gridcolor=ui.tok("grid"),
                     zeroline=False)
    ui.plot(fig, height=440, key="shap-bee")

with st.container(key="card-dependence"):
    h1, h2 = st.columns([2.2, 1], vertical_alignment="center")
    with h1:
        ui.card_title("Bağımlılık grafiği",
                      "Seçilen değişkenin değeri (x) ile SHAP katkısı (y). Mercan: skoru artıran, "
                      "turkuaz: azaltan müşteriler.")
    feat = h2.selectbox("Değişken", top_feats, index=0, format_func=labels.get, key="shap_dep",
                        label_visibility="collapsed")
    j = features.index(feat)
    v, raw = vals[:, j], X[feat].to_numpy(dtype=float)
    jit = rng.uniform(-0.12, 0.12, len(raw)) if (feat in BINARY or feat in ("products_number", "tenure")) else 0
    fig = go.Figure(go.Scattergl(
        x=raw + jit, y=v, mode="markers",
        marker=dict(size=6, color=np.where(v > 0, ui.CORAL, ui.TURQ), opacity=.7),
        customdata=raw, hovertemplate=f"{labels[feat]} = %{{customdata:,.2f}}<br>SHAP: %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=ui.tok("subtle"), width=1))
    fig.update_xaxes(title=labels[feat], showgrid=True, gridcolor=ui.tok("grid"))
    if feat in BINARY:
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Hayır", "Evet"])
    fig.update_yaxes(title=f"SHAP ({SHAP_UNIT[mk]})")
    ui.plot(fig, height=320, key="shap-dep")
    pos = v > 0
    st.caption(f"{labels[feat]}: skoru artırdığı müşteri payı {ui.fmt_pct(pos.mean())}. "
               f"Ortalama |SHAP| = {ui.fmt_num(imp[feat], 4)} ({SHAP_UNIT[mk]}). SHAP ilişkiyi açıklar, "
               "nedensellik göstermez.")
