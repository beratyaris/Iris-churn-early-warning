"""Confounding kontrolü: Almanya'daki yüksek churn ülkeden mi, bakiye yapısından mı?

Tam veri seti (10.000 müşteri) üzerinde betimsel ve çıkarımsal istatistik; model tahmini
kullanılmaz. Hesaplar analysis.py'dedir; bu sayfa yalnız sunar.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis
import iris_ui as ui
from iris_data import load_data, wilson

COUNTRY_TR = {"France": "Fransa", "Germany": "Almanya", "Spain": "İspanya"}
C_COL = {"France": ui.tok("france"), "Germany": ui.CORAL, "Spain": ui.tok("neutral")}
TERM_TR = {"Germany": "Almanya", "Spain": "İspanya", "has_balance": "Bakiyesi var",
           "log_balance_c": "log(bakiye), merkezlenmiş", "age": "Yaş", "male": "Erkek",
           "active_member": "Aktif üye", "products_2": "2 ürün", "products_3plus": "3+ ürün"}


@st.cache_data
def confounding_results(df):
    counts, row_pct, chi = analysis.country_balance_table(df)
    strat = analysis.stratified_churn(df)
    bal_df, bal_stats = analysis.balance_distribution(df)
    germany, change, _, res_b = analysis.germany_comparison(df)
    return {"counts": counts, "row_pct": row_pct, "chi": chi, "strat": strat,
            "bal_df": bal_df, "bal_stats": bal_stats, "germany": germany, "change": change,
            "coef_b": analysis.odds_ratio_table(res_b),
            "text": analysis.interpret(strat, germany, change)}


df = load_data()
r = confounding_results(df)

ui.page_header("Analiz", "Confounding",
               "Özgünlük: Almanya'daki yüksek churn ülkenin kendisinden mi, yoksa bakiye yapısından mı "
               f"kaynaklanıyor? Tam veri ({ui.fmt_num(len(df))} müşteri), model tahmini kullanılmaz.")

ui.html(ui.callout("Sonuç (sonuçlardan otomatik üretilir)", r["text"][:-1], tone=ui.CORAL,
                   note=r["text"][-1]))

# ---------- 1) Ülke × bakiye ----------
c1, c2 = st.columns(2)
with c1, st.container(key="card-crosstab"):
    chi = r["chi"]
    ui.card_title("1) Ülke × bakiye durumu",
                  f"Satır yüzdeleri. Ki-kare bağımsızlık testi: χ² = {ui.fmt_num(chi['chi2'], 1)}, "
                  f"sd = {chi['dof']}, p {ui.fmt_p(chi['p'])} (en küçük beklenen frekans "
                  f"{ui.fmt_num(chi['min_expected'], 0)} ≥ 5).")
    rp, cnt = r["row_pct"], r["counts"]
    fig = go.Figure()
    for col, color in [("Bakiye = 0", ui.tok("neutral")), ("Bakiye > 0", ui.tok("ink"))]:
        fig.add_trace(go.Bar(
            y=[COUNTRY_TR[c] for c in rp.index], x=rp[col], name=col, orientation="h",
            marker=dict(color=color, cornerradius=3),
            text=[ui.fmt_pct(v / 100, 1) for v in rp[col]], textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color=ui.tok("surface") if col == "Bakiye > 0"
                          else ("#FFFFFF" if ui.mode() == "dark" else ui.NAVY)),
            customdata=cnt[col], hovertemplate=f"%{{y}} · {col}<br>%%{{x:.1f}} · %{{customdata:,}} müşteri<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_xaxes(range=[0, 100], ticksuffix="", tickprefix="%", showgrid=True,
                     gridcolor=ui.tok("grid"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    ui.plot(fig, height=260, key="crosstab")
    st.caption("Almanya'da bakiyesi 0 olan müşteri yok; bu yüzden ülke ve bakiye durumu iç içe geçmiş "
               "(olası karıştırıcı).")

# ---------- 2) Tabakalı churn ----------
strat = r["strat"].copy()
strat["k"] = np.round(strat["Churn oranı (%)"].fillna(0) * strat["n"] / 100).astype(int)
with c2, st.container(key="card-strat"):
    ui.card_title("2) Tabakalı churn oranları",
                  "Bakiye durumu sabitken ülkeler karşılaştırılır; ince çizgiler %95 Wilson GA. "
                  "Boş hücre: o tabakada o ülkeden müşteri yok.")
    fig = go.Figure()
    order = ["Tümü", "Bakiye = 0", "Bakiye > 0"]
    for c in analysis.COUNTRIES:
        s = strat[strat["Ülke"] == c].set_index("Tabaka").reindex(order)
        ci = np.array([wilson(k, n) if n else (np.nan, np.nan) for k, n in zip(s["k"], s["n"])])
        rate = s["Churn oranı (%)"].to_numpy()
        fig.add_trace(go.Bar(
            x=order, y=rate, name=COUNTRY_TR[c], marker=dict(color=C_COL[c], cornerradius=3),
            error_y=dict(type="data", symmetric=False, array=ci[:, 1] * 100 - rate,
                         arrayminus=rate - ci[:, 0] * 100, color=ui.tok("subtle"), thickness=1.2, width=3),
            customdata=np.c_[s["n"], ci * 100],
            hovertemplate=f"{COUNTRY_TR[c]} · %{{x}}<br>Churn %%{{y:.1f}} "
                          "(GA %%{customdata[1]:.1f} – %%{customdata[2]:.1f})<br>n = %{customdata[0]:,}"
                          "<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_yaxes(tickprefix="%", rangemode="tozero")
    ui.plot(fig, height=300, key="strat")

# ---------- 3) Bakiye dağılımı ----------
c3, c4 = st.columns(2)
with c3, st.container(key="card-balance"):
    ui.card_title("3) Bakiye dağılımı (yalnız bakiye > 0)",
                  "Kutular: medyan ve çeyrekler arası aralık; aykırı noktalar gösterilmez (bakiye/maaş "
                  "oranında çok düşük maaşlı birkaç müşteri ölçeği bozuyor).")
    bal = r["bal_df"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Bakiye", "Bakiye / maaş oranı"),
                        horizontal_spacing=0.12)
    for i, col in enumerate(["balance", "balance_salary_ratio"], start=1):
        for c in analysis.COUNTRIES:
            v = bal.loc[bal["country"] == c, col]
            q1, med, q3 = v.quantile([.25, .5, .75])
            iqr = q3 - q1
            lo, hi = v[v >= q1 - 1.5 * iqr].min(), v[v <= q3 + 1.5 * iqr].max()
            fig.add_trace(go.Box(
                x=[COUNTRY_TR[c]], q1=[q1], median=[med], q3=[q3], lowerfence=[lo], upperfence=[hi],
                name=COUNTRY_TR[c], marker_color=C_COL[c], fillcolor=C_COL[c], opacity=.75,
                line=dict(color=ui.tok("ink"), width=1.2), showlegend=False, boxpoints=False,
                hovertemplate=None), row=1, col=i)
    for a in fig.layout.annotations:
        a.font = dict(color=ui.tok("muted"), size=12)
    fig.update_yaxes(showgrid=True, gridcolor=ui.tok("grid"))
    fig.update_layout(margin=dict(l=4, r=4, t=34, b=4))  # alt grafik başlıklarına yer
    ui.plot(fig, height=300, key="balance-box")
    bs = r["bal_stats"].copy()
    bs["Değişken"] = bs["Değişken"].map({"balance": "Bakiye", "balance_salary_ratio": "Bakiye / maaş"})
    bs["Ülke"] = bs["Ülke"].map(COUNTRY_TR)
    num_fmt = {c: (lambda v: ui.fmt_num(v, 2)) for c in ("Q1", "Medyan", "Q3", "IQR")}
    num_fmt["n"] = ui.fmt_num
    with st.expander("Medyan ve IQR tablosu"):
        ui.html(ui.table_html(bs, fmt=num_fmt, num=("n", "Q1", "Medyan", "Q3", "IQR")))

# ---------- 4) Lojistik regresyon: Germany OR ----------
g = r["germany"]
with c4, st.container(key="card-or"):
    ui.card_title("4) Lojistik regresyon: Almanya odds ratio'su (ref. Fransa)",
                  f"Kontroller eklenince OR değişimi: <b>{'−' if r['change'] < 0 else '+'}"
                  f"{ui.fmt_pct(abs(r['change']) / 100)}</b>. "
                  "Nokta: OR, çizgi: %95 Wald GA; kesikli çizgi OR = 1 (etki yok).")
    fig = go.Figure()
    names = ["(a) Yalnız ülke", "(b) Ülke + kontroller"]
    fig.add_trace(go.Scatter(
        x=g["OR"], y=names, mode="markers",
        marker=dict(color=[ui.tok("neutral"), ui.CORAL], size=14, symbol="square",
                    line=dict(color=ui.tok("surface"), width=1.5)),
        error_x=dict(type="data", symmetric=False, array=g["GA üst"] - g["OR"],
                     arrayminus=g["OR"] - g["GA alt"], color=ui.tok("ink"), thickness=2, width=6),
        text=[f"OR {ui.fmt_num(o, 2)} [{ui.fmt_num(a, 2)}; {ui.fmt_num(b, 2)}]"
              for o, a, b in zip(g["OR"], g["GA alt"], g["GA üst"])],
        textposition="top center", hovertemplate="%{y}<br>%{text}<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=ui.tok("subtle"), width=1, dash="dash"))
    fig.update_xaxes(type="log", title="Odds ratio (log ölçek)", showgrid=True, gridcolor=ui.tok("grid"),
                     range=[np.log10(0.8), np.log10(max(g["GA üst"].max() * 1.25, 3))])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(showlegend=False)
    ui.plot(fig, height=230, key="or-forest")
    gt = g.copy()
    gt["Model"] = names
    ui.html(ui.table_html(gt[["Model", "OR", "GA alt", "GA üst", "p", "Pseudo R²"]],
                          fmt={"OR": lambda v: ui.fmt_num(v, 3), "GA alt": lambda v: ui.fmt_num(v, 3),
                               "GA üst": lambda v: ui.fmt_num(v, 3), "p": ui.fmt_p,
                               "Pseudo R²": lambda v: ui.fmt_num(v, 3)},
                          num=("OR", "GA alt", "GA üst", "p", "Pseudo R²")))
    st.caption(f"Her iki model n = {ui.fmt_num(int(g['n'].iloc[0]))} müşteri. Model (b) kontrolleri: bakiyesi var, merkezlenmiş log(bakiye) (bakiye > 0 için), yaş, "
               "cinsiyet, aktif üyelik, ürün sayısı (1 / 2 / 3+). Birincil hipotez tek (Almanya "
               "katsayısı), bu yüzden çoklu karşılaştırma düzeltmesi yapılmadı.")

with st.expander("Model (b) tüm katsayılar (odds ratio) · diğer p-değerleri yalnız betimsel"):
    coef = r["coef_b"].copy()
    coef.insert(0, "Değişken", [TERM_TR.get(i, i) for i in coef.index])
    ui.html(ui.table_html(coef.reset_index(drop=True),
                          fmt={"OR": lambda v: ui.fmt_num(v, 3), "GA alt": lambda v: ui.fmt_num(v, 3),
                               "GA üst": lambda v: ui.fmt_num(v, 3), "p": ui.fmt_p},
                          num=("OR", "GA alt", "GA üst", "p")))
