"""Genel Bakış: portföy profili, filtreler, KPI kartları ve risk renkli churn kırılımları.

Tüm sayılar tam veri setinden (10.000 müşteri) ya da test seti tahminlerinden hesaplanır.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import iris_ui as ui
from iris_data import current_settings, load_artifacts, load_data, test_frame
from train import MODEL_NAMES

COUNTRY_TR = {"France": "Fransa", "Germany": "Almanya", "Spain": "İspanya"}
GENDER_TR = {"Female": "Kadın", "Male": "Erkek"}
AGE_BINS, AGE_LABELS = [17, 29, 39, 49, 59, 120], ["18-29", "30-39", "40-49", "50-59", "60+"]

df, art = load_data(), load_artifacts()
key, thr = current_settings(art)
tf = test_frame(art)

ui.page_header("Karar Destek", "Genel Bakış",
               f"Portföy profili ve churn kırılımları · tam veri ({ui.fmt_num(len(df))} müşteri)",
               chip=f"{MODEL_NAMES[key]} · eşik {ui.fmt_num(thr, 2)}")

# ---------- Filtreler ----------
with st.container(key="filterbar"):
    c1, c2, c3, c4, c5 = st.columns([1.3, 0.9, 1.05, 1.15, 1.1])
    countries = c1.pills("Ülke", list(COUNTRY_TR), selection_mode="multi",
                         default=list(COUNTRY_TR), format_func=COUNTRY_TR.get, key="f_country")
    genders = c2.pills("Cinsiyet", list(GENDER_TR), selection_mode="multi",
                       default=list(GENDER_TR), format_func=GENDER_TR.get, key="f_gender")
    products = c3.pills("Ürün sayısı", [1, 2, 3, 4], selection_mode="multi",
                        default=[1, 2, 3, 4], key="f_products")
    member = c4.segmented_control("Üyelik", ["Tümü", "Aktif", "Pasif"], default="Tümü",
                                  required=True, key="f_member")
    balance = c5.segmented_control("Bakiye", ["Tümü", "> 0", "= 0"], default="Tümü",
                                   required=True, key="f_balance")


def apply_filters(d):
    """Boş çoklu seçim 'hepsi' anlamına gelir (filtre uygulanmaz)."""
    m = pd.Series(True, index=d.index)
    if countries:
        m &= d["country"].isin(countries)
    if genders:
        m &= d["gender"].isin(genders)
    if products:
        m &= d["products_number"].isin(products)
    if member == "Aktif":
        m &= d["active_member"] == 1
    elif member == "Pasif":
        m &= d["active_member"] == 0
    if balance == "> 0":
        m &= d["balance"] > 0
    elif balance == "= 0":
        m &= d["balance"] == 0
    return d[m]


d = apply_filters(df)
t = apply_filters(tf)
if d.empty:
    st.info("Bu filtre birleşiminde müşteri yok. Filtrelerden birini genişletin.")
    st.stop()

base_rate = df["churn"].mean()
rate = d["churn"].mean()

# ---------- KPI kartları ----------
# 1) Müşteri sayısı + yaş dağılımı
age_hist, age_edges = np.histogram(d["age"], bins=np.arange(18, 96, 6))

card_n = ui.kpi_card(
    "Müşteri", ui.count(len(d)),
    sub=f"Portföy payı <b>{ui.fmt_pct(len(d) / len(df))}</b> · ortalama yaş "
        f"<b>{ui.fmt_num(d['age'].mean(), 1)}</b>",
    mini=ui.minibars(age_hist, ["#9FB3C8"] * len(age_hist), title="Yaş dağılımı"),
    cap="Mini grafik: yaş dağılımı (18 → 95)", tone=ui.NAVY)

# 2) Churn oranı + müşteri süresine göre churn (sparkline)
by_tenure = d.groupby("tenure")["churn"].mean().reindex(range(11))
card_rate = ui.kpi_card(
    "Churn oranı", ui.count(rate * 100, 2, pre="%"),
    sub=f"{ui.delta_badge((rate - base_rate) * 100)} portföy ortalamasına "
        f"({ui.fmt_pct(base_rate, 2)}) göre",
    mini=ui.sparkline(by_tenure.interpolate().bfill().ffill(), ui.rate_color(rate),
                      title="Müşteri süresine (0-10 yıl) göre churn oranı"),
    cap="Mini grafik: müşteri süresine göre churn (0 → 10 yıl)", tone=ui.rate_color(rate))

# 3) Terk eden müşteri + bakiye durumuna göre churn
zero = d["balance"] == 0
r0 = d.loc[zero, "churn"].mean() if zero.any() else np.nan
r1 = d.loc[~zero, "churn"].mean() if (~zero).any() else np.nan
bal_vals = [0 if np.isnan(r) else r for r in (r0, r1)]
card_churned = ui.kpi_card(
    "Terk eden müşteri", ui.count(int(d["churn"].sum())),
    sub=f"Sıfır bakiyeli: <b>{ui.fmt_pct(zero.mean())}</b> · churn "
        f"<b>{'–' if np.isnan(r0) else ui.fmt_pct(r0)}</b> (bakiye = 0) / "
        f"<b>{'–' if np.isnan(r1) else ui.fmt_pct(r1)}</b> (&gt; 0)",
    mini=ui.minibars(bal_vals, [ui.rate_color(v) for v in bal_vals], width=56,
                     title=f"Churn: bakiye = 0 {ui.fmt_pct(bal_vals[0])}, "
                           f"bakiye > 0 {ui.fmt_pct(bal_vals[1])}"),
    cap="Mini grafik: bakiye = 0 / > 0 churn oranı", tone=ui.CORAL)

# 4) Model uyarısı (test setinin filtreye uyan kısmı)
if len(t):
    proba = t[f"proba_{key}"]
    alert = proba >= thr
    pos = int(t["churn"].sum())
    tp = int((alert & (t["churn"] == 1)).sum())
    ph, pe = np.histogram(proba, bins=np.linspace(0, 1, 21))
    card_alert = ui.kpi_card(
        "Model uyarısı (test)", ui.count(int(alert.sum())) +
        f'<small> / {ui.fmt_num(len(t))}</small>',
        sub=f"Yakalanan churn <b>{tp} / {pos}</b> · recall "
            f"<b>{ui.fmt_num(tp / pos, 2) if pos else '–'}</b>",
        mini=ui.minibars(ph, [ui.proba_color((a + b) / 2, thr) for a, b in zip(pe[:-1], pe[1:])],
                         title="Churn olasılığı dağılımı (test)", marker=thr),
        cap=f"Mini grafik: churn olasılığı dağılımı, kesikli çizgi eşik ({ui.fmt_num(thr, 2)})",
        tone=ui.AMBER)
else:
    card_alert = ui.kpi_card("Model uyarısı (test)", "–",
                             sub="Filtreye uyan test müşterisi yok.", tone=ui.AMBER)

ui.kpi_row([card_n, card_rate, card_churned, card_alert])


# ---------- Grafikler ----------
def rate_bar(frame, col, order, labels=None):
    g = (frame.groupby(col, observed=True)["churn"].agg(["mean", "size", "sum"])
         .reindex(order).dropna())
    x = [labels.get(v, v) if labels else str(v) for v in g.index]
    fig = go.Figure(go.Bar(
        x=x, y=g["mean"] * 100,
        marker=dict(color=[ui.rate_color(v) for v in g["mean"]], cornerradius=4),
        text=[ui.fmt_pct(v) for v in g["mean"]], textposition="outside",
        textfont=dict(color=ui.NAVY, size=12), cliponaxis=False,
        customdata=np.c_[g["size"], g["sum"]],
        hovertemplate="<b>%{x}</b><br>Churn oranı: %%{y:.1f}<br>Müşteri: %{customdata[0]:,.0f}"
                      "<br>Terk eden: %{customdata[1]:,.0f}<extra></extra>"))
    fig.add_hline(y=rate * 100, line=dict(color=ui.SUBTLE, width=1.2, dash="dot"))
    top = max(g["mean"].max() * 100, rate * 100)
    fig.update_yaxes(range=[0, top * 1.22], ticksuffix="", tickprefix="%", nticks=5)
    fig.update_layout(showlegend=False, bargap=0.42)
    return fig


chart_d = d.assign(age_group=pd.cut(d["age"], AGE_BINS, labels=AGE_LABELS),
                   member=d["active_member"].map({1: "Aktif", 0: "Pasif"}),
                   has_bal=np.where(d["balance"] > 0, "Bakiye > 0", "Bakiye = 0"))

charts = [
    ("Ülkeye göre churn", "country", list(COUNTRY_TR), COUNTRY_TR),
    ("Cinsiyete göre churn", "gender", list(GENDER_TR), GENDER_TR),
    ("Aktif üyeliğe göre churn", "member", ["Aktif", "Pasif"], None),
    ("Yaş grubuna göre churn", "age_group", AGE_LABELS, None),
    ("Ürün sayısına göre churn", "products_number", [1, 2, 3, 4], None),
    ("Bakiye durumuna göre churn", "has_bal", ["Bakiye = 0", "Bakiye > 0"], None),
]
ui.html('<div class="iris-section"><div><div class="iris-card-title">Churn kırılımları</div>'
        '<div class="iris-card-sub">Çubuk rengi churn oranını gösterir; üzerine gelince '
        'müşteri ve terk sayısı görünür</div></div>'
        + ui.band_legend() + '</div>')
for row in (charts[:3], charts[3:]):
    cols = st.columns(3)
    for col, (title, field, order, labels) in zip(cols, row):
        with col, st.container(key=f"card-{field}"):
            ui.card_title(title, f"Kesikli çizgi: seçimin ortalaması ({ui.fmt_pct(rate)})")
            ui.plot(rate_bar(chart_d, field, order, labels), height=250, key=f"bar-{field}")

# Yaş dağılımı: kalan vs terk eden
with st.container(key="card-age-dist"):
    ui.card_title("Yaş dağılımı: kalan ve terk eden müşteriler",
                  "Seçimdeki müşteriler, 3 yaşlık aralıklar; çubuklar üst üste (toplam = aralıktaki müşteri)")
    fig = go.Figure()
    for label, flag, color in [("Kaldı", 0, ui.NEUTRAL), ("Terk etti", 1, ui.CORAL)]:
        ages = d.loc[d["churn"] == flag, "age"]
        fig.add_trace(go.Histogram(
            x=ages, name=f"{label} ({ui.fmt_num(len(ages))})", xbins=dict(start=18, end=93, size=3),
            marker=dict(color=color, line=dict(color="#FFFFFF", width=1)),
            hovertemplate=f"<b>{label}</b><br>Yaş %{{x}}<br>Müşteri: %{{y:,.0f}}<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.08)
    fig.update_xaxes(title="Yaş")
    fig.update_yaxes(title="Müşteri sayısı")
    ui.plot(fig, height=280, key="age-dist")

st.caption(f"Eksik değer: {int(df.isna().sum().sum())}. Çubuk renkleri churn oranı bantlarıdır "
           f"({', '.join(ui.BAND_LABELS)}); portföy ortalaması {ui.fmt_pct(base_rate)}. "
           "Model uyarısı kartı yalnız test setini (eğitimde görülmemiş müşteriler) kullanır; "
           "diğer kartlar ve grafikler tam veriyi.")
