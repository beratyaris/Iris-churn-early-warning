"""Genel Bakış: portföy profili, filtreler, KPI kartları ve churn kırılımları.

Tüm sayılar tam veri setinden (10.000 müşteri) ya da test seti tahminlerinden hesaplanır.
Çubuk renkleri sabit bantlarla değil, portföy ortalamasına göre belirlenir: grubun churn
oranı için %95 Wilson güven aralığı ortalamanın tamamen altındaysa turkuaz, ortalamayı
içeriyorsa amber, tamamen üstündeyse mercan.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import iris_ui as ui
from iris_data import current_settings, load_artifacts, load_data, test_frame, wilson
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

base_rate = float(df["churn"].mean())
rate = float(d["churn"].mean())
filtered = len(d) < len(df)


def group_color(k, n):
    lo, hi = wilson(k, n)
    return ui.rel_color(lo, hi, base_rate)


# ---------- KPI kartları ----------
# 1) Müşteri sayısı + yaş dağılımı
age_hist, _ = np.histogram(d["age"], bins=np.arange(18, 96, 6))
card_n = ui.kpi_card(
    "Müşteri", ui.count(len(d)),
    sub=f"Portföy payı <b>{ui.fmt_pct(len(d) / len(df))}</b> · ortalama yaş "
        f"<b>{ui.fmt_num(d['age'].mean(), 1)}</b>",
    mini=ui.minibars(age_hist, [ui.tok("neutral")] * len(age_hist), title="Yaş dağılımı (18 → 95)"),
    tip="Mini grafik: seçimdeki müşterilerin yaş dağılımı (18 → 95, 6 yaşlık aralıklar).",
    tone=ui.tok("ink"))

# 2) Churn oranı + müşteri süresine göre churn (sparkline)
by_tenure = d.groupby("tenure")["churn"].mean().reindex(range(11))
rate_col = group_color(int(d["churn"].sum()), len(d)) if filtered else ui.AMBER
rate_sub = (f"{ui.delta_badge((rate - base_rate) * 100)} portföy ortalamasına "
            f"({ui.fmt_pct(base_rate, 2)}) göre" if filtered
            else "Portföyün tamamı · filtre uygulanınca ortalamaya göre fark görünür")
card_rate = ui.kpi_card(
    "Churn oranı", ui.count(rate * 100, 2, pre="%"), sub=rate_sub,
    mini=ui.sparkline(by_tenure.interpolate().bfill().ffill(), rate_col,
                      title="Müşteri süresine (0 → 10 yıl) göre churn oranı"),
    tip="Mini grafik: müşteri süresine (0 → 10 yıl) göre churn oranı. Kart rengi: seçimin "
        "%95 Wilson GA'sı portföy ortalamasına göre (altında / ayırt edilemez / üstünde).",
    tone=rate_col)

# 3) Terk eden müşteri + bakiye durumuna göre churn
zero = d["balance"] == 0
r0 = d.loc[zero, "churn"].mean() if zero.any() else np.nan
r1 = d.loc[~zero, "churn"].mean() if (~zero).any() else np.nan
bal_vals = [0 if np.isnan(r) else r for r in (r0, r1)]
bal_cols = [group_color(int(d.loc[m, "churn"].sum()), int(m.sum())) if m.any() else ui.tok("neutral")
            for m in (zero, ~zero)]
card_churned = ui.kpi_card(
    "Terk eden müşteri", ui.count(int(d["churn"].sum())),
    sub=f"Sıfır bakiyeli: <b>{ui.fmt_pct(zero.mean())}</b> · churn "
        f"<b>{'–' if np.isnan(r0) else ui.fmt_pct(r0)}</b> (bakiye = 0) / "
        f"<b>{'–' if np.isnan(r1) else ui.fmt_pct(r1)}</b> (&gt; 0)",
    mini=ui.minibars(bal_vals, bal_cols, width=56,
                     title=f"Churn: bakiye = 0 {ui.fmt_pct(bal_vals[0])}, "
                           f"bakiye > 0 {ui.fmt_pct(bal_vals[1])}"),
    tip="Mini grafik: bakiye = 0 (sol) ve bakiye > 0 (sağ) müşterilerde churn oranı; "
        "renkler portföy ortalamasına göre.",
    tone=ui.CORAL)

# 4) Uyarı listesi (test setinin filtreye uyan kısmı)
if len(t):
    proba = t[f"proba_{key}"]
    alert = proba >= thr
    pos = int(t["churn"].sum())
    tp = int((alert & (t["churn"] == 1)).sum())
    ph, pe = np.histogram(proba, bins=np.linspace(0, 1, 21))
    card_alert = ui.kpi_card(
        "Uyarı listesi (test)", ui.count(int(alert.sum())) +
        f'<small> / {ui.fmt_num(len(t))}</small>',
        sub=f"Yakalanan churn <b>{tp} / {pos}</b> · recall "
            f"<b>{ui.fmt_num(tp / pos, 2) if pos else '–'}</b>",
        mini=ui.minibars(ph, [ui.proba_color((a + b) / 2, thr) for a, b in zip(pe[:-1], pe[1:])],
                         title="Model skoru dağılımı (test)", marker=thr),
        tip=f"Skoru eşiğe ({ui.fmt_num(thr, 2)}) eşit veya üstünde olan test müşterileri. "
            "Mini grafik: model skoru dağılımı; kesikli çizgi eşik, mercan çubuklar listede.",
        tone=ui.AMBER)
else:
    card_alert = ui.kpi_card("Uyarı listesi (test)", "–",
                             sub="Filtreye uyan test müşterisi yok.", tone=ui.AMBER)

ui.kpi_row([card_n, card_rate, card_churned, card_alert])


# ---------- Grafikler ----------
def rate_bar(frame, col, order, labels=None):
    g = (frame.groupby(col, observed=True)["churn"].agg(["mean", "size", "sum"])
         .reindex(order).dropna())
    ci = np.array([wilson(int(k), int(n)) for k, n in zip(g["sum"], g["size"])])
    colors = [ui.rel_color(lo, hi, base_rate) for lo, hi in ci]
    x = [labels.get(v, v) if labels else str(v) for v in g.index]
    fig = go.Figure(go.Bar(
        x=x, y=g["mean"] * 100,
        marker=dict(color=colors, cornerradius=4),
        error_y=dict(type="data", symmetric=False, array=(ci[:, 1] - g["mean"]) * 100,
                     arrayminus=(g["mean"] - ci[:, 0]) * 100, color=ui.tok("subtle"),
                     thickness=1.2, width=4),
        customdata=np.c_[g["size"], g["sum"], ci * 100],
        hovertemplate="<b>%{x}</b><br>Churn oranı: %%{y:.1f}"
                      "<br>%95 Wilson GA: %%{customdata[2]:.1f} – %%{customdata[3]:.1f}"
                      "<br>Müşteri: %{customdata[0]:,.0f}<br>Terk eden: %{customdata[1]:,.0f}"
                      "<extra></extra>"))
    # Değer etiketi GA bıyığının üstünde (bıyıkla çakışmasın)
    # x = kategori sırası (0, 1, ...): "1".."4" gibi etiketler sayı sanılıp kaymasın
    for xi, (m, hi) in enumerate(zip(g["mean"], ci[:, 1])):
        fig.add_annotation(x=xi, y=hi * 100, text=ui.fmt_pct(m), showarrow=False, yshift=11,
                           font=dict(color=ui.tok("ink"), size=12))
    fig.add_hline(y=base_rate * 100, line=dict(color=ui.tok("subtle"), width=1.4, dash="dot"))
    top = max(ci[:, 1].max() * 100, base_rate * 100)
    fig.update_yaxes(range=[0, top * 1.2], tickprefix="%", nticks=5)
    fig.update_xaxes(type="category")  # "1".."4" gibi etiketler sayı eksenine dönüşmesin
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
        '<div class="iris-card-sub">Renk, grubun portföy ortalamasından istatistiksel olarak '
        'ayrışıp ayrışmadığını gösterir; ince çizgiler %95 Wilson GA</div></div>'
        + ui.rel_legend() + '</div>')
for row in (charts[:3], charts[3:]):
    cols = st.columns(3)
    for col, (title, field, order, labels) in zip(cols, row):
        with col, st.container(key=f"card-{field}"):
            ui.card_title(title, f"Kesikli çizgi: portföy ortalaması ({ui.fmt_pct(base_rate)})")
            ui.plot(rate_bar(chart_d, field, order, labels), height=250, key=f"bar-{field}")

# Yaş dağılımı: kalan vs terk eden
with st.container(key="card-age-dist"):
    ui.card_title("Yaş dağılımı: kalan ve terk eden müşteriler",
                  "Seçimdeki müşteriler, 3 yaşlık aralıklar; çubuklar üst üste (toplam = aralıktaki müşteri)")
    fig = go.Figure()
    for label, flag, color in [("Kaldı", 0, ui.tok("neutral")), ("Terk etti", 1, ui.CORAL)]:
        ages = d.loc[d["churn"] == flag, "age"]
        fig.add_trace(go.Histogram(
            x=ages, name=f"{label} ({ui.fmt_num(len(ages))})", xbins=dict(start=18, end=93, size=3),
            marker=dict(color=color, line=dict(color=ui.tok("surface"), width=1)),
            hovertemplate=f"<b>{label}</b><br>Yaş %{{x}}<br>Müşteri: %{{y:,.0f}}<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.08)
    fig.update_xaxes(title="Yaş")
    fig.update_yaxes(title="Müşteri sayısı")
    ui.plot(fig, height=280, key="age-dist")

st.caption(f"Eksik değer: {int(df.isna().sum().sum())}. Renk kuralı: grubun churn oranı için %95 "
           f"Wilson güven aralığı portföy ortalamasının ({ui.fmt_pct(base_rate, 2)}) tamamen altındaysa "
           "turkuaz, ortalamayı içeriyorsa amber (ayırt edilemez), tamamen üstündeyse mercan. "
           "Uyarı listesi kartı yalnız test setini (eğitimde görülmemiş müşteriler) kullanır; "
           "diğer kartlar ve grafikler tam veriyi.")
