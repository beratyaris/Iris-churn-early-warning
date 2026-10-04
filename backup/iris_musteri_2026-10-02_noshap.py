"""Tek müşteri bileşenleri: Müşteri Risk Skoru ve "Model ne derdi?" simülatörü ortak kullanır.

- customer_form: ham veri biçiminde müşteri girişi; sınırlar EĞİTİM verisindeki gerçek aralık
  (iris_data.input_bounds), kategorik seçenekler veriden.
- gauge_html: tek ölçek kuralı. Göstergede yalnız kalibre "gitme olasılığı"; uyarı durumu ayrı
  satırda kendi ölçeğinde (model skoru ≥ eşik).
- shap_waterfall: modelin kendi çıktı ölçeğinde (XGBoost/LR log-odds, RF olasılık) SHAP şelalesi.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import iris_ui as ui
from iris_data import FEATURE_TR, SHAP_UNIT, calibrate, explain_row, fmt_feature, input_bounds
from iris_karar import COUNTRY_TR, GENDER_TR
from train import prepare_features

# Elle giriş için başlangıç profili (Risk Skoru'nun önceki varsayılanları; hepsi eğitim aralığında)
DEFAULT_PROFILE = {"credit_score": 650, "country": "France", "gender": "Female", "age": 40, "tenure": 5,
                   "balance": 0, "products_number": 1, "estimated_salary": 100000, "credit_card": 1,
                   "active_member": 1}
_KEYS = {"credit_score": "cs", "country": "c", "gender": "g", "age": "a", "tenure": "t", "balance": "b",
         "products_number": "p", "estimated_salary": "s", "credit_card": "cc", "active_member": "am"}
_FLAGS = ("credit_card", "active_member")
_INTS = ("credit_score", "age", "tenure", "balance", "products_number", "estimated_salary")


def widget_key(prefix, col):
    return f"{prefix}_{_KEYS[col]}"


def clamp_profile(profile):
    """Profili form sınırlarına getirir: sayılar tam sayıya yuvarlanır ve eğitim aralığına
    kırpılır. Dönüş: (yeni profil, aralık dışında kalıp kırpılan sütunlar)."""
    b, out, clipped = input_bounds(), dict(profile), []
    for c in _INTS:
        v = int(round(float(out[c])))
        lo, hi = b[c]
        if v < lo or v > hi:
            clipped.append(c)
        out[c] = int(min(max(v, lo), hi))
    for c in _FLAGS:
        out[c] = int(out[c])
    return out, clipped


def set_form(prefix, profile):
    """Form değerlerini (widget durumunu) bir profile ayarlar; geri çağırmadan çağrılır."""
    for col, v in profile.items():
        st.session_state[widget_key(prefix, col)] = bool(v) if col in _FLAGS else v


def profile_from_test_row(row):
    """test_predictions.csv satırından ham veri biçiminde profil."""
    r = row.iloc[0] if isinstance(row, pd.DataFrame) else row
    country = "Germany" if r["country_Germany"] else ("Spain" if r["country_Spain"] else "France")
    return {"credit_score": r["credit_score"], "country": country,
            "gender": "Male" if r["gender_Male"] else "Female", "age": r["age"], "tenure": r["tenure"],
            "balance": r["balance"], "products_number": r["products_number"],
            "estimated_salary": r["estimated_salary"], "credit_card": r["credit_card"],
            "active_member": r["active_member"]}


def customer_form(prefix, sliders=False, initial=None):
    """Müşteri girişi; ham veri biçiminde profil (dict) döner. Sınırlar eğitim verisinden.
    sliders=True: yaş, kredi skoru, kıdem ve ürün sayısı kaydırıcıyla (simülatör)."""
    b, ss = input_bounds(), st.session_state
    start = {**DEFAULT_PROFILE, **(initial or {})}
    for col, v in start.items():
        if widget_key(prefix, col) not in ss:
            ss[widget_key(prefix, col)] = bool(v) if col in _FLAGS else v

    def rng(col):
        lo, hi = b[col]
        return f"Eğitim verisindeki aralık: {ui.fmt_num(lo)}–{ui.fmt_num(hi)}"

    def num(container, label, col, step=1):
        lo, hi = b[col]
        k = widget_key(prefix, col)
        if sliders and col in ("credit_score", "age", "tenure", "products_number"):
            return container.slider(label, lo, hi, step=step, key=k, help=rng(col))
        return container.number_input(label, lo, hi, step=step, key=k, help=rng(col))

    c1, c2 = st.columns(2)
    row = {
        "credit_score": num(c1, "Kredi skoru", "credit_score"),
        "country": c1.selectbox("Ülke", b["country"], format_func=COUNTRY_TR.get,
                                key=widget_key(prefix, "country")),
        "gender": c1.selectbox("Cinsiyet", b["gender"], format_func=GENDER_TR.get,
                               key=widget_key(prefix, "gender")),
        "age": num(c1, "Yaş", "age"),
        "tenure": num(c1, "Müşteri süresi (yıl)", "tenure"),
        "balance": num(c2, "Bakiye", "balance", step=1000),
        "products_number": num(c2, "Ürün sayısı", "products_number"),
        "estimated_salary": num(c2, "Tahmini maaş", "estimated_salary", step=1000),
        "credit_card": int(c2.checkbox("Kredi kartı var", key=widget_key(prefix, "credit_card"))),
        "active_member": int(c2.checkbox("Aktif üye", key=widget_key(prefix, "active_member"))),
    }
    return row


def features(art, profile):
    """Ham profil → modelin girdi satırı (train.prepare_features ile aynı dönüşüm)."""
    return prepare_features(pd.DataFrame([profile]))[art["features"]]


def score(art, key, x_raw):
    """(model skoru, kalibre olasılık). Uyarı kararı skorla, gösterge olasılıkla."""
    s = float(art["models"][key].predict_proba(x_raw)[0, 1])
    return s, float(calibrate(art, key, [s])[0])


# ---------- Gösterge + uyarı satırı (tek ölçek) ----------
ALERT_CSS = """<style>
.iris-alertline { display:flex; align-items:center; justify-content:center; gap:.45rem; flex-wrap:wrap;
  margin:.65rem auto 0; padding:.5rem .8rem; border:1px solid var(--iris-border); border-radius:12px;
  background: var(--iris-soft); color: var(--iris-ink); font-size:.86rem; max-width: 360px; }
.iris-alertline .al-score { color: var(--iris-muted); font-size:.8rem; font-variant-numeric: tabular-nums; }
.iris-info { display:inline-flex; width:16px; height:16px; border-radius:50%; flex:0 0 auto;
  border:1px solid var(--iris-subtle); color: var(--iris-subtle); font-size:.64rem; font-weight:700;
  align-items:center; justify-content:center; cursor:help; font-style:normal; }
</style>"""


def alert_tip(thr, calibrated):
    return ("Gitme olasılığı kalibre edilmiştir: gerçek churn oranıyla uyumlu, yüzde olarak "
            "okunabilir. Uyarı kararı ise modelin ham skoru ile maliyet ve F2'ye göre seçilen "
            f"eşik ({ui.fmt_num(thr, 2)}) karşılaştırılarak verilir. Kalibrasyon müşterilerin "
            "sıralamasını değiştirmediği için uyarı listesi iki ölçekte de aynıdır."
            if calibrated else
            "Bu model için kalibrasyon uygulanmadı; gösterilen değer modelin ham skorudur. Uyarı "
            f"kararı ham skor ≥ eşik ({ui.fmt_num(thr, 2)}) kuralıyla verilir.")


def gauge_html(prob, s, thr, calibrated, caption="gitme olasılığı", extra=""):
    """Kalibre olasılık göstergesi + büyük sayı + ayrı uyarı satırı (model skoru ≥ eşik)."""
    in_list = s >= thr
    dec = 3 if round(s, 2) == round(thr, 2) else 2  # 0,209 gibi skorlar eşikle aynı görünmesin
    tip = alert_tip(thr, calibrated)
    g = ui.gauge_svg(prob, 1.0, bands=ui.risk_bands(), separators=False,
                     labels=[(0, "%0"), (0.5, "%50"), (1, "%100")],
                     on_dark=ui.mode() == "dark", cls="iris-gauge-sm",
                     alt=f"Gitme olasılığı {ui.fmt_pct(prob)}")
    pill = ('<span class="iris-pill warn">Evet</span>' if in_list
            else '<span class="iris-pill ok">Hayır</span>')
    return (f'{g}<div style="text-align:center">'
            f'<div class="iris-big">{ui.count(prob * 100, 1, pre="%")}</div>'
            f'<div class="iris-card-sub">{caption}</div>'
            f'<div class="iris-alertline" title="{tip}"><b>Uyarı listesinde:</b> {pill}'
            f'<span class="al-score">(model skoru {ui.fmt_num(s, dec)} {"≥" if in_list else "<"} '
            f'eşik {ui.fmt_num(thr, 2)})</span><i class="iris-info" title="{tip}">i</i></div>'
            f'{extra}</div>')


# ---------- SHAP şelalesi ----------
def shap_waterfall(art, key, x_raw, chart_key, height=430, k=9):
    """Tek müşteri için SHAP şelalesi (modelin kendi çıktı ölçeğinde). Dönüş: base, fx, unit, dec."""
    feats = art["features"]
    expl = explain_row(key, art, x_raw)
    vals = np.asarray(expl.values, dtype=float)
    order = np.argsort(-np.abs(vals))
    shown, rest = order[:k], order[k:]
    labels = [f"{FEATURE_TR[feats[i]]} = {fmt_feature(feats[i], x_raw.iloc[0, i])}" for i in shown]
    contrib = [float(v) for v in vals[shown]]
    if len(rest):
        labels.append(f"Diğer {len(rest)} değişken")
        contrib.append(float(vals[rest].sum()))
    unit = SHAP_UNIT[key]
    dec = 3 if unit == "olasılık" else 2
    base = float(expl.base_values)
    fx = base + float(vals.sum())
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
    ui.plot(fig, height=height, key=chart_key)
    return {"base": base, "fx": fx, "unit": unit, "dec": dec}
