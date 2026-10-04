"""Sunum Modu: tez savunması için adım adım rehber.

Her adım = bir soru + mevcut verilerden tek bir küçük grafik + tek cümlelik cevap + ayrıntılı
sayfaya bağlantı. Adım st.session_state["sunum_adim"] içinde tutulur (Önceki / Sonraki).
Sunum ana modeli (XGBoost) ve onun varsayılan (F2-optimal) eşiğini kullanır; kenar
çubuğundaki ayardan bağımsızdır. Tüm sayılar artifacts/ ve veri setinden okunur.
Adım değişince adımın kartı (soru, grafik, cevap, sayan sayılar) iris_ui.replay_entrance ile
sayfa açılışındaki kademeli girişi bir kez oynatır; diğer yeniden çalıştırmalarda animasyon yok.
"""
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis
import iris_ui as ui
from iris_data import FEATURE_TR, expected_churn, load_artifacts, load_data
from train import MODEL_NAMES, OUT

KEY = "xgb"  # tezin ana modeli
SHORT = {"Lojistik Regresyon": "LR", "Random Forest": "RF", "XGBoost": "XGB"}
SMOTE_FILE = OUT / "smote_karsilastirma.json"

art, df = load_artifacts(), load_data()
THR = float(art["thresholds"][KEY])
preds = art["preds"]
y, s = preds["churn"].to_numpy(), preds[f"proba_{KEY}"].to_numpy()
base_rate = float(df["churn"].mean())


@st.cache_data
def germany(data):
    g, change, _, _ = analysis.germany_comparison(data)
    return g, change


@st.cache_data
def smote_summary(mtime):
    with open(SMOTE_FILE, encoding="utf-8") as f:
        return json.load(f)


smote = smote_summary(SMOTE_FILE.stat().st_mtime) if SMOTE_FILE.exists() else None
# Maliyet sayfasıyla aynı varsayımsal birimler (FN = 5, FP = 1); karşılaştırma dosyasında kayıtlı
COST_FN = smote["cost_fn"] if smote else 5
COST_FP = smote["cost_fp"] if smote else 1


def pct(v, dec=1):
    return ui.fmt_pct(v, dec)


def tr_lower(text):
    """Türkçe küçük harf (I→ı, İ→i); 'Ülke: Almanya' gibi özel adlı etiketler aynen kalır."""
    if ":" in text:
        return text
    return text.replace("I", "ı").replace("İ", "i").lower()


def mean_vs_actual(bars, actual, key):
    """Ortalama tahmin çubukları + gerçek oran kesikli çizgisi (adım 2 ve SMOTE adımı)."""
    fig = go.Figure(go.Bar(
        x=[b[0] for b in bars], y=[b[1] * 100 for b in bars],
        marker=dict(color=[b[2] for b in bars], cornerradius=5),
        text=[pct(b[1], 2) for b in bars], textposition="outside", cliponaxis=False,
        textfont=dict(color=ui.tok("ink"), size=13),
        hovertemplate="%{x}<br>Ortalama %%{y:.2f}<extra></extra>"))
    fig.add_hline(y=actual * 100, line=dict(color=ui.tok("subtle"), width=1.5, dash="dash"))
    # Etiket çizim alanının sağında (çubuk etiketleriyle çakışmasın)
    fig.add_annotation(x=1, xref="paper", y=actual * 100, xanchor="left", xshift=6, showarrow=False,
                       align="left", text=f"Gerçek oran<br><b>{pct(actual, 2)}</b>",
                       font=dict(color=ui.tok("muted"), size=12))
    fig.update_yaxes(tickprefix="%", range=[0, max(b[1] for b in bars) * 100 * 1.2])
    fig.update_layout(showlegend=False, bargap=0.45, margin=dict(l=4, r=92, t=24, b=4))
    ui.plot(fig, height=300, key=key)


# ---------- Adımlar ----------
def step_alert():
    alert = s >= THR
    tp, fn, fp = int((alert & (y == 1)).sum()), int((~alert & (y == 1)).sum()), int((alert & (y == 0)).sum())
    n_alert, n_churn = int(alert.sum()), int((y == 1).sum())
    rows = [("Gidenler",[("Listede (yakalandı)", tp, ui.TURQ), ("Kaçırıldı", fn, ui.CORAL)]),
            ("Uyarı listesi", [("Gerçekten gitti", tp, ui.TURQ), ("Yanlış alarm", fp, ui.tok("neutral"))])]
    fig = go.Figure()
    for label, parts in rows:
        total = sum(p[1] for p in parts)
        for j, (name, v, color) in enumerate(parts):
            # Küçük son parça (ör. kaçırılanlar) etiketini çubuğun dışına yazar
            outside = j == len(parts) - 1 and v < 0.25 * total
            fig.add_trace(go.Bar(
                y=[f"{label} ({ui.fmt_num(total)})"], x=[v], orientation="h",
                name=name, marker=dict(color=color, cornerradius=3), text=[f"{name}: {ui.fmt_num(v)}"],
                textposition="outside" if outside else "inside", insidetextanchor="middle",
                textangle=0, cliponaxis=False,
                textfont=dict(color=ui.tok("ink") if outside or color == ui.tok("neutral") else "#FFFFFF",
                              size=12),
                showlegend=False, hovertemplate=f"{name}: %{{x:,}}<extra></extra>"))
    # Sol boşluk sabit: yazı tipi geç yüklenince otomatik kenar boşluğu etiketleri kırpıyordu
    fig.update_layout(barmode="stack", bargap=0.35, margin=dict(l=140, r=60, t=12, b=4))
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=False, ticklabelstandoff=6)
    fig.update_xaxes(showgrid=True, gridcolor=ui.tok("grid"), title="Müşteri sayısı (test seti)")
    ui.plot(fig, height=260, key="s1")
    return dict(
        answer=f"XGBoost, skoru {ui.fmt_num(THR, 2)} ve üstü olan {ui.fmt_num(n_alert)} müşteriyi uyarı "
               f"listesine alıyor; gerçekten giden {ui.fmt_num(n_churn)} müşteriden {ui.fmt_num(tp)} "
               f"tanesi bu listede.",
        facts=[("Recall", pct(tp / n_churn)), ("Precision", pct(tp / max(n_alert, 1))),
               ("Uyarı listesi", f"{ui.fmt_num(n_alert)} / {ui.fmt_num(len(s))}")])


def step_calibration():
    exp = expected_churn(art, KEY, base_rate)
    n = exp["n"]
    bars = [("Ham skor ortalaması", exp["raw_mean"], ui.CORAL)]
    if exp["calibrated"]:
        bars.append(("Kalibre olasılık ortalaması", exp["value"], ui.TURQ))
    mean_vs_actual(bars, exp["actual"], "s2")
    return dict(
        answer=f"Kalibrasyondan sonra model test setinde {pct(exp['value'])} churn bekliyor (yaklaşık "
               f"{ui.fmt_num(round(exp['value'] * n))} müşteri); gerçekte giden {pct(exp['actual'], 2)} "
               f"({ui.fmt_num(round(exp['actual'] * n))} müşteri).",
        facts=[("Ham ortalama", pct(exp["raw_mean"])), ("Kalibre", pct(exp["value"])),
               ("Gerçek", pct(exp["actual"], 2))])


def step_mcnemar():
    mc = art["mcnemar"]
    fig = go.Figure()
    wins, sig_n = {}, 0
    for _, r in mc.iterrows():
        a, b = r["Karşılaştırma"].split(" vs ")
        n1, n2 = int(r["Yalnız 1. doğru"]), int(r["Yalnız 2. doğru"])
        winner = a if n1 >= n2 else b
        sig = r["p (Holm)"] < 0.05
        sig_n += sig
        if sig:
            wins[winner] = wins.get(winner, 0) + 1
        label = f"{SHORT.get(a, a)} – {SHORT.get(b, b)}<br>Holm p {ui.fmt_p(r['p (Holm)'])}"
        for m, v, side in [(a, n1, 0), (b, n2, 1)]:
            fig.add_trace(go.Bar(
                x=[label], y=[v], name=m, offsetgroup=str(side), showlegend=False,
                marker=dict(color=ui.TURQ if m == winner else ui.tok("neutral"), cornerradius=4),
                text=[f"{SHORT.get(m, m)} {ui.fmt_num(v)}"], textposition="outside", cliponaxis=False,
                textfont=dict(color=ui.tok("ink"), size=11),
                hovertemplate=f"{m} tek başına doğru: %{{y}}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.1, margin=dict(l=4, r=4, t=24, b=4))
    fig.update_yaxes(title="Tek başına doğru (müşteri)",
                     range=[0, float(mc[["Yalnız 1. doğru", "Yalnız 2. doğru"]].to_numpy().max()) * 1.18])
    ui.plot(fig, height=300, key="s3")
    best = max(wins, key=wins.get) if wins else None
    total = len(mc)
    if sig_n == total and best and wins[best] == total - 1:
        answer = (f"Evet. Üç ikili karşılaştırmanın üçünde de fark anlamlı (McNemar, Holm düzeltmeli "
                  f"p < 0,05); {best} diğer iki modelden daha az hata yapıyor.")
    else:
        answer = (f"{ui.fmt_num(total)} karşılaştırmanın {ui.fmt_num(sig_n)} tanesinde fark anlamlı "
                  "(McNemar, Holm düzeltmeli p < 0,05).")
    return dict(answer=answer,
                facts=[("Test", "McNemar"), ("Düzeltme", "Holm"), ("Anlamlı", f"{sig_n} / {total}")])


def step_cost():
    val = art["val_preds"]
    opt, curve = analysis.cost_optimal_threshold(val["churn"], val[f"proba_{KEY}"], COST_FN, COST_FP)
    r = analysis.evaluate_at(y, s, THR, COST_FN, COST_FP)
    vmin = float(curve["Maliyet"].min())
    fig = go.Figure(go.Scatter(x=curve["Eşik"], y=curve["Maliyet"], mode="lines",
                               line=dict(color=ui.tok("ink"), width=2.4), name="Validation",
                               hovertemplate="Eşik %{x:.2f}<br>Maliyet %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[opt], y=[vmin], mode="markers", name="En düşük maliyet",
                             marker=dict(color=ui.TURQ, size=20, symbol="circle-open", line=dict(width=3)),
                             hovertemplate="En düşük maliyet: eşik %{x:.2f}<extra></extra>"))
    fig.add_vline(x=THR, line=dict(color=ui.CORAL, width=1.2, dash="dash"))
    fig.add_annotation(x=THR, y=1, yref="paper", yanchor="top", xanchor="left", xshift=6, showarrow=False,
                       text=f"Kullanılan eşik {ui.fmt_num(THR, 2)}", bgcolor=ui.tok("surface"),
                       font=dict(color=ui.CORAL, size=11))
    fig.update_xaxes(title="Karar eşiği (model skoru)", range=[0, 1], dtick=0.2, tickformat=".1f",
                     ticklabelstandoff=10,
                     showgrid=True, gridcolor=ui.tok("grid"))
    fig.update_yaxes(title="Toplam maliyet (validation)", rangemode="tozero", tickformat=",.0f",
                     ticklabelstandoff=8)
    fig.update_layout(showlegend=False)
    ui.plot(fig, height=300, key="s4")
    ratio = ui.fmt_num(COST_FN / COST_FP, 0)
    if abs(opt - THR) < 1e-9:
        answer = (f"Çünkü kaçırılan müşteri yanlış alarmdan {ratio} kat pahalı sayıldığında, toplam maliyeti "
                  f"en aza indiren eşik de {ui.fmt_num(THR, 2)}; F2 ile seçilen eşikle aynı.")
    else:
        answer = (f"Eşik {ui.fmt_num(THR, 2)} F2 ile seçildi; kaçırılan müşteri {ratio} kat pahalı "
                  f"sayıldığında maliyeti en aza indiren eşik {ui.fmt_num(opt, 2)}.")
    return dict(answer=answer,
                facts=[("FN : FP", f"{ratio} : 1"), ("Test maliyeti", ui.fmt_num(r["Toplam maliyet"])),
                       ("FN · FP", f"{ui.fmt_num(r['FN'])} · {ui.fmt_num(r['FP'])}")])


def step_shap():
    vals = np.asarray(art["shap"][KEY]["values"])
    imp = pd.Series(np.abs(vals).mean(axis=0), index=art["features"]).sort_values(ascending=False)
    top = imp.iloc[:6][::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=[FEATURE_TR[f] for f in top.index], orientation="h",
        marker=dict(color=[ui.TURQ if i >= len(top) - 3 else ui.tok("neutral") for i in range(len(top))],
                    cornerradius=4),
        text=[ui.fmt_num(v, 3) for v in top.values], textposition="outside", cliponaxis=False,
        textfont=dict(color=ui.tok("ink"), size=11),
        hovertemplate="%{y}<br>Ortalama |SHAP|: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ortalama |SHAP| (log-odds)", showgrid=True, gridcolor=ui.tok("grid"))
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=4, r=40, t=8, b=4), bargap=0.3)
    ui.plot(fig, height=300, key="s5")
    names = [tr_lower(FEATURE_TR[f]) for f in imp.index[:3]]
    return dict(
        answer=f"Kararı en çok {names[0]}, {names[1]} ve {names[2]} belirliyor; tek bir müşterinin "
               "gerekçesi Müşteri Risk Skoru sayfasında gösteriliyor.",
        facts=[("Yöntem", "SHAP"), ("Model", "XGBoost"), ("Müşteri", ui.fmt_num(len(vals)))],
        extra=("sayfalar/risk_skoru.py", "Tek müşteri gerekçesi", ":material/person_search:"))


def step_germany():
    g, change = germany(df)
    names = ["Ham (yalnız ülke)", "Kontrollü"]
    fig = go.Figure(go.Scatter(
        x=g["OR"], y=names, mode="markers+text",
        marker=dict(color=[ui.tok("neutral"), ui.CORAL], size=14, symbol="square",
                    line=dict(color=ui.tok("surface"), width=1.5)),
        error_x=dict(type="data", symmetric=False, array=g["GA üst"] - g["OR"],
                     arrayminus=g["OR"] - g["GA alt"], color=ui.tok("ink"), thickness=2, width=6),
        text=[f"OR {ui.fmt_num(o, 2)}" for o in g["OR"]], textposition="top center",
        textfont=dict(color=ui.tok("ink"), size=12),
        customdata=np.c_[g["GA alt"], g["GA üst"]],
        hovertemplate="%{y}<br>OR %{x:.2f} [%{customdata[0]:.2f}; %{customdata[1]:.2f}]<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=ui.tok("subtle"), width=1, dash="dash"))
    fig.add_annotation(x=0, y=0, xref="x", yref="paper", text="OR = 1 (etki yok)", showarrow=False,
                       xanchor="left", yanchor="bottom", xshift=4, font=dict(color=ui.tok("muted"), size=11))
    fig.update_xaxes(type="log", title="Almanya odds ratio'su (ref. Fransa, log ölçek)", showgrid=True,
                     gridcolor=ui.tok("grid"), range=[np.log10(0.8), np.log10(max(g["GA üst"].max() * 1.25, 3))])
    fig.update_yaxes(showgrid=False, range=[1.6, -0.7])
    fig.update_layout(showlegend=False)
    ui.plot(fig, height=260, key="s6")
    or_a, or_b = (float(v) for v in g["OR"])
    lo_b, hi_b = float(g["GA alt"].iloc[1]), float(g["GA üst"].iloc[1])
    if lo_b > 1:
        answer = (f"Evet. Bakiye, yaş, cinsiyet, aktiflik ve ürün sayısı sabit tutulunca da Almanya'nın odds ratio'su "
                  f"{ui.fmt_num(or_a, 2)} → {ui.fmt_num(or_b, 2)}; güven aralığı 1'in üstünde, yani etki "
                  "bakiye yapısıyla açıklanmıyor.")
    else:
        answer = (f"Kontrollerle odds ratio {ui.fmt_num(or_a, 2)} → {ui.fmt_num(or_b, 2)}; güven aralığı "
                  "1'i içeriyor, ek bir Almanya etkisi gösterilemiyor.")
    return dict(answer=answer,
                facts=[("OR ham", ui.fmt_num(or_a, 2)), ("OR kontrollü", ui.fmt_num(or_b, 2)),
                       ("%95 GA", f"{ui.fmt_num(lo_b, 2)}; {ui.fmt_num(hi_b, 2)}")])


def step_smote():
    x, mc = smote["results"][KEY], smote["mcnemar_xgb"]
    bars = [("SMOTE'lu (ham)", x["smote"]["mean_pred"], ui.CORAL),
            ("SMOTE'suz", x["no_smote"]["mean_pred"], ui.tok("france"))]
    cal = art.get("calibration")
    if cal and cal["summary"]["models"][KEY]["applied"]:
        bars.append(("SMOTE'lu + kalibrasyon", cal["summary"]["models"][KEY]["mean_pred_test_cal"], ui.TURQ))
    mean_vs_actual(bars, x["smote"]["actual_rate"], "s7")
    sig = mc["p"] < 0.05
    answer = (("Hayır. " if not sig else "") +
              f"SMOTE'lu ve SMOTE'suz XGBoost arasında yakalama farkı "
              f"{'anlamlı' if sig else 'anlamlı değil'} (McNemar p = {ui.fmt_num(mc['p'], 2)}); SMOTE "
              "olasılıkları şişirdi, kalibrasyon bunu düzeltti.")
    return dict(answer=answer,
                facts=[("Recall SMOTE'lu", pct(x["smote"]["f2_threshold"]["recall"])),
                       ("Recall SMOTE'suz", pct(x["no_smote"]["f2_threshold"]["recall"])),
                       ("McNemar p", ui.fmt_num(mc["p"], 2))])


STEPS = [
    dict(q="Hangi müşteriler gidecek?", short="Kim gider?", fn=step_alert,
         link=("sayfalar/risk_skoru.py", "Müşteri Risk Skoru", ":material/person_search:")),
    dict(q="Kaç müşteri gider?", short="Kaç kişi?", fn=step_calibration,
         link=("sayfalar/model_karsilastirma.py", "Model Karşılaştırma · Kalibrasyon", ":material/leaderboard:")),
    dict(q="Modeller gerçekten farklı mı?", short="Modeller", fn=step_mcnemar,
         link=("sayfalar/model_karsilastirma.py", "Model Karşılaştırma · McNemar", ":material/leaderboard:")),
    dict(q=f"Eşik neden {ui.fmt_num(THR, 2)}?", short="Eşik", fn=step_cost,
         link=("sayfalar/maliyet.py", "Maliyet ve Eşik", ":material/payments:")),
    dict(q="Model neden bu kararı verdi?", short="Gerekçe", fn=step_shap,
         link=("sayfalar/aciklanabilirlik.py", "Açıklanabilirlik (SHAP)", ":material/psychology:")),
    dict(q="Almanya etkisi gerçek mi?", short="Almanya", fn=step_germany,
         link=("sayfalar/confounding.py", "Confounding", ":material/account_tree:")),
]
if smote:
    STEPS.append(dict(q="SMOTE sonucu değiştirdi mi?", short="SMOTE", fn=step_smote,
                      link=("sayfalar/smote_duyarlilik.py", "Duyarlılık: SMOTE", ":material/tune:")))
N = len(STEPS)

ss = st.session_state
ss["sunum_adim"] = int(min(max(ss.get("sunum_adim", 0), 0), N - 1))
# İsteğe bağlı doğrudan bağlantı: ?adim=4 sayfayı 4. adımda açar (yalnız ilk okumada)
_qp = st.query_params.get("adim")
if _qp and _qp.isdigit() and ss.get("_sunum_qp") != _qp:
    ss["_sunum_qp"] = _qp
    ss["sunum_adim"] = int(min(max(int(_qp) - 1, 0), N - 1))


def go_to(i):
    new = int(min(max(i, 0), N - 1))
    if new != ss["sunum_adim"]:
        # Adım değişince giriş animasyonu yalnız yeni adımın kartında bir kez oynar (iris_ui)
        ui.replay_entrance(f".st-key-card-sunum-{new}")
    ss["sunum_adim"] = new


_NUM = re.compile(r"(%?)(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?")


def countable(text):
    """Yalnız sayıdan oluşan değer ('%20,4', '0,02', '2.037') sayan sayı olur (son metin aynı);
    'McNemar', '5 : 1' gibi değerler olduğu gibi kalır."""
    m = _NUM.fullmatch(text)
    if not m:
        return text
    pre, whole, dec = m.groups()
    value = float(whole.replace(".", "") + ("." + dec if dec else ""))
    out = ui.count(value, len(dec or ""), pre=pre)
    return out if re.sub(r"<[^>]+>", "", out) == text else text


ui.html("""<style>
.iris-steps { display:flex; flex-wrap:wrap; gap:.45rem; margin:.7rem 0 .9rem; }
.iris-step { display:inline-flex; align-items:center; gap:.45rem; padding:.28rem .75rem .28rem .3rem;
  border-radius:999px; border:1px solid var(--iris-border); background: var(--iris-surface);
  color: var(--iris-muted); font-size:.8rem; transition: all .25s ease; }
.iris-step b { width:22px; height:22px; border-radius:50%; display:inline-flex; align-items:center;
  justify-content:center; background: var(--iris-soft); color: var(--iris-muted); font-size:.74rem; }
.iris-step.done b { background: rgba(20,184,166,.18); color: var(--iris-accent-text); }
.iris-step.cur { border-color: var(--iris-turq); color: var(--iris-ink); font-weight:650;
  box-shadow: var(--iris-shadow); }
.iris-step.cur b { background: var(--iris-turq); color:#FFFFFF; }
.iris-q-eyebrow { font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700;
  color: var(--iris-accent-text); }
.iris-q { font-size:1.65rem; font-weight:750; color: var(--iris-ink); line-height:1.2; margin:.15rem 0 .4rem;
  letter-spacing:-.01em; }
.iris-answer { border-left:4px solid var(--iris-turq); padding:.2rem 0 .2rem .9rem; color: var(--iris-ink);
  font-size:1.08rem; line-height:1.5; font-weight:550; margin:.2rem 0 .9rem; }
.iris-answer-lab { font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700;
  color: var(--iris-subtle); margin-bottom:.3rem; }
[class*="st-key-card-sunum"] [data-testid="stPageLink"] a { border:1px solid var(--iris-border);
  border-radius:999px; padding:.2rem .9rem .2rem .5rem; background: var(--iris-soft); }
[class*="st-key-card-sunum"] [data-testid="stPageLink"] a:hover { border-color: var(--iris-turq); }
</style>""")

ui.page_header("Sunum", "Sunum Modu",
               "Tez savunması için adım adım: her adımda bir soru, bir grafik ve tek cümlelik cevap.",
               chip=f"{MODEL_NAMES[KEY]} · eşik {ui.fmt_num(THR, 2)}")

i = ss["sunum_adim"]
chips = "".join(
    f'<span class="iris-step {"cur" if j == i else "done" if j < i else ""}"><b>{j + 1}</b>'
    f'{st_["short"]}</span>' for j, st_ in enumerate(STEPS))
ui.html(f'<div class="iris-steps">{chips}</div>')

step = STEPS[i]
with st.container(key=f"card-sunum-{i}"):
    ui.html(f'<div class="iris-q-eyebrow">Soru {i + 1} / {N}</div><div class="iris-q">{step["q"]}</div>')
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        out = step["fn"]()
    with c2:
        ui.html(f'<div class="iris-answer-lab">Kısa cevap</div><div class="iris-answer">{out["answer"]}</div>'
                + ui.facts([(k, countable(v)) for k, v in out["facts"]], cols=3), js=True)
        page, label, icon = step["link"]
        st.page_link(page, label=f"Ayrıntı: {label}", icon=icon)
        if out.get("extra"):
            p2, l2, i2 = out["extra"]
            st.page_link(p2, label=f"Ayrıntı: {l2}", icon=i2)

b1, b2, b3 = st.columns([1, 2, 1], vertical_alignment="center")
b1.button("Önceki", key="sunum_prev", icon=":material/arrow_back:", on_click=go_to, args=(i - 1,),
          disabled=i == 0, width="stretch")
b2.html(f'<div class="iris-card-sub" style="text-align:center">Adım <b>{i + 1}</b> / {N} · sunum ana modeli '
        f'({MODEL_NAMES[KEY]}) ve varsayılan eşiği kullanır; kenar çubuğu ayarından bağımsızdır.</div>')
if i < N - 1:
    b3.button("Sonraki", key="sunum_next", icon=":material/arrow_forward:", on_click=go_to, args=(i + 1,),
              type="primary", width="stretch")
else:
    b3.button("Başa dön", key="sunum_next", icon=":material/replay:", on_click=go_to, args=(0,),
              width="stretch")
