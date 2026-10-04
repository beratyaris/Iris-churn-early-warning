"""Duyarlılık Analizi: SMOTE'un etkisi (SMOTE-NC'li ana modeller vs SMOTE'suz eğitim).

Sayılar smote_karsilastirma.py'nin ürettiği artifacts/smote_karsilastirma.json'dan, kalibre
ortalama ise artifacts/calibration.json'dan okunur. Burada hiçbir model yeniden eğitilmez
ya da yeniden hesaplanmaz. Ana model SMOTE-NC'li kalır; bu sayfa yalnız bir duyarlılık
analizidir.
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import iris_ui as ui
from iris_data import load_artifacts
from train import MODEL_NAMES, OUT

SMOTE_FILE = OUT / "smote_karsilastirma.json"
MODEL_COLORS = {"lr": ui.tok("subtle"), "rf": ui.tok("ink"), "xgb": ui.TURQ}


@st.cache_data
def load_smote(mtime):
    with open(SMOTE_FILE, encoding="utf-8") as f:
        return json.load(f)


ui.page_header("Analiz", "Duyarlılık Analizi: SMOTE'un Etkisi",
               "Aynı modeller SMOTE-NC ile ve SMOTE'suz eğitildiğinde sonuçlar ne kadar değişiyor? "
               "Test seti (2.000 müşteri).",
               chip="Kaynak: smote_karsilastirma.py")

if not SMOTE_FILE.exists():
    st.info("Karşılaştırma dosyası yok. Önce `.\\.venv\\Scripts\\python.exe smote_karsilastirma.py` "
            "çalıştırın.")
    st.stop()

sm = load_smote(SMOTE_FILE.stat().st_mtime)
res, mc = sm["results"], sm["mcnemar_xgb"]
art = load_artifacts()
cal = art.get("calibration")
xgb = res["xgb"]
actual = float(xgb["smote"]["actual_rate"])
cal_mean = (float(cal["summary"]["models"]["xgb"]["mean_pred_test_cal"])
            if cal and cal["summary"]["models"]["xgb"]["applied"] else None)
p_txt = ui.fmt_num(mc["p"], 2)

# ---------- Bulgu ----------
fix = ("ana modelde bu kalibrasyonla düzeltildi." if cal_mean is not None else
       "kalibrasyon dosyası olmadığı için bu fark burada düzeltilmiş olarak gösterilemiyor.")
ui.html(f'<div class="iris-callout" style="--tone:{ui.TURQ}"><div class="c-title">Bulgu</div>'
        f"SMOTE gidenleri yakalama başarısını anlamlı şekilde değiştirmedi "
        f"(McNemar p = {p_txt}); asıl etkisi olasılıkları şişirmek oldu; {fix}</div>")

# ---------- Tablo ----------
rows = []
for k, name in MODEL_NAMES.items():
    for variant, label in [("smote", "SMOTE-NC (ana)"), ("no_smote", "SMOTE'suz")]:
        r = res[k][variant]
        f2 = r["f2_threshold"]
        rows.append({
            "Model": f'<span class="iris-dot" style="background:{MODEL_COLORS[k]}"></span>{name}',
            "Eğitim": label, "Eşik": f2["threshold"], "Recall": f2["recall"], "PR-AUC": r["pr_auc"],
            "Maliyet": f2["cost"], "Ort. olasılık": r["mean_pred"], "Brier": r["brier"]})
tab = pd.DataFrame(rows)
f3 = lambda v: ui.fmt_num(v, 3)  # noqa: E731
with st.container(key="card-smote-table"):
    ui.card_title("SMOTE-NC'li ve SMOTE'suz eğitim (test seti)",
                  "Eşik: her varyant için validation'da F2'yi en yükselten eşik. Maliyet: o eşikte "
                  f"FN = {sm['cost_fn']}, FP = {sm['cost_fp']} ile toplam (birim). Ort. olasılık ve "
                  f"Brier ham (kalibrasyonsuz) çıktıdan; gerçek churn oranı <b>{ui.fmt_pct(actual, 2)}</b>.")
    ui.html(ui.table_html(
        tab, fmt={"Eşik": lambda v: ui.fmt_num(v, 2), "Recall": f3, "PR-AUC": f3,
                  "Maliyet": ui.fmt_num, "Ort. olasılık": lambda v: ui.fmt_pct(v, 2),
                  "Brier": lambda v: ui.fmt_num(v, 4)},
        num=("Eşik", "Recall", "PR-AUC", "Maliyet", "Ort. olasılık", "Brier"),
        sel_rows=set(tab.index[tab["Eğitim"] == "SMOTE-NC (ana)"]), raw=("Model",)))
    st.caption("Vurgulu satırlar uygulamadaki ana modellerdir. Recall, PR-AUC ve maliyet iki varyantta "
               "birbirine yakın; belirgin fark ortalama olasılıkta (SMOTE'lu modeller gerçek orandan "
               "yüksek tahmin ediyor).")

# ---------- Grafik + McNemar ----------
c1, c2 = st.columns([1.45, 1], gap="medium")
with c1, st.container(key="card-smote-mean"):
    ui.card_title("Ortalama tahmin vs gerçek oran (XGBoost)",
                  "Test setindeki 2.000 müşteri için tahmin edilen olasılıkların ortalaması. Kesikli "
                  "çizgi gerçekleşen churn oranı.")
    bars = [("SMOTE'lu (ham)", float(xgb["smote"]["mean_pred"]), ui.CORAL),
            ("SMOTE'suz", float(xgb["no_smote"]["mean_pred"]), ui.tok("france"))]
    if cal_mean is not None:
        bars.append(("SMOTE'lu + kalibrasyon", cal_mean, ui.TURQ))
    fig = go.Figure(go.Bar(
        x=[b[0] for b in bars], y=[b[1] * 100 for b in bars],
        marker=dict(color=[b[2] for b in bars], cornerradius=5),
        text=[ui.fmt_pct(b[1], 2) for b in bars], textposition="outside", cliponaxis=False,
        textfont=dict(color=ui.tok("ink"), size=13),
        hovertemplate="%{x}<br>Ortalama tahmin %%{y:.2f}<extra></extra>"))
    fig.add_hline(y=actual * 100, line=dict(color=ui.tok("subtle"), width=1.5, dash="dash"))
    # Etiket çizim alanının sağında (çubuk etiketleriyle çakışmasın)
    fig.add_annotation(x=1, xref="paper", y=actual * 100, xanchor="left", xshift=6, showarrow=False,
                       align="left", text=f"Gerçek oran<br><b>{ui.fmt_pct(actual, 2)}</b>",
                       font=dict(color=ui.tok("muted"), size=12))
    fig.update_yaxes(tickprefix="%", rangemode="tozero",
                     range=[0, max(b[1] for b in bars) * 100 * 1.18])
    fig.update_layout(showlegend=False, bargap=0.45, margin=dict(l=4, r=92, t=24, b=4))
    ui.plot(fig, height=320, key="smote-mean")

with c2, st.container(key="card-smote-mcnemar"):
    ui.card_title("McNemar testi (XGBoost)",
                  "SMOTE'lu ve SMOTE'suz model, kendi F2 eşiklerinde aynı test müşterileri üzerinde "
                  "eşleştirilerek karşılaştırıldı.")
    sig = mc["p"] < 0.05
    ui.html(ui.facts([
        ("Yalnız SMOTE'lu doğru", ui.fmt_num(mc["only_smote_correct"])),
        ("Yalnız SMOTE'suz doğru", ui.fmt_num(mc["only_no_smote_correct"])),
        ("χ² istatistiği", ui.fmt_num(mc["statistic"], 2)),
        ("p-değeri", ui.fmt_p(mc["p"])),
        ("Recall (SMOTE'lu)", ui.fmt_pct(xgb["smote"]["f2_threshold"]["recall"])),
        ("Recall (SMOTE'suz)", ui.fmt_pct(xgb["no_smote"]["f2_threshold"]["recall"])),
    ]))
    ui.html('<div class="iris-status" style="justify-content:flex-start;margin-top:.7rem">'
            + (f'<span class="iris-pill warn">Anlamlı fark (p {ui.fmt_p(mc["p"])})</span>' if sig else
               f'<span class="iris-pill flat">Anlamlı fark yok (p = {p_txt})</span>') + "</div>")
    n_disc = int(mc["only_smote_correct"]) + int(mc["only_no_smote_correct"])
    st.caption(f"İki model 2.000 müşterinin yalnız {ui.fmt_num(n_disc)} tanesinde farklı karar veriyor; "
               "bu ayrışma iki yöne yaklaşık eşit dağılıyor.")

# ---------- Not ----------
ui.html(ui.callout(
    "Not", ["Ana model <b>SMOTE-NC'li kalmıştır</b>; bu sayfa, yöntem seçiminin sonuçları "
            "değiştirmediğini gösteren bir <b>duyarlılık analizidir</b>.",
            "SMOTE'un şişirdiği olasılıklar ana modelde validation setinde eğitilen kalibratörle "
            "düzeltilir; uyarı listesi ham skor ve eşikle verildiği için kalibrasyondan etkilenmez."],
    tone=ui.AMBER,
    note="Modeller bu sayfada yeniden eğitilmez; sayılar smote_karsilastirma.py çıktısından okunur."
         + (" Karşılaştırmadaki SMOTE'lu varyantın uygulamadaki modellerle aynı tahminleri ve "
            "eşikleri ürettiği doğrulanmıştır." if all(v.get("reproduced") for v in
                                                    sm.get("sanity_check", {}).values()) and
            sm.get("sanity_check") else "")))
