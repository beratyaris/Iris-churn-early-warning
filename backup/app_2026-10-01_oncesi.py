r"""Banka Müşteri Terki (Churn) Erken Uyarı Sistemi - Streamlit arayüzü.

Önce modeller eğitilmeli:  .\.venv\Scripts\python.exe train.py
Sonra:                     .\.venv\Scripts\streamlit.exe run app.py
Bu dosya eğitim yapmaz; yalnızca artifacts/ klasöründeki çıktıları okur.
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")  # grafikleri pencere açmadan çiz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

import analysis
from train import DATA, MODEL_NAMES, OUT, prepare_features

# Model renkleri (her model her grafikte aynı renkte)
COLORS = {"lr": "#2a78d6", "rf": "#eb6834", "xgb": "#1baf7a"}
BAR_COLOR = "#2a78d6"
# Ülke renkleri (confounding sekmesi; model renklerinden ayrı)
COUNTRY_COLORS = {"France": "#4a5fc1", "Germany": "#c2410c", "Spain": "#7c8a99"}

# Maliyet paneli varsayılanları - VARSAYIMSAL birimler, gerçek banka verisi değildir.
# Gerekçe: kaçırılan bir churn müşterisi (FN) gelecekteki gelirin tamamen kaybıdır;
# yanlış alarm (FP) yalnızca bir elde tutma kampanyasının maliyetidir. FN'nin FP'den
# birkaç kat pahalı olduğu varsayımıyla 5:1 seçildi; kullanıcı arayüzden değiştirebilir.
DEFAULT_COST_FN = 5.0
DEFAULT_COST_FP = 1.0


# ---------- Yükleme ----------
@st.cache_data
def load_data():
    return pd.read_csv(DATA)


@st.cache_resource
def load_artifacts():
    with open(OUT / "thresholds.json", encoding="utf-8") as f:
        thr = json.load(f)
    return {
        "models": joblib.load(OUT / "models.joblib"),
        "shap": joblib.load(OUT / "shap_test.joblib"),
        "background": joblib.load(OUT / "background.joblib"),
        "metrics": pd.read_csv(OUT / "metrics.csv"),
        "mcnemar": pd.read_csv(OUT / "mcnemar.csv"),
        "preds": pd.read_csv(OUT / "test_predictions.csv"),
        "val_preds": pd.read_csv(OUT / "val_predictions.csv"),
        "features": thr["feature_names"],
        "thresholds": {k: thr[k]["threshold"] for k in MODEL_NAMES},
    }


@st.cache_resource
def get_explainer(key, _pipe, _background):
    """Ağaç modelleri için TreeExplainer, Lojistik Regresyon için LinearExplainer."""
    model = _pipe.named_steps["model"]
    if key == "lr":
        bg = _pipe.named_steps["scaler"].transform(_background)
        return shap.LinearExplainer(model, shap.maskers.Independent(bg, max_samples=len(bg)))
    return shap.TreeExplainer(model)


def explain_row(key, art, x_raw):
    """Tek müşteri için SHAP açıklaması. Model ölçeklenmiş veriyle çalışır,
    grafikte ise müşterinin ham değerleri gösterilir."""
    pipe = art["models"][key]
    explainer = get_explainer(key, pipe, art["background"])
    x_scaled = pipe.named_steps["scaler"].transform(x_raw)
    sv = explainer(x_scaled)
    values, base = sv.values[0], sv.base_values[0]
    if values.ndim == 2:  # Random Forest: sınıf 1 (churn) sütunu
        values, base = values[:, 1], base[1]
    return shap.Explanation(values=values, base_values=float(base),
                            data=x_raw.iloc[0].to_numpy(), feature_names=list(x_raw.columns))


def churn_rate_chart(df, col, title):
    rate = df.groupby(col, observed=True)["churn"].mean().mul(100).rename("Churn oranı (%)")
    st.markdown(f"**{title}**")
    st.bar_chart(rate, color=BAR_COLOR, y_label="Churn oranı (%)", x_label="")


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="#e1e0d9", linewidth=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


# ---------- Sekme 1: Veri Özeti ----------
def tab_veri(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Müşteri sayısı", f"{len(df):,}".replace(",", "."))
    c2.metric("Churn oranı", f"%{df['churn'].mean() * 100:.2f}")
    c3.metric("Sıfır bakiyeli müşteri", f"%{(df['balance'] == 0).mean() * 100:.1f}")
    c4.metric("Eksik değer", int(df.isna().sum().sum()))

    d = df.copy()
    d["Yaş grubu"] = pd.cut(d["age"], bins=[17, 29, 39, 49, 59, 100],
                            labels=["18-29", "30-39", "40-49", "50-59", "60+"])
    d["Bakiye"] = np.where(d["balance"] > 0, "Bakiye > 0", "Bakiye = 0")
    d["Ülke"], d["Cinsiyet"] = d["country"], d["gender"].map({"Male": "Erkek", "Female": "Kadın"})

    left, right = st.columns(2)
    with left:
        churn_rate_chart(d, "Ülke", "Ülkeye göre churn oranı")
        churn_rate_chart(d, "Yaş grubu", "Yaş grubuna göre churn oranı")
    with right:
        churn_rate_chart(d, "Cinsiyet", "Cinsiyete göre churn oranı")
        churn_rate_chart(d, "Bakiye", "Bakiye durumuna göre churn oranı")


# ---------- Sekme 2: Model Karşılaştırma ----------
def tab_karsilastirma(art):
    st.markdown("**Test seti metrikleri** (eşik: 0.50 ve validation setinde F2'yi "
                "maksimize eden eşik). Ana metrikler Recall ve F2; Accuracy yalnızca bilgi için.")
    st.dataframe(art["metrics"].round(3), hide_index=True, width="stretch")

    preds, y = art["preds"], art["preds"]["churn"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, name in MODEL_NAMES.items():
        p = preds[f"proba_{key}"]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax1.plot(fpr, tpr, color=COLORS[key], lw=2, label=name)
        ax2.plot(rec, prec, color=COLORS[key], lw=2, label=name)
    ax1.plot([0, 1], [0, 1], color="#898781", lw=1, ls="--", label="Rastgele")
    ax2.axhline(y.mean(), color="#898781", lw=1, ls="--", label="Rastgele (churn oranı)")
    style_axes(ax1, "ROC eğrisi", "Yanlış pozitif oranı", "Doğru pozitif oranı (Recall)")
    style_axes(ax2, "Precision-Recall eğrisi", "Recall", "Precision")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**McNemar testi** (test seti, optimize eşikle tahminler; "
                "p-değerleri Holm yöntemiyle çoklu karşılaştırma için düzeltildi)")
    mc = art["mcnemar"].copy()
    for col in ["p", "p (Holm)"]:
        mc[col] = mc[col].map(lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}")
    mc["İstatistik"] = mc["İstatistik"].round(2)
    st.dataframe(mc, hide_index=True, width="stretch")
    st.caption("Not: Modeller farklı eşiklerde çalıştığı için McNemar, her modelin kendi "
               "optimize eşiğindeki hata oranlarını karşılaştırır.")


# ---------- Sekme 3: Müşteri Risk Skoru ----------
def manual_input_form():
    """Elle müşteri bilgisi girişi; ham veri formatında tek satırlık tablo döner."""
    c1, c2, c3 = st.columns(3)
    row = {
        "credit_score": c1.number_input("Kredi skoru", 300, 900, 650),
        "country": c1.selectbox("Ülke", ["France", "Germany", "Spain"]),
        "gender": c1.selectbox("Cinsiyet", ["Female", "Male"],
                               format_func=lambda g: "Kadın" if g == "Female" else "Erkek"),
        "age": c2.number_input("Yaş", 18, 100, 40),
        "tenure": c2.number_input("Müşteri süresi (yıl)", 0, 10, 5),
        "balance": c2.number_input("Bakiye", 0.0, 300000.0, 0.0, step=1000.0),
        "products_number": c3.number_input("Ürün sayısı", 1, 4, 1),
        "credit_card": int(c3.checkbox("Kredi kartı var", value=True)),
        "active_member": int(c3.checkbox("Aktif üye", value=True)),
        "estimated_salary": c3.number_input("Tahmini maaş", 1.0, 250000.0, 100000.0, step=1000.0),
    }
    return prepare_features(pd.DataFrame([row]))


def tab_risk(art):
    preds, features = art["preds"], art["features"]
    key = st.selectbox("Model", list(MODEL_NAMES), format_func=MODEL_NAMES.get, index=2)
    opt = art["thresholds"][key]
    thr = st.slider("Karar eşiği (olasılık ≥ eşik → uyarı)", 0.05, 0.95, float(opt), 0.01,
                    key=f"thr_{key}", help=f"Varsayılan: validation setinde F2'yi "
                                           f"maksimize eden eşik ({opt:.2f})")

    # Eşiğin test setindeki etkisi
    y = preds["churn"]
    alert = preds[f"proba_{key}"] >= thr
    tp = int((alert & (y == 1)).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uyarı sayısı (test)", f"{int(alert.sum())} / {len(preds)}")
    c2.metric("Recall", f"{tp / max(int(y.sum()), 1):.3f}")
    c3.metric("Precision", f"{tp / max(int(alert.sum()), 1):.3f}")
    c4.metric("Yakalanan churn", f"{tp} / {int(y.sum())}")
    st.divider()

    mode = st.radio("Müşteri", ["Test setinden seç", "Elle değer gir"], horizontal=True)
    if mode == "Test setinden seç":
        cid = st.selectbox("customer_id", preds["customer_id"])
        x_raw = preds.loc[preds["customer_id"] == cid, features]
        actual = int(preds.loc[preds["customer_id"] == cid, "churn"].iloc[0])
        st.caption(f"Gerçek durum: {'Terk etti (churn=1)' if actual else 'Kaldı (churn=0)'}")
    else:
        x_raw = manual_input_form()[features]

    proba = float(art["models"][key].predict_proba(x_raw)[0, 1])
    high = proba >= thr
    c1, c2 = st.columns(2)
    c1.metric("Churn olasılığı", f"%{proba * 100:.1f}")
    c2.metric("Risk etiketi", "YÜKSEK RİSK - uyarı" if high else "Düşük risk")

    st.markdown("**SHAP waterfall** (bu müşterinin skorunu hangi değişkenler artırdı/azalttı)")
    unit = {"lr": "log-odds", "xgb": "log-odds", "rf": "olasılık"}[key]
    st.caption(f"Birim: {unit}. Kırmızı çubuklar churn riskini artırır, mavi olanlar azaltır.")
    shap.plots.waterfall(explain_row(key, art, x_raw), max_display=10, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)


# ---------- Sekme 4: Açıklanabilirlik ----------
def tab_shap(art):
    key = st.radio("Model", ["xgb", "rf"], format_func=MODEL_NAMES.get, horizontal=True)
    s = art["shap"][key]
    X_test = art["preds"][art["features"]]
    expl = shap.Explanation(values=s["values"], base_values=np.full(len(X_test), s["base"]),
                            data=X_test.to_numpy(), feature_names=art["features"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Global önem** (ortalama |SHAP|)")
        shap.plots.bar(expl, max_display=13, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Beeswarm** (her nokta bir test müşterisi)")
        shap.plots.beeswarm(expl, max_display=13, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
    top = pd.Series(np.abs(s["values"]).mean(axis=0), index=art["features"])
    st.dataframe(top.sort_values(ascending=False).round(4).rename("Ortalama |SHAP|"),
                 width="stretch")


# ---------- Sekme 5: Maliyet Analizi ----------
def tab_maliyet(art):
    st.markdown("**Maliyete duyarlı eşik.** Toplam maliyet = FN maliyeti × kaçırılan churn "
                "+ FP maliyeti × yanlış alarm. Eşik **validation** setinde seçilir, "
                "sonuçlar **test** setinde raporlanır (test setine eşik ayarlatılmaz).")
    c1, c2 = st.columns(2)
    cost_fn = c1.number_input("FN maliyeti: kaçırılan churn müşterisi (varsayımsal birim)",
                              0.01, 1000.0, DEFAULT_COST_FN, 0.5, key="cost_fn")
    cost_fp = c2.number_input("FP maliyeti: yanlış alarm / kampanya (varsayımsal birim)",
                              0.01, 1000.0, DEFAULT_COST_FP, 0.5, key="cost_fp")
    st.caption(f"Varsayılan değerler (FN={DEFAULT_COST_FN:g}, FP={DEFAULT_COST_FP:g}) "
               "varsayımsaldır: kaçırılan müşterinin kampanya maliyetinden birkaç kat pahalı "
               "olduğu kabul edildi. Gerçek banka maliyetleriyle değiştirilmelidir.")

    val, test = art["val_preds"], art["preds"]
    bayes = analysis.bayes_threshold(cost_fn, cost_fp)
    fig, ax = plt.subplots(figsize=(9, 4))
    rows = []
    for key, name in MODEL_NAMES.items():
        opt, curve = analysis.cost_optimal_threshold(val["churn"], val[f"proba_{key}"],
                                                     cost_fn, cost_fp)
        ax.plot(curve["Eşik"], curve["Maliyet"], color=COLORS[key], lw=2, label=name)
        ax.scatter([opt], [curve["Maliyet"].min()], color=COLORS[key], s=40, zorder=3)
        for label, t in [("Maliyet-optimal", opt), ("F2-optimal", art["thresholds"][key]),
                         ("0.50", 0.50)]:
            rows.append({"Model": name, "Eşik türü": label, "Eşik": t,
                         **analysis.evaluate_at(test["churn"], test[f"proba_{key}"],
                                                t, cost_fn, cost_fp)})
    ax.axvline(bayes, color="#898781", lw=1, ls="--", label=f"Bayes eşiği ({bayes:.2f})")
    style_axes(ax, "Validation setinde toplam maliyet (noktalar: minimum)",
               "Karar eşiği", "Toplam maliyet (birim)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Test seti sonuçları** (üç eşik; maliyetler aynı FN/FP değerleriyle)")
    table = pd.DataFrame(rows)
    st.dataframe(table.round({"Eşik": 2, "Recall": 3, "Precision": 3, "Toplam maliyet": 1}),
                 hide_index=True, width="stretch")
    st.caption(f"Teorik referans: olasılıklar kalibre ise beklenen maliyeti minimize eden "
               f"Bayes eşiği FP/(FP+FN) = {bayes:.3f}. Bu, kalibrasyon varsayımı gerektirir; "
               f"SMOTE-NC eğitimde churn oranını yükselttiği için model olasılıkları yukarı "
               f"kayar, bu yüzden ampirik optimum bu değerle birebir örtüşmeyebilir.")


# ---------- Sekme 6: Özgünlük - Confounding Kontrolü ----------
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


def fmt_p(v):
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def tab_confounding(df):
    st.markdown("**Soru:** Almanya'daki yüksek churn ülkenin kendisinden mi, yoksa bakiye "
                "yapısından mı kaynaklanıyor? Tam veri seti (10.000 müşteri) üzerinde "
                "betimsel ve çıkarımsal istatistik; model tahmini kullanılmaz.")
    r = confounding_results(df)

    st.markdown("**1) Ülke × bakiye durumu**")
    left, right = st.columns(2)
    left.caption("Sayılar")
    left.dataframe(r["counts"], width="stretch")
    right.caption("Satır yüzdeleri (%)")
    right.dataframe(r["row_pct"].round(1), width="stretch")
    chi = r["chi"]
    st.caption(f"Ki-kare bağımsızlık testi: χ² = {chi['chi2']:.1f}, sd = {chi['dof']}, "
               f"p {fmt_p(chi['p'])} (en küçük beklenen frekans {chi['min_expected']:.0f} ≥ 5).")

    st.markdown("**2) Tabakalı churn oranları** (bakiye durumu sabitken ülkeler)")
    strat = r["strat"]
    wide = strat.pivot(index="Tabaka", columns="Ülke", values="Churn oranı (%)")
    wide = wide.reindex(["Tümü", "Bakiye = 0", "Bakiye > 0"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x, w = np.arange(len(wide)), 0.26
    for i, c in enumerate(analysis.COUNTRIES):
        ax.bar(x + (i - 1) * w, wide[c].fillna(0), w, color=COUNTRY_COLORS[c], label=c)
    ax.set_xticks(x, wide.index)
    style_axes(ax, "Ülke bazında churn oranı", "", "Churn oranı (%)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(strat.round(1), hide_index=True, width="stretch")
    st.caption("Boş (NaN) hücre: o tabakada o ülkeden müşteri yok (grafikte çubuk çizilmez).")

    st.markdown("**3) Bakiye dağılımı** (yalnız bakiye > 0 müşteriler)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, title in [(axes[0], "balance", "Bakiye"),
                           (axes[1], "balance_salary_ratio", "Bakiye / maaş oranı")]:
        data = [r["bal_df"].loc[r["bal_df"]["country"] == c, col] for c in analysis.COUNTRIES]
        box = ax.boxplot(data, tick_labels=analysis.COUNTRIES, showfliers=False,
                         patch_artist=True, medianprops={"color": "black"})
        for patch, c in zip(box["boxes"], analysis.COUNTRIES):
            patch.set_facecolor(COUNTRY_COLORS[c])
            patch.set_alpha(0.6)
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(r["bal_stats"].round(2), hide_index=True, width="stretch")
    st.caption("Kutu grafiklerinde aykırı değerler gösterilmedi (bakiye/maaş oranında çok "
               "düşük maaşlı birkaç müşteri ölçeği bozuyor); medyan ve IQR tabloda.")

    st.markdown("**4) Lojistik regresyon: Germany odds ratio'su** (referans: France)")
    g = r["germany"].copy()
    g["p"] = g["p"].map(fmt_p)
    st.dataframe(g.round({"OR": 3, "GA alt": 3, "GA üst": 3, "Pseudo R²": 3}),
                 hide_index=True, width="stretch")
    st.caption(f"Kontrol eklenince OR değişimi: %{r['change']:+.1f}. Model (b) kontrolleri: "
               "has_balance, merkezlenmiş log(bakiye) (bakiye>0 için), yaş, cinsiyet, aktif "
               "üyelik, ürün sayısı (1 / 2 / 3+). %95 GA Wald yöntemiyle. Birincil hipotez "
               "tek (Germany katsayısı), bu yüzden çoklu karşılaştırma düzeltmesi yapılmadı; "
               "aşağıdaki diğer katsayıların p-değerleri yalnız betimsel okunmalı.")
    with st.expander("Model (b) tüm katsayılar (odds ratio)"):
        coef = r["coef_b"].copy()
        coef["p"] = coef["p"].map(fmt_p)
        st.dataframe(coef.round(3), width="stretch")

    st.markdown("**Yorum** (sonuçlardan otomatik üretilir)")
    st.info("\n\n".join(r["text"]))


def main():
    st.set_page_config(page_title="Churn Erken Uyarı", layout="wide")
    st.title("Banka Müşteri Terki (Churn) Erken Uyarı Sistemi")
    if not (OUT / "models.joblib").exists():
        st.error("Model dosyaları bulunamadı. Önce `train.py` çalıştırın.")
        st.stop()
    df, art = load_data(), load_artifacts()
    t1, t2, t3, t4, t5, t6 = st.tabs(["Veri Özeti", "Model Karşılaştırma",
                                      "Müşteri Risk Skoru", "Açıklanabilirlik",
                                      "Maliyet Analizi", "Özgünlük: Confounding Kontrolü"])
    with t1:
        tab_veri(df)
    with t2:
        tab_karsilastirma(art)
    with t3:
        tab_risk(art)
    with t4:
        tab_shap(art)
    with t5:
        tab_maliyet(art)
    with t6:
        tab_confounding(df)


if __name__ == "__main__":
    main()
