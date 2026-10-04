"""Özgünlük ekleri için analiz fonksiyonları (app.py ve smoke_test.py kullanır).

1) Confounding kontrolü: Almanya'daki yüksek churn ülkeden mi, bakiye yapısından mı?
   Tam veri (10.000 müşteri) üzerinde betimsel ve çıkarımsal istatistik; model tahmini değil.
2) Maliyete duyarlı eşik: validation tahminleriyle FN/FP maliyetini minimize eden eşik.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
# statsmodels.api yerine doğrudan alt modüller: api, Windows Akıllı Uygulama Denetimi'nin
# engellediği bir zaman serisi DLL'ini de yüklüyor; Logit'in buna ihtiyacı yok.
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

COUNTRIES = ["France", "Germany", "Spain"]  # France referans kategori


# ---------- 1) Confounding kontrolü ----------
def country_balance_table(df):
    """Ülke x has_balance çapraz tablosu (sayı ve satır yüzdesi) + ki-kare bağımsızlık testi."""
    has_bal = pd.Series(np.where(df["balance"] > 0, "Bakiye > 0", "Bakiye = 0"),
                        index=df.index, name="Bakiye")
    counts = pd.crosstab(df["country"].rename("Ülke"), has_bal).reindex(COUNTRIES)
    row_pct = counts.div(counts.sum(axis=1), axis=0) * 100
    chi2, p, dof, expected = chi2_contingency(counts)
    test = {"chi2": float(chi2), "dof": int(dof), "p": float(p),
            "min_expected": float(expected.min())}  # ki-kare varsayımı: beklenen >= 5
    return counts, row_pct, test


def stratified_churn(df):
    """Ülke bazında churn oranı (%): tüm müşteriler, yalnız balance=0, yalnız balance>0.
    Boş hücrede (ör. balance=0 olan Alman müşteri yoksa) oran NaN kalır."""
    strata = {"Tümü": df,
              "Bakiye = 0": df[df["balance"] == 0],
              "Bakiye > 0": df[df["balance"] > 0]}
    rows = []
    for name, d in strata.items():
        g = d.groupby("country")["churn"].agg(["mean", "size"]).reindex(COUNTRIES)
        for country, r in g.iterrows():
            n = 0 if pd.isna(r["size"]) else int(r["size"])
            rows.append({"Tabaka": name, "Ülke": country, "n": n,
                         "Churn oranı (%)": r["mean"] * 100 if n else np.nan})
    return pd.DataFrame(rows)


def balance_distribution(df):
    """Yalnız balance>0 müşterilerde ülke bazında bakiye ve bakiye/maaş oranı: medyan ve IQR."""
    d = df[df["balance"] > 0].assign(balance_salary_ratio=lambda x: x["balance"] / x["estimated_salary"])
    rows = []
    for col in ["balance", "balance_salary_ratio"]:
        for country, s in d.groupby("country")[col]:
            q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
            rows.append({"Değişken": col, "Ülke": country, "n": len(s),
                         "Q1": q1, "Medyan": med, "Q3": q3, "IQR": q3 - q1})
    return d, pd.DataFrame(rows)


def design_matrices(df):
    """Logit için açık tasarım matrisleri (formül yerine elle; ne girdiği net görünsün).
    (a) yalnız ülke dummy'leri; (b) ülke + bakiye + kontrol değişkenleri."""
    Xa = pd.DataFrame({
        "Germany": (df["country"] == "Germany").astype(float),
        "Spain": (df["country"] == "Spain").astype(float),
    })
    has_bal = (df["balance"] > 0).astype(float)
    log_bal = np.log(df["balance"].where(df["balance"] > 0))
    Xb = Xa.assign(
        has_balance=has_bal,
        # Bakiye düzeyi: bakiye>0 olanlarda log(balance)'ın ortalamadan sapması, balance=0 için 0.
        # Merkezleme has_balance ile log1p(balance) arasındaki ~%99 korelasyonu önler; böylece
        # has_balance katsayısı "ortalama pozitif bakiyede" bakiye var/yok farkını gösterir.
        log_balance_c=(log_bal - log_bal.mean()).fillna(0.0),
        age=df["age"].astype(float),
        male=(df["gender"] == "Male").astype(float),
        active_member=df["active_member"].astype(float),
        # Ürün sayısı ilişkisi monoton değil: kategorik (ref: 1 ürün). 4 ürünlü 60 müşterinin
        # tamamı churn olduğu için (tam ayrışma) 3 ve 4 birleştirildi.
        products_2=(df["products_number"] == 2).astype(float),
        products_3plus=(df["products_number"] >= 3).astype(float),
    )
    return add_constant(Xa), add_constant(Xb)


def fit_logits(df):
    """İki Logit modeli: (a) ham ülke etkisi, (b) kontrollü. Sonuç nesnelerini döner."""
    Xa, Xb = design_matrices(df)
    y = df["churn"].astype(float)
    return Logit(y, Xa).fit(disp=0), Logit(y, Xb).fit(disp=0)


def odds_ratio_table(res):
    """Katsayıları odds ratio, %95 Wald GA ve p-değeri olarak döner."""
    ci = np.exp(res.conf_int())
    return pd.DataFrame({"OR": np.exp(res.params), "GA alt": ci[0], "GA üst": ci[1],
                         "p": res.pvalues}).drop(index="const")


def germany_comparison(df):
    """Germany OR'si iki modelde yan yana + kontrol eklenince % değişim."""
    res_a, res_b = fit_logits(df)
    rows = []
    for name, res in [("(a) Yalnız ülke", res_a), ("(b) Ülke + kontroller", res_b)]:
        r = odds_ratio_table(res).loc["Germany"]
        rows.append({"Model": name, "OR": r["OR"], "GA alt": r["GA alt"],
                     "GA üst": r["GA üst"], "p": r["p"], "n": int(res.nobs),
                     "Pseudo R²": res.prsquared})
    table = pd.DataFrame(rows)
    or_a, or_b = table["OR"]
    change = (or_b - or_a) / or_a * 100
    return table, float(change), res_a, res_b


def _tr(v, dec):
    """Türkçe ondalık biçim (virgül)."""
    return f"{v:.{dec}f}".replace(".", ",")


def interpret(strat, germany, change):
    """Sonuçlardan otomatik, abartısız Türkçe yorum üretir (sabit iddia yok)."""
    or_a, or_b = germany["OR"]
    lo_b, hi_b, p_b = germany.iloc[1][["GA alt", "GA üst", "p"]]
    lines = []

    # Tabakalı karşılaştırma
    pos = strat[strat["Tabaka"] == "Bakiye > 0"].set_index("Ülke")["Churn oranı (%)"]
    zero_de = strat[(strat["Tabaka"] == "Bakiye = 0") & (strat["Ülke"] == "Germany")]["n"].iloc[0]
    if zero_de == 0:
        lines.append("Veride bakiyesi 0 olan Alman müşteri yok; bu yüzden Almanya yalnızca "
                     "bakiye > 0 tabakasında diğer ülkelerle karşılaştırılabiliyor.")
    others = pos.drop("Germany")
    lines.append(f"Bakiye > 0 tabakasında churn oranı Almanya'da %{_tr(pos['Germany'], 1)}, "
                 f"Fransa'da %{_tr(pos['France'], 1)}, İspanya'da %{_tr(pos['Spain'], 1)}; "
                 + ("aynı bakiye durumunda da Almanya daha yüksek." if pos["Germany"] > others.max()
                    else "aynı bakiye durumunda Almanya'nın farkı belirgin değil."))

    # Logit: OR değişimi
    direction = "azaldı" if change < 0 else "arttı"
    lines.append(f"Germany odds ratio'su kontrol değişkenleri eklenince {_tr(or_a, 2)} → "
                 f"{_tr(or_b, 2)} oldu; %{_tr(abs(change), 1)} {direction}.")
    if lo_b > 1 and p_b < 0.05:
        # %10'dan küçük değişim pratikte "yaklaşık aynı" sayılır (yaygın confounding kuralı)
        strength = ("" if abs(change) < 10 else
                    "zayıflayarak " if change < 0 else "güçlenerek ")
        lines.append(f"Kontrollü modelde %95 GA [{_tr(lo_b, 2)}; {_tr(hi_b, 2)}] 1'i içermiyor: "
                     f"Almanya etkisi bakiye ve diğer değişkenler sabit tutulduğunda da "
                     f"{strength}sürüyor. Yani yüksek churn yalnızca bakiye yapısıyla açıklanmıyor.")
    else:
        lines.append(f"Kontrollü modelde %95 GA [{_tr(lo_b, 2)}; {_tr(hi_b, 2)}] 1'i içeriyor: kontrol "
                     f"sonrası Almanya için anlamlı ek etki görülmüyor; ham fark büyük ölçüde "
                     f"bakiye yapısı ve diğer değişkenlerle açıklanıyor olabilir.")
    lines.append("Not: Bunlar gözlemsel verideki ilişkilerdir, nedensellik değildir; ölçülmeyen "
                 "değişkenler (ör. bölgesel rekabet, ürün koşulları) etkiyi açıklıyor olabilir. "
                 "Ayrıca odds ratio çökertilemez (non-collapsible) bir ölçüdür: değişken eklemek "
                 "confounding olmasa da OR'yi bir miktar değiştirebilir.")
    return lines


# ---------- 2) Maliyete duyarlı eşik ----------
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)  # 0.01 - 0.99


def confusion_at(y_true, proba, threshold):
    y_true = np.asarray(y_true)
    pred = np.asarray(proba) >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    return tp, fp, fn


def cost_curve(y_true, proba, cost_fn, cost_fp, grid=THRESHOLD_GRID):
    """Her eşikte toplam maliyet = cost_fn * FN + cost_fp * FP."""
    rows = []
    for t in grid:
        tp, fp, fn = confusion_at(y_true, proba, t)
        rows.append({"Eşik": float(t), "FN": fn, "FP": fp, "Maliyet": cost_fn * fn + cost_fp * fp})
    return pd.DataFrame(rows)


def cost_optimal_threshold(y_true, proba, cost_fn, cost_fp):
    """Maliyeti minimize eden eşik (eşitlikte en küçük eşik). Validation ile çağrılmalı."""
    curve = cost_curve(y_true, proba, cost_fn, cost_fp)
    best = curve.loc[curve["Maliyet"].idxmin()]
    return float(best["Eşik"]), curve


def evaluate_at(y_true, proba, threshold, cost_fn, cost_fp):
    """Bir eşikte recall, precision, FN, FP ve toplam maliyet."""
    tp, fp, fn = confusion_at(y_true, proba, threshold)
    return {"Recall": tp / max(tp + fn, 1), "Precision": tp / max(tp + fp, 1),
            "FN": fn, "FP": fp, "Toplam maliyet": cost_fn * fn + cost_fp * fp}


def bayes_threshold(cost_fn, cost_fp):
    """Olasılıklar kalibre ise beklenen maliyeti minimize eden teorik eşik: FP / (FP + FN)."""
    return cost_fp / (cost_fp + cost_fn)
