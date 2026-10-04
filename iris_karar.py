"""Karar Destek hesapları: profil nadirlik kontrolü ve kampanya planı.

Saf pandas/numpy; Streamlit, SHAP ya da model gerektirmez (smoke_test bunları doğrudan sınar).

1) Nadirlik kontrolü ("Model ne derdi?" simülatörü)
   Kara kutu yok, yalnız sayım. Profilin her değişkeni okunaklı bir gruba konur (ör. yaş 40–49,
   bakiye 0). Sonra EĞİTİM setinde (6.000 müşteri; modelin gördüğü veri):
     - tek değişken: profilin grubundaki müşteri sayısı eğitim setinin %1'inden azsa  -> çok nadir
     - değişken çifti: profilin iki grubunu birlikte taşıyan müşteri sayısı 10'dan azsa -> çok nadir
     - herhangi bir sayı 0 ise                                                        -> yok
   Örnek: Almanya × bakiye 0 eğitimde 0 müşteri (tüm veride 0 / 2.509) -> "yok".
2) Kampanya planı ("Kampanya planlayıcı")
   Müşteriler model skoruna göre sıralanır, bütçenin yettiği ilk N kişiye ulaşılır.
   Beklenen terk eden = bu N kişinin KALİBRE olasılıklarının toplamı; beklenen kurtarılan =
   beklenen terk eden × başarı oranı; net fayda = kurtarılan × müşteri değeri − N × temas maliyeti.
"""
from itertools import combinations

import numpy as np
import pandas as pd

# ---------- 1) Nadirlik kontrolü ----------
RARE_SINGLE_SHARE = 0.01   # tek değişken grubu: eğitim setinin %1'inden az müşteri -> çok nadir
RARE_PAIR_N = 10           # değişken çifti: 10'dan az müşteri -> çok nadir; 0 -> yok

GROUP_LABELS = {
    "country": "Ülke", "gender": "Cinsiyet", "age": "Yaş", "credit_score": "Kredi skoru",
    "tenure": "Müşteri süresi", "balance": "Bakiye", "products_number": "Ürün sayısı",
    "estimated_salary": "Tahmini maaş", "active_member": "Aktif üye", "credit_card": "Kredi kartı",
}
COUNTRY_TR = {"France": "Fransa", "Germany": "Almanya", "Spain": "İspanya"}
GENDER_TR = {"Female": "Kadın", "Male": "Erkek"}
_YESNO = {0: "Hayır", 1: "Evet"}


def _pick(idx, labels):
    return np.asarray(labels, dtype=object)[np.asarray(idx, dtype=int)]


def profile_groups(df):
    """Ham veri biçimindeki satırları (country, gender, age, ...) okunaklı gruplara çevirir."""
    age, cs = df["age"].to_numpy(float), df["credit_score"].to_numpy(float)
    ten, bal = df["tenure"].to_numpy(float), df["balance"].to_numpy(float)
    sal = df["estimated_salary"].to_numpy(float)
    g = pd.DataFrame(index=df.index)
    g["country"] = df["country"].map(COUNTRY_TR).to_numpy()
    g["gender"] = df["gender"].map(GENDER_TR).to_numpy()
    g["age"] = _pick(np.digitize(age, [30, 40, 50, 60, 70]),
                     ["18–29", "30–39", "40–49", "50–59", "60–69", "70+"])
    g["credit_score"] = _pick(np.digitize(cs, [500, 600, 700, 800]),
                              ["500 altı", "500–599", "600–699", "700–799", "800+"])
    g["tenure"] = _pick(np.digitize(ten, [1, 4, 7, 10]),
                        ["0 yıl", "1–3 yıl", "4–6 yıl", "7–9 yıl", "10 yıl"])
    g["balance"] = _pick(np.where(bal <= 0, 0, 1 + np.digitize(bal, [50000, 100000, 150000], right=True)),
                         ["0", "0–50.000 (0 hariç)", "50.000–100.000", "100.000–150.000", "150.000 üstü"])
    g["products_number"] = df["products_number"].astype(int).astype(str).to_numpy()
    g["estimated_salary"] = _pick(np.digitize(sal, [50000, 100000, 150000], right=True),
                                  ["50.000 ve altı", "50.000–100.000", "100.000–150.000", "150.000 üstü"])
    g["active_member"] = df["active_member"].astype(int).map(_YESNO).to_numpy()
    g["credit_card"] = df["credit_card"].astype(int).map(_YESNO).to_numpy()
    return g


def build_rarity_index(train_raw):
    """Eğitim setinden tek değişken ve değişken çifti sayımları (bir kez hesaplanır)."""
    g = profile_groups(train_raw)
    cols = list(g.columns)
    single = {c: g[c].value_counts().to_dict() for c in cols}
    pair = {(a, b): g.groupby([a, b]).size().to_dict() for a, b in combinations(cols, 2)}
    return {"n": len(g), "cols": cols, "single": single, "pair": pair}


def rarity_check(index, profile):
    """Tek profil için nadirlik kararı.

    Dönüş: level ('ok' | 'nadir' | 'yok'), issues (en küçük sayıdan büyüğe; her biri
    {'cols', 'groups', 'n', 'ref'}), min_n / min_desc (en küçük grup), n_train."""
    g = profile_groups(pd.DataFrame([profile])).iloc[0]
    n_train, single, pair = index["n"], index["single"], index["pair"]
    single_min = RARE_SINGLE_SHARE * n_train
    cells, issues = [], []
    for c in index["cols"]:
        n = int(single[c].get(g[c], 0))
        cell = {"cols": (c,), "groups": (g[c],), "n": n, "ref": n_train}
        cells.append(cell)
        if n < single_min:
            issues.append(cell)
    for a, b in combinations(index["cols"], 2):
        n = int(pair[(a, b)].get((g[a], g[b]), 0))
        # Bağlam: iki gruptan küçük olanın eğitimdeki büyüklüğü (ör. "Almanya: 1.504 müşteri")
        na, nb = int(single[a].get(g[a], 0)), int(single[b].get(g[b], 0))
        ref_col, ref = (a, na) if na <= nb else (b, nb)
        cell = {"cols": (a, b), "groups": (g[a], g[b]), "n": n, "ref": ref, "ref_col": ref_col}
        cells.append(cell)
        if n < RARE_PAIR_N:
            issues.append(cell)
    issues.sort(key=lambda d: (d["n"], len(d["cols"])))
    smallest = min(cells, key=lambda d: d["n"])
    level = "yok" if any(d["n"] == 0 for d in issues) else ("nadir" if issues else "ok")
    return {"level": level, "issues": issues, "min_n": smallest["n"], "min_desc": describe(smallest),
            "n_train": n_train}


def describe(cell):
    """'Ülke = Almanya × Bakiye = 0' biçiminde okunur grup adı."""
    return " × ".join(f"{GROUP_LABELS[c]} = {v}" for c, v in zip(cell["cols"], cell["groups"]))


# ---------- 2) Kampanya planı ----------
# Maliyet varsayımları - VARSAYIMSAL göreli birimler, gerçek banka verisi değildir.
# (Önceden sayfalar/maliyet.py içindeydi; Maliyet ve Eşik ile Kampanya planlayıcı aynı değeri
# kullansın diye buraya taşındı.) Gerekçe: kaçırılan bir churn müşterisi (FN) gelecekteki
# gelirin tamamen kaybıdır; yanlış alarm (FP) yalnızca bir elde tutma kampanyasının
# maliyetidir. FN'nin FP'den birkaç kat pahalı olduğu varsayımıyla 5:1 seçildi; kullanıcı
# arayüzden değiştirebilir. Kampanya planlayıcıda: temas maliyeti = FP, kaybedilen müşterinin
# maliyeti = FN.
DEFAULT_COST_FN = 5
DEFAULT_COST_FP = 1
# Kampanya başarı oranı için projede kaynak YOK; %50 yalnız başlangıç değeridir (ekranda
# "varsayım, kaynak yok" diye etiketli, kullanıcı değiştirir).
DEFAULT_SUCCESS = 0.50


def campaign_curve(scores, probs, contact_cost, success_rate, value):
    """N = 0..n için kümülatif plan. scores: model (ham) skoru -> sıralama; probs: kalibre
    olasılık -> beklenen terk. Kalibratör monoton olduğundan iki ölçek aynı sırayı verir."""
    scores, probs = np.asarray(scores, float), np.asarray(probs, float)
    order = np.argsort(-scores, kind="stable")
    exp_churn = np.concatenate([[0.0], np.cumsum(probs[order])])
    n = np.arange(len(scores) + 1)
    saved = exp_churn * success_rate
    cost = n * contact_cost
    return pd.DataFrame({"N": n, "beklenen_terk": exp_churn, "kurtarilan": saved, "maliyet": cost,
                         "net": saved * value - cost,
                         "son_skor": np.concatenate([[np.nan], scores[order]])})


def reach_for_budget(budget, contact_cost, n_max):
    """Bütçenin yettiği müşteri sayısı (tam kişi, en fazla n_max)."""
    if contact_cost <= 0:
        return int(n_max)
    return int(min(n_max, np.floor(budget / contact_cost + 1e-9)))
