r"""SMOTE-NC'li ve SMOTE'suz (yalnız eşik ayarlı) modellerin karşılaştırması - yalnız analiz.

Çalıştırma:  .\.venv\Scripts\python.exe smote_karsilastirma.py

Savunma sorusu: "SMOTE olmadan, sadece eşiği ayarlayarak aynı sonucu alır mıydınız?"
(van den Goorbergh vd. 2022: yeniden örnekleme ayrıştırmayı iyileştirmez, kalibrasyonu bozar.)

train.py ile birebir aynı: veri, özellikler, %60/%20/%20 tabakalı ayrım (seed 42),
hiperparametreler, ölçekleme, F2-optimal eşik (validation). Maliyet-optimal eşik
analysis.py'deki fonksiyonla (FN=5, FP=1, validation). Tek fark: (b) varyantında SMOTE-NC
adımı çıkarılır; sınıf ağırlığı / scale_pos_weight EKLENMEZ.

Mevcut artifacts dosyalarına dokunmaz; yalnızca şunları yazar:
  artifacts/smote_karsilastirma.json, artifacts/smote_karsilastirma.md
"""
import json

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar

from analysis import cost_optimal_threshold, evaluate_at
from train import (DATA, MODEL_NAMES, OUT, SEED, best_f2_threshold, build_models,
                   metrics_at, prepare_features)

COST_FN, COST_FP = 5, 1
VARIANTS = {"smote": "SMOTE-NC", "no_smote": "SMOTE'suz"}


def split():
    df = pd.read_csv(DATA)
    X, y = prepare_features(df), df["churn"]
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=SEED)
    return X, X_train, X_val, X_test, y_train, y_val, y_test


def without_smote(pipe):
    """Aynı pipeline, SMOTE adımı çıkarılmış (ölçekleme + aynı hiperparametreli model)."""
    return Pipeline([(n, clone(s)) for n, s in pipe.steps if n != "smote"])


def evaluate(y_test, proba, thr):
    m = metrics_at(y_test, proba, thr)
    c = evaluate_at(y_test, proba, thr, COST_FN, COST_FP)
    return {"threshold": thr, "recall": m["Recall"], "precision": m["Precision"],
            "f2": m["F2"], "fn": c["FN"], "fp": c["FP"], "cost": c["Toplam maliyet"]}


def main():
    X, X_train, X_val, X_test, y_train, y_val, y_test = split()
    yt = y_test.to_numpy()
    base = build_models(list(X.columns))
    ref_pred = pd.read_csv(OUT / "test_predictions.csv")
    with open(OUT / "thresholds.json", encoding="utf-8") as f:
        ref_thr = json.load(f)

    results, probas, sanity = {}, {}, {}
    for key, pipe in base.items():
        results[key] = {}
        for var, model in [("smote", clone(pipe)), ("no_smote", without_smote(pipe))]:
            model.fit(X_train, y_train)
            pv = model.predict_proba(X_val)[:, 1]
            pt = model.predict_proba(X_test)[:, 1]
            probas[(key, var)] = pt
            f2_thr, val_f2 = best_f2_threshold(y_val, pv)
            cost_thr, _ = cost_optimal_threshold(y_val, pv, COST_FN, COST_FP)
            m = metrics_at(y_test, pt, f2_thr)
            results[key][var] = {
                "val_f2": val_f2,
                "f2_threshold": evaluate(y_test, pt, f2_thr),
                "cost_threshold": evaluate(y_test, pt, cost_thr),
                "roc_auc": m["ROC-AUC"], "pr_auc": m["PR-AUC"],
                "mean_pred": float(pt.mean()), "actual_rate": float(yt.mean()),
                "brier": float(brier_score_loss(yt, pt)),
            }
            if var == "smote":
                diff = float(np.max(np.abs(pt - ref_pred[f"proba_{key}"].to_numpy())))
                sanity[key] = {"max_abs_proba_diff": diff,
                               "threshold_now": f2_thr,
                               "threshold_artifact": ref_thr[key]["threshold"],
                               "reproduced": bool(diff < 1e-6
                                                  and f2_thr == ref_thr[key]["threshold"])}
            r = results[key][var]
            print(f"{MODEL_NAMES[key]:<20} {VARIANTS[var]:<10} F2-eşik {f2_thr:.2f} "
                  f"recall {r['f2_threshold']['recall']:.3f} F2 {r['f2_threshold']['f2']:.3f} "
                  f"maliyet {r['f2_threshold']['cost']} | maliyet-eşik {cost_thr:.2f} "
                  f"maliyet {r['cost_threshold']['cost']} | PR-AUC {r['pr_auc']:.3f} "
                  f"ROC-AUC {r['roc_auc']:.3f} | ort. {r['mean_pred']:.4f} Brier {r['brier']:.4f}")

    # McNemar: XGB SMOTE vs SMOTE'suz, her biri kendi F2 eşiğinde (tek karşılaştırma)
    c = {v: ((probas[("xgb", v)] >= results["xgb"][v]["f2_threshold"]["threshold"]).astype(int)
             == yt) for v in VARIANTS}
    a, b = c["smote"], c["no_smote"]
    table = [[int(np.sum(a & b)), int(np.sum(a & ~b))], [int(np.sum(~a & b)), int(np.sum(~a & ~b))]]
    n_disc = table[0][1] + table[1][0]
    res = mcnemar(table, exact=n_disc < 25, correction=True)
    mc = {"comparison": "XGBoost SMOTE-NC vs SMOTE'suz (kendi F2 eşikleri)",
          "only_smote_correct": table[0][1], "only_no_smote_correct": table[1][0],
          "statistic": float(res.statistic), "p": float(res.pvalue)}
    print(f"\nMcNemar XGB: {mc}")
    print(f"Yeniden üretim kontrolü: {sanity}")

    out = {"seed": SEED, "cost_fn": COST_FN, "cost_fp": COST_FP,
           "note": "Eşikler validation'da seçildi; metrikler test setinde. Brier/ortalama ham olasılık.",
           "sanity_check": sanity, "results": results, "mcnemar_xgb": mc}
    with open(OUT / "smote_karsilastirma.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    write_md(results, sanity, mc, float(yt.mean()))
    print(f"Kaydedildi: {OUT / 'smote_karsilastirma.json'}, {OUT / 'smote_karsilastirma.md'}")


def write_md(results, sanity, mc, rate):
    L = ["# SMOTE-NC ve SMOTE'suz karşılaştırma (test seti)", "",
         "Aynı ayrım, özellikler, hiperparametreler; eşikler validation'da seçildi. "
         f"Maliyet: FN={COST_FN}, FP={COST_FP}. Gerçek churn oranı (test): %{rate * 100:.2f}.", "",
         "| Model | Varyant | F2 eşiği | Recall | Precision | F2 | Maliyet | Maliyet eşiği | "
         "Maliyet (maliyet eşiği) | PR-AUC | ROC-AUC | Ort. tahmin | Brier |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in results.items():
        for var, r in d.items():
            f, c = r["f2_threshold"], r["cost_threshold"]
            L.append(f"| {MODEL_NAMES[key]} | {VARIANTS[var]} | {f['threshold']:.2f} | "
                     f"{f['recall']:.3f} | {f['precision']:.3f} | {f['f2']:.3f} | {f['cost']} | "
                     f"{c['threshold']:.2f} | {c['cost']} | {r['pr_auc']:.3f} | {r['roc_auc']:.3f} | "
                     f"%{r['mean_pred'] * 100:.2f} | {r['brier']:.4f} |")
    L += ["", "## Yeniden üretim kontrolü (SMOTE-NC varyantı vs mevcut artifacts)", ""]
    for key, s in sanity.items():
        L.append(f"- {MODEL_NAMES[key]}: maks. olasılık farkı {s['max_abs_proba_diff']:.2e}, "
                 f"eşik {s['threshold_now']:.2f} (artifact {s['threshold_artifact']:.2f}) -> "
                 f"{'birebir' if s['reproduced'] else 'UYUMSUZ'}")
    L += ["", "## McNemar (XGBoost, SMOTE-NC vs SMOTE'suz, kendi F2 eşikleri)", "",
          f"Yalnız SMOTE-NC doğru: {mc['only_smote_correct']}, yalnız SMOTE'suz doğru: "
          f"{mc['only_no_smote_correct']}, istatistik {mc['statistic']:.3f}, p = {mc['p']:.4f}."]
    (OUT / "smote_karsilastirma.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
