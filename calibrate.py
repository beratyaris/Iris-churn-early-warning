r"""Olasılık kalibrasyonu kontrolü ve (gerekirse) kalibratör eğitimi.

Çalıştırma:  .\.venv\Scripts\python.exe calibrate.py      (train.py'den SONRA)

Neden: modeller eğitimde SMOTE-NC ile (churn payı yapay olarak artırılarak) eğitildi; bu
yüzden tahmin edilen olasılıklar gerçek churn oranına göre yukarı kayar. Bu betik:
  1) Test setinde her model için ortalama tahmin vs gerçek churn oranı, Brier skoru ve
     10 eşit genişlikli kutuda güvenilirlik (reliability) eğrisi hesaplar.
  2) Fark > 3 puan ise kalibratörü YALNIZ validation setinde eğitir. İki yöntem denenir:
     isotonic ve Platt/sigmoid (logit(p) üzerinde lojistik regresyon). Seçim validation
     içinde 5 katlı tabakalı çapraz doğrulama Brier skoruyla yapılır (kalibratörü eğitildiği
     veride değerlendirmek isotonic'i her zaman kazandırır; CV bunu önler). Seçilen yöntem
     tüm validation setiyle yeniden eğitilir; test seti yalnız raporlama içindir.
  3) Sonuçları artifacts/calibration.joblib (kalibratörler) ve artifacts/calibration.json
     (özet sayılar + eğriler) dosyalarına yazar. train.py ve çıktıları DEĞİŞMEZ.

Önemli: karar eşikleri (thresholds.json) kalibre EDİLMEMİŞ skorlar üzerinde seçildi.
Uyarı listesi / sınıflandırma bu yüzden orijinal skor + orijinal eşikle yapılır; kalibre
olasılık yalnız "beklenen churn oranı" ve ekranda gösterilen olasılık içindir.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from train import MODEL_NAMES, OUT, SEED

MATERIAL_GAP = 0.03   # ortalama tahmin ile gerçek oran arası "anlamlı" fark (3 puan)
N_BINS = 10
EPS = 1e-6


class PlattCalibrator:
    """Platt ölçekleme: logit(p) üzerinde tek değişkenli lojistik regresyon."""

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(p), y)
        return self

    @staticmethod
    def _z(p):
        p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, p):
        return self.lr.predict_proba(self._z(p))[:, 1]


def make(method):
    if method == "isotonic":
        return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    return PlattCalibrator()


def reliability(y, p, n_bins=N_BINS):
    """10 eşit genişlikli kutu: her kutuda ortalama tahmin, gerçek oran, n."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "mean_pred": float(p[m].mean()),
                         "frac_pos": float(y[m].mean()), "n": int(m.sum())})
    return rows


def cv_brier(p, y, method, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    out = np.empty(len(y))
    for tr, te in skf.split(p.reshape(-1, 1), y):
        out[te] = make(method).fit(p[tr], y[tr]).predict(p[te])
    return float(brier_score_loss(y, out))


def main():
    test = pd.read_csv(OUT / "test_predictions.csv")
    val = pd.read_csv(OUT / "val_predictions.csv")
    yt, yv = test["churn"].to_numpy(), val["churn"].to_numpy()
    summary = {"material_gap": MATERIAL_GAP, "n_bins": N_BINS, "actual_rate_test": float(yt.mean()),
               "actual_rate_val": float(yv.mean()), "models": {}}
    calibrators = {}
    for key in MODEL_NAMES:
        pt, pv = test[f"proba_{key}"].to_numpy(), val[f"proba_{key}"].to_numpy()
        info = {"mean_pred_test": float(pt.mean()), "brier_test": float(brier_score_loss(yt, pt)),
                "gap_test": float(pt.mean() - yt.mean()),
                "reliability_test": reliability(yt, pt)}
        info["cv_brier_val"] = {m: cv_brier(pv, yv, m) for m in ("isotonic", "sigmoid")}
        info["brier_val_raw"] = float(brier_score_loss(yv, pv))
        method = min(info["cv_brier_val"], key=info["cv_brier_val"].get)
        cal = make(method).fit(pv, yv)
        pc = np.clip(cal.predict(pt), 0, 1)
        info.update({
            "method": method,
            "applied": bool(abs(info["gap_test"]) > MATERIAL_GAP),
            "mean_pred_test_cal": float(pc.mean()),
            "brier_test_cal": float(brier_score_loss(yt, pc)),
            "reliability_test_cal": reliability(yt, pc),
        })
        summary["models"][key] = info
        calibrators[key] = cal
        print(f"{MODEL_NAMES[key]:<20} ort. tahmin {pt.mean():.4f} (gerçek {yt.mean():.4f}, "
              f"fark {info['gap_test'] * 100:+.1f} puan) | Brier {info['brier_test']:.4f} -> "
              f"{info['brier_test_cal']:.4f} ({method}; CV Brier iso "
              f"{info['cv_brier_val']['isotonic']:.4f} / sig {info['cv_brier_val']['sigmoid']:.4f}) "
              f"| kalibre ort. {pc.mean():.4f} | uygulandı: {info['applied']}")
    joblib.dump(calibrators, OUT / "calibration.joblib")
    with open(OUT / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Kaydedildi: {OUT / 'calibration.joblib'}, {OUT / 'calibration.json'}")


if __name__ == "__main__":
    # Modülü adıyla içe aktarıp çalıştır: pickle'daki sınıf yolu "__main__" değil
    # "calibrate.PlattCalibrator" olsun ki uygulama kalibratörü yükleyebilsin.
    import calibrate
    calibrate.main()
