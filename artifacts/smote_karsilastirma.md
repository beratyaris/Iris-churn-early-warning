# SMOTE-NC ve SMOTE'suz karşılaştırma (test seti)

Aynı ayrım, özellikler, hiperparametreler; eşikler validation'da seçildi. Maliyet: FN=5, FP=1. Gerçek churn oranı (test): %20.35.

| Model | Varyant | F2 eşiği | Recall | Precision | F2 | Maliyet | Maliyet eşiği | Maliyet (maliyet eşiği) | PR-AUC | ROC-AUC | Ort. tahmin | Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lojistik Regresyon | SMOTE-NC | 0.21 | 0.840 | 0.296 | 0.614 | 1140 | 0.28 | 1054 | 0.480 | 0.765 | %29.95 | 0.1465 |
| Lojistik Regresyon | SMOTE'suz | 0.11 | 0.887 | 0.279 | 0.617 | 1165 | 0.19 | 1059 | 0.482 | 0.775 | %20.28 | 0.1346 |
| Random Forest | SMOTE-NC | 0.25 | 0.840 | 0.403 | 0.690 | 832 | 0.25 | 832 | 0.689 | 0.863 | %27.10 | 0.1111 |
| Random Forest | SMOTE'suz | 0.19 | 0.813 | 0.444 | 0.697 | 794 | 0.21 | 791 | 0.706 | 0.866 | %20.44 | 0.1040 |
| XGBoost | SMOTE-NC | 0.21 | 0.821 | 0.434 | 0.697 | 800 | 0.21 | 800 | 0.711 | 0.866 | %24.43 | 0.1041 |
| XGBoost | SMOTE'suz | 0.16 | 0.801 | 0.436 | 0.686 | 827 | 0.16 | 827 | 0.708 | 0.862 | %20.22 | 0.1016 |

## Yeniden üretim kontrolü (SMOTE-NC varyantı vs mevcut artifacts)

- Lojistik Regresyon: maks. olasılık farkı 1.67e-16, eşik 0.21 (artifact 0.21) -> birebir
- Random Forest: maks. olasılık farkı 3.33e-16, eşik 0.25 (artifact 0.25) -> birebir
- XGBoost: maks. olasılık farkı 2.97e-08, eşik 0.21 (artifact 0.21) -> birebir

## McNemar (XGBoost, SMOTE-NC vs SMOTE'suz, kendi F2 eşikleri)

Yalnız SMOTE-NC doğru: 63, yalnız SMOTE'suz doğru: 68, istatistik 0.122, p = 0.7267.
