"""Iris - shap paketi olmadan SHAP değerleri (shap -> numba zinciri Windows Akıllı Uygulama
Denetimi tarafından engellendiği için; bkz. tasarim-taslak/noshap/ilerleme.md).

Önceki shap ayarlarıyla matematiksel olarak aynı sonuçlar:
- XGBoost: shap.TreeExplainer(model) (arka plan yok -> tree_path_dependent, çıktı log-odds)
  == XGBoost'un yerleşik TreeSHAP'i: Booster.predict(DMatrix, pred_contribs=True). Son sütun
  (bias) = taban değer (expected value).
- Random Forest: shap.TreeExplainer(model) (tree_path_dependent, çıktı olasılık). sklearn ağaçları
  aynı yapıda (eşik, yaprak oranı / ağaç sayısı, düğüm örnek ağırlığı = cover) bir XGBoost
  Booster'ına aktarılır ve yine pred_contribs ile hesaplanır.
- Lojistik Regresyon: shap.LinearExplainer + Independent masker (arka plan 200 örnek, interventional)
  == katkı_i = coef_i * (x_i - ortalama_i), taban = coef · ortalama + intercept (log-odds).
Hepsi modelin ölçeklenmiş girdisiyle çalışır (pipeline'daki scaler); SMOTE yalnız eğitimde vardı.
Doğrulama (2026-10-02, test 2.000 satır, shap 0.52 ile kaydedilmiş shap_test.joblib'e karşı):
XGB maks. fark 0,0; RF maks. fark 2e-7 (float32 yuvarlaması).
"""
import json
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

_ROOT_PARENT = 2147483647


@dataclass
class Explanation:
    """shap.Explanation'ın uygulamanın kullandığı alanları (tek satır)."""
    values: np.ndarray
    base_values: float
    data: np.ndarray
    feature_names: list


def _sk_threshold_to_xgb(thr):
    """sklearn: sol <=> x(float32) <= eşik(float64). XGBoost: sol <=> x(float32) < koşul(float32).
    Eşdeğer koşul: eşiğe eşit/altındaki en büyük float32'nin bir sonraki float32 değeri."""
    t = np.float32(thr)
    if float(t) > thr:
        t = np.nextafter(t, np.float32(-np.inf))
    return float(np.nextafter(t, np.float32(np.inf)))


def _sk_tree_to_xgb(est, tree_id, scale, cls=1):
    t = est.tree_
    n = t.node_count
    left, right = t.children_left.copy(), t.children_right.copy()
    is_leaf = left == -1
    val = t.value[:, 0, :]
    frac = val[:, cls] / val.sum(axis=1)  # sklearn sürümünden bağımsız: sınıf oranı
    parents = np.full(n, _ROOT_PARENT, dtype=np.int64)
    for i in np.where(~is_leaf)[0]:
        parents[left[i]] = i
        parents[right[i]] = i
    cond = np.where(is_leaf, frac * scale,
                    [_sk_threshold_to_xgb(th) if not lf else 0.0 for th, lf in zip(t.threshold, is_leaf)])
    return {
        "base_weights": [float(v) for v in frac * scale],
        "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": [],
        "default_left": [0] * n,
        "id": tree_id,
        "left_children": [int(v) for v in left],
        "loss_changes": [0.0] * n,
        "parents": [int(v) for v in parents],
        "right_children": [int(v) for v in right],
        "split_conditions": [float(v) for v in cond],
        "split_indices": [int(f) if not lf else 0 for f, lf in zip(t.feature, is_leaf)],
        "split_type": [0] * n,
        "sum_hessian": [float(v) for v in t.weighted_n_node_samples],
        "tree_param": {"num_deleted": "0", "num_feature": str(est.n_features_in_),
                       "num_nodes": str(n), "size_leaf_vector": "1"},
    }


def rf_to_booster(rf, cls=1):
    """sklearn RandomForestClassifier -> XGBoost Booster (sınıf `cls` olasılığı = ağaç ortalaması)."""
    trees = rf.estimators_
    k = len(trees)
    model = {
        "learner": {
            "attributes": {}, "feature_names": [], "feature_types": [],
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": str(k)},
                    "iteration_indptr": list(range(k + 1)),
                    "tree_info": [0] * k,
                    "trees": [_sk_tree_to_xgb(e, i, 1.0 / k, cls) for i, e in enumerate(trees)],
                },
                "name": "gbtree",
            },
            "learner_model_param": {"base_score": "[0E0]", "boost_from_average": "0", "num_class": "0",
                                    "num_feature": str(rf.n_features_in_), "num_target": "1"},
            "objective": {"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "1"}},
        },
        "version": [int(v) for v in xgb.__version__.split(".")[:3]],
    }
    b = xgb.Booster()
    b.load_model(bytearray(json.dumps(model).encode("utf-8")))
    return b


def booster_for(key, pipe):
    """Ağaç modelleri için TreeSHAP'i çalıştıracak XGBoost Booster."""
    model = pipe.named_steps["model"]
    if key == "xgb":
        return model.get_booster()
    if key == "rf":
        return rf_to_booster(model, cls=list(model.classes_).index(1))
    raise ValueError(key)


def tree_contribs(booster, x_scaled):
    """TreeSHAP (tree_path_dependent). Dönüş: (katkılar [n, p], taban [n])."""
    c = booster.predict(xgb.DMatrix(np.asarray(x_scaled, dtype=np.float32)),
                        pred_contribs=True, approx_contribs=False)
    return c[:, :-1].astype(float), c[:, -1].astype(float)


def linear_contribs(model, x_scaled, bg_scaled):
    """LinearExplainer (Independent masker, interventional) ile aynı: log-odds."""
    coef = np.ravel(model.coef_)
    mean = np.asarray(bg_scaled, dtype=float).mean(axis=0)
    phi = (np.asarray(x_scaled, dtype=float) - mean) * coef
    base = float(np.dot(coef, mean) + np.ravel(model.intercept_)[0])
    return phi, np.full(len(phi), base)


def contribs(key, pipe, x_raw, background=None, booster=None):
    """Ham girdi (DataFrame) → (katkılar [n, p], taban [n]); modelin kendi çıktı ölçeğinde
    (XGB/LR log-odds, RF olasılık). LR için background (ham) gerekir."""
    scaler, model = pipe.named_steps["scaler"], pipe.named_steps["model"]
    x_scaled = scaler.transform(x_raw)
    if key == "lr":
        return linear_contribs(model, x_scaled, scaler.transform(background))
    return tree_contribs(booster if booster is not None else booster_for(key, pipe), x_scaled)
