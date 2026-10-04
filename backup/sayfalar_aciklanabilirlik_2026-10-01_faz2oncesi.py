"""Açıklanabilirlik (SHAP) sayfası (faz 1: eski sekme içeriği, ortak tema ile)."""
import iris_ui as ui
from iris_data import load_artifacts, load_data  # noqa: F401
from iris_legacy import tab_shap

ui.page_header("Analiz", "Açıklanabilirlik (SHAP)", "Global değişken önemi ve beeswarm")
tab_shap(load_artifacts())
