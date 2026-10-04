"""Confounding sayfası (faz 1: eski sekme içeriği, ortak tema ile)."""
import iris_ui as ui
from iris_data import load_artifacts, load_data  # noqa: F401
from iris_legacy import tab_confounding

ui.page_header("Analiz", "Confounding", "Özgünlük: Almanya etkisi ülkeden mi, bakiye yapısından mı?")
tab_confounding(load_data())
