"""Model Karşılaştırma sayfası (faz 1: eski sekme içeriği, ortak tema ile)."""
import iris_ui as ui
from iris_data import load_artifacts, load_data  # noqa: F401
from iris_legacy import tab_karsilastirma

ui.page_header("Analiz", "Model Karşılaştırma", "Test seti metrikleri, ROC/PR eğrileri ve McNemar testi")
tab_karsilastirma(load_artifacts())
