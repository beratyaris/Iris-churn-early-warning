"""Müşteri Risk Skoru sayfası (faz 1: eski sekme içeriği, ortak tema ile)."""
import iris_ui as ui
from iris_data import load_artifacts, load_data  # noqa: F401
from iris_legacy import tab_risk

ui.page_header("Karar Destek", "Müşteri Risk Skoru", "Tek müşterinin churn olasılığı ve SHAP açıklaması")
tab_risk(load_artifacts())
