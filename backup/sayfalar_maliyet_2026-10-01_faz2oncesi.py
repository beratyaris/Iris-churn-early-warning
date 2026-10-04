"""Maliyet ve Eşik sayfası (faz 1: eski sekme içeriği, ortak tema ile)."""
import iris_ui as ui
from iris_data import load_artifacts, load_data  # noqa: F401
from iris_legacy import tab_maliyet

ui.page_header("Karar Destek", "Maliyet ve Eşik", "Maliyete duyarlı karar eşiği (validation → test)")
tab_maliyet(load_artifacts())
