"""Synthetic clinical chart generator package.

Pipeline: data_generator -> document_builder -> image_degrader -> tiff_exporter.
"""

from .data_generator import ChartData, generate_chart_data
from .document_builder import build_pdf
from .image_degrader import DegradationProfile, degrade_image
from .tiff_exporter import export_tiff

__all__ = [
    "ChartData",
    "generate_chart_data",
    "build_pdf",
    "DegradationProfile",
    "degrade_image",
    "export_tiff",
]
