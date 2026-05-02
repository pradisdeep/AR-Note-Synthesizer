"""Synthetic clinical chart generator package.

Pipeline: data_generator -> document_builder -> image_degrader -> tiff_exporter.
"""

from .data_generator import ChartData, generate_chart_data
from .document_builder import build_pdf
from .image_degrader import DegradationProfile, degrade_image
from .noise_sections import NoiseRow, NoiseSection, generate_noise_sections
from .tiff_exporter import export_tiff

__all__ = [
    "ChartData",
    "DegradationProfile",
    "NoiseRow",
    "NoiseSection",
    "build_pdf",
    "degrade_image",
    "export_tiff",
    "generate_chart_data",
    "generate_noise_sections",
]
