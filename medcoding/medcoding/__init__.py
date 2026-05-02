"""Medical-coding pipeline: TIFF -> structured Markdown -> ICD/CPT codes."""

from .config import Config, load
from .extractors import load_extractor
from .models import (
    BoundingBox,
    CodeRow,
    ExtractedChart,
    Page,
    Section,
    TextBlock,
)
from .normalizer import normalize

__all__ = [
    "BoundingBox",
    "CodeRow",
    "Config",
    "ExtractedChart",
    "Page",
    "Section",
    "TextBlock",
    "load",
    "load_extractor",
    "normalize",
]
