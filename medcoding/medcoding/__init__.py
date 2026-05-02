"""Medical-coding pipeline: TIFF -> structured Markdown -> ICD/CPT codes."""

from .chunker import CodingChunk, chunk_for_coding
from .coders import load_coder
from .config import Config, load
from .extractors import load_extractor
from .models import (
    BoundingBox,
    CodeRow,
    CodeSuggestion,
    CodeType,
    CodingResult,
    ExtractedChart,
    Page,
    Section,
    TextBlock,
)
from .normalizer import normalize

__all__ = [
    "BoundingBox",
    "CodeRow",
    "CodeSuggestion",
    "CodeType",
    "CodingChunk",
    "CodingResult",
    "Config",
    "ExtractedChart",
    "Page",
    "Section",
    "TextBlock",
    "chunk_for_coding",
    "load",
    "load_coder",
    "load_extractor",
    "normalize",
]
