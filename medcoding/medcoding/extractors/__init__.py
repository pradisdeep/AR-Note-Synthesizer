"""Pluggable extraction backends. Pick via Config.extractor."""

from .base import Extractor

__all__ = ["Extractor", "load_extractor"]


def load_extractor(name: str) -> Extractor:
    """Factory: resolve a backend by name. Adding a new backend means dropping
    a new module here and a new branch — no other code changes.
    """

    name = name.lower()
    if name == "tesseract":
        from .tesseract import TesseractExtractor

        return TesseractExtractor()
    raise ValueError(
        f"Unknown extractor {name!r}. Available: tesseract. "
        "PaddleOCR / VLM backends can be added under medcoding/extractors/."
    )
