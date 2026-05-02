"""Extractor protocol. Every backend (OCR, VLM) implements this."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..models import Page


class Extractor(Protocol):
    """Convert a multi-page TIFF to a list of Page records.

    The protocol is intentionally narrow: input is a path, output is text +
    layout. Section reconstruction lives in the normalizer, not here, so
    OCR backends and VLM backends are interchangeable.
    """

    name: str

    def extract(self, tiff_path: Path) -> list[Page]: ...
