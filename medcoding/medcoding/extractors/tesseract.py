"""Tesseract-based OCR extractor.

Reasonable baseline for CPU-only dev: ~1-3 sec per page on modern hardware.
Quality degrades on heavily noisy scans (the "complex" generator tier);
swap to PaddleOCR or a VLM in production if recall on those cases matters.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytesseract
from PIL import Image, ImageSequence

from ..models import BoundingBox, Page, TextBlock

log = logging.getLogger(__name__)


class TesseractExtractor:
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 6) -> None:
        # PSM 6 = "Assume a single uniform block of text." For scanned charts
        # this performs better than the default (PSM 3 / fully automatic) which
        # over-fragments tables.
        self._lang = lang
        self._psm = psm

    def extract(self, tiff_path: Path) -> list[Page]:
        tiff_path = Path(tiff_path)
        if not tiff_path.is_file():
            raise FileNotFoundError(tiff_path)

        pages: list[Page] = []
        with Image.open(tiff_path) as img:
            for idx, frame in enumerate(ImageSequence.Iterator(img)):
                page = self._extract_page(frame.copy(), page_number=idx + 1)
                pages.append(page)
                log.debug(
                    "tesseract page %d: %d blocks, %d chars",
                    page.page_number,
                    len(page.blocks),
                    len(page.full_text),
                )
        return pages

    def _extract_page(self, image: Image.Image, page_number: int) -> Page:
        # Convert grayscale or 1-bit modes to L for Tesseract.
        if image.mode not in ("L", "RGB"):
            image = image.convert("L")

        config = f"--psm {self._psm}"
        data = pytesseract.image_to_data(
            image,
            lang=self._lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        blocks: list[TextBlock] = []
        n = len(data["text"])
        # Group by (block_num, par_num, line_num) so we emit one TextBlock per
        # OCR line — keeps section-header detection simple downstream.
        groups: dict[tuple[int, int, int], list[int]] = {}
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            groups.setdefault(key, []).append(i)

        for indices in groups.values():
            words = [data["text"][i] for i in indices]
            confs = [float(data["conf"][i]) for i in indices]
            xs = [data["left"][i] for i in indices]
            ys = [data["top"][i] for i in indices]
            xs2 = [data["left"][i] + data["width"][i] for i in indices]
            ys2 = [data["top"][i] + data["height"][i] for i in indices]
            line_text = " ".join(words)
            blocks.append(
                TextBlock(
                    text=line_text,
                    bbox=BoundingBox(min(xs), min(ys), max(xs2), max(ys2)),
                    confidence=sum(confs) / len(confs) / 100.0,
                    page=page_number,
                )
            )

        # Sort top-to-bottom, then left-to-right within a row tolerance so
        # downstream readers see lines in human reading order.
        blocks.sort(key=lambda b: (b.bbox.y0 // 10, b.bbox.x0))

        full_text = "\n".join(b.text for b in blocks)
        w, h = image.size
        return Page(
            page_number=page_number,
            width=w,
            height=h,
            blocks=blocks,
            full_text=full_text,
        )
