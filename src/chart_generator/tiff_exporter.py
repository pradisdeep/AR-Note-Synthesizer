"""Convert a PDF (rendered chart) into a degraded multi-page TIFF.

Uses pdf2image (Poppler) to rasterize each page, runs each page through
the degradation pipeline, then packages the result into a single
multi-page Group 4-compressed TIFF for OCR-friendly storage.
"""

from __future__ import annotations

from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from .image_degrader import DegradationProfile, degrade_image


def export_tiff(
    pdf_path: Path,
    tiff_path: Path,
    profile: DegradationProfile,
    *,
    dpi: int = 200,
    compression: str = "tiff_lzw",
) -> Path:
    """Rasterize `pdf_path`, degrade each page, write to a multi-page TIFF.

    `compression`: "tiff_lzw" keeps grayscale (preferred for OCR test
    fidelity); "group4" forces a 1-bit fax-like output if the caller
    explicitly wants the smallest, most fax-realistic artifact.
    """

    pdf_path = Path(pdf_path)
    tiff_path = Path(tiff_path)
    tiff_path.parent.mkdir(parents=True, exist_ok=True)

    pages = convert_from_path(str(pdf_path), dpi=dpi)
    if not pages:
        raise RuntimeError(f"pdf2image produced no pages for {pdf_path}")

    degraded: list[Image.Image] = []
    for i, page in enumerate(pages):
        # Stagger the per-page seed so each page gets its own rotation/noise
        # but the overall document remains reproducible from profile.seed.
        page_profile = DegradationProfile(**{**profile.__dict__})
        if profile.seed is not None:
            page_profile.seed = profile.seed + i
        degraded.append(degrade_image(page, page_profile))

    if compression == "group4":
        degraded = [img.convert("1") for img in degraded]
        save_kwargs = {"compression": "group4"}
    else:
        save_kwargs = {"compression": compression}

    first, rest = degraded[0], degraded[1:]
    first.save(
        tiff_path,
        format="TIFF",
        save_all=True,
        append_images=rest,
        **save_kwargs,
    )
    return tiff_path
