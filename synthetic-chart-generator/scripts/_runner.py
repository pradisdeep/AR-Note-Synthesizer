"""Shared batch runner used by the small/medium/complex entry points."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

# Allow `python scripts/generate_*.py` to find the src/ package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chart_generator import (  # noqa: E402
    DegradationProfile,
    build_pdf,
    export_tiff,
    generate_chart_data,
)


def parse_args(level: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Generate synthetic clinical charts at the '{level}' complexity level."
    )
    parser.add_argument("--count", type=int, default=5, help="Number of charts to generate.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed; per-chart seeds are derived as base + index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / f"{level}_charts",
        help="Directory to write TIFFs and metadata into.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Rasterization DPI.")
    parser.add_argument(
        "--keep-pdf",
        action="store_true",
        help="Also keep the source PDF next to each TIFF.",
    )
    return parser.parse_args()


def run(
    *,
    level: str,
    profile: DegradationProfile,
    chart_kwargs: dict,
    args: argparse.Namespace,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for i in range(args.count):
            chart_seed = (args.seed + i) if args.seed is not None else None
            chart = generate_chart_data(seed=chart_seed, **chart_kwargs)

            stem = f"{level}_chart_{i:04d}"
            tiff_path = args.output_dir / f"{stem}.tiff"

            with tempfile.TemporaryDirectory() as tmp:
                tmp_pdf = Path(tmp) / f"{stem}.pdf"
                build_pdf(chart, tmp_pdf)

                page_profile = DegradationProfile(**{**profile.__dict__, "seed": chart_seed})
                export_tiff(tmp_pdf, tiff_path, page_profile, dpi=args.dpi)

                if args.keep_pdf:
                    kept = args.output_dir / f"{stem}.pdf"
                    kept.write_bytes(tmp_pdf.read_bytes())

            manifest.write(
                json.dumps(
                    {
                        "level": level,
                        "tiff": str(tiff_path.relative_to(args.output_dir)),
                        "seed": chart_seed,
                        "patient_mrn": chart.patient["mrn"],
                        "encounter_id": chart.encounter["encounter_id"],
                        "diagnoses": chart.diagnoses,
                        "procedures": chart.procedures,
                        "profile": asdict(profile),
                    }
                )
                + "\n"
            )
            print(f"[{level}] {tiff_path}")

    print(f"Wrote manifest to {manifest_path}")
