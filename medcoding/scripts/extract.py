"""CLI: TIFF -> structured Markdown.

Usage:
    python scripts/extract.py path/to/chart.tiff
    python scripts/extract.py path/to/chart.tiff --output chart.md --json chart.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medcoding import load, load_extractor, normalize  # noqa: E402


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a TIFF to structured Markdown.")
    parser.add_argument("tiff", type=Path, help="Path to a multi-page TIFF.")
    parser.add_argument("--output", type=Path, help="Write Markdown here (default: stdout).")
    parser.add_argument("--json", type=Path, help="Also write the full ExtractedChart as JSON.")
    parser.add_argument(
        "--extractor",
        default=None,
        help="Override config (default: from MEDCODING_EXTRACTOR or 'tesseract').",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load()
    extractor_name = args.extractor or cfg.extractor
    extractor = load_extractor(extractor_name)

    pages = extractor.extract(args.tiff)
    chart = normalize(
        pages,
        source_path=args.tiff,
        extractor_name=extractor.name,
    )

    if args.output:
        args.output.write_text(chart.markdown, encoding="utf-8")
        print(f"Wrote Markdown to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(chart.markdown)

    if args.json:
        # Pages can be huge; strip block-level detail when serializing.
        payload = {
            "source_path": chart.source_path,
            "extracted_at": chart.extracted_at.isoformat(),
            "extractor": chart.extractor,
            "page_count": len(chart.pages),
            "sections": [
                {"name": s.name, "title": s.title, "pages": s.pages, "text": s.text}
                for s in chart.sections
            ],
            "icd_rows": [asdict(r) for r in chart.icd_rows],
            "cpt_rows": [asdict(r) for r in chart.cpt_rows],
            "extractor_metadata": chart.extractor_metadata,
        }
        args.json.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        print(f"Wrote JSON to {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
