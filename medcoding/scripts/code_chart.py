"""End-to-end CLI: TIFF -> ICD-10 + CPT codes.

Runs Stage 2 (extract) -> Stage 3 (normalize) -> Stage 5 (code).

Usage:
    python scripts/code_chart.py path/to/chart.tiff
    python scripts/code_chart.py path/to/chart.tiff --coder mock
    python scripts/code_chart.py path/to/chart.tiff --output result.json
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

from medcoding import load, load_coder, load_extractor, normalize  # noqa: E402

log = logging.getLogger("code_chart")


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def _build_coder(coder_name: str, cfg):
    if coder_name == "lm_studio":
        return load_coder(
            "lm_studio",
            base_url=cfg.lm_studio_base_url,
            api_key=cfg.lm_studio_api_key,
            model=cfg.lm_studio_model,
            temperature=cfg.lm_studio_temperature,
            max_tokens=cfg.lm_studio_max_tokens,
            request_timeout=cfg.lm_studio_timeout_s,
        )
    if coder_name == "mock":
        return load_coder("mock")
    return load_coder(coder_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="TIFF -> ICD-10 + CPT codes via the configured coder.")
    parser.add_argument("tiff", type=Path, help="Path to a multi-page TIFF.")
    parser.add_argument("--extractor", default=None, help="Override extractor (default: config).")
    parser.add_argument("--coder", default=None, help="Override coder (default: config).")
    parser.add_argument("--output", type=Path, help="Write CodingResult JSON here.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load()
    extractor = load_extractor(args.extractor or cfg.extractor)
    coder = _build_coder(args.coder or cfg.coder, cfg)

    log.info("Extracting %s with %s", args.tiff, extractor.name)
    pages = extractor.extract(args.tiff)
    chart = normalize(pages, source_path=args.tiff, extractor_name=extractor.name)
    log.info("Normalized %d pages into %d sections", len(chart.pages), len(chart.sections))

    log.info("Coding with %s", coder.name)
    result = coder.code(chart)

    print()
    print(f"=== {args.tiff.name} ===")
    print(f"Coder: {result.coder}  ({result.coder_metadata})")
    print()
    print("ICD-10:")
    if not result.icd_suggestions:
        print("  (none)")
    for s in result.icd_suggestions:
        print(f"  {s.code}  conf={s.confidence:.2f}  {s.description}")
        if s.evidence_quote:
            print(f"    evidence: {s.evidence_quote[:120]}")
    print()
    print("CPT:")
    if not result.cpt_suggestions:
        print("  (none)")
    for s in result.cpt_suggestions:
        print(f"  {s.code}  conf={s.confidence:.2f}  {s.description}")
        if s.evidence_quote:
            print(f"    evidence: {s.evidence_quote[:120]}")

    if args.output:
        payload = {
            "chart_source": result.chart_source,
            "coder": result.coder,
            "coded_at": result.coded_at.isoformat(),
            "coder_metadata": result.coder_metadata,
            "icd_suggestions": [asdict(s) for s in result.icd_suggestions],
            "cpt_suggestions": [asdict(s) for s in result.cpt_suggestions],
            "raw_responses": result.raw_responses,
        }
        args.output.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        log.info("Wrote result to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
