"""Evaluate end-to-end coding accuracy against the synthetic chart manifest.

Runs full pipeline (extract -> normalize -> code) on each chart and scores
ICD-10 and CPT predictions against ground truth from manifest.jsonl.

Usage:
    python scripts/evaluate_coding.py path/to/manifest.jsonl
    python scripts/evaluate_coding.py path/to/manifest.jsonl --coder mock --limit 5

Confidence threshold: predictions below `--min-confidence` are dropped before
scoring (simulates the production rejection-to-human-review policy).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medcoding import load, load_coder, load_extractor, normalize  # noqa: E402

log = logging.getLogger("evaluate_coding")


@dataclass
class ChartScore:
    chart_id: str
    icd_truth: set[str]
    icd_predicted: set[str]
    cpt_truth: set[str]
    cpt_predicted: set[str]

    @property
    def icd_recall(self) -> float:
        return len(self.icd_truth & self.icd_predicted) / len(self.icd_truth) if self.icd_truth else 1.0

    @property
    def icd_precision(self) -> float:
        return len(self.icd_truth & self.icd_predicted) / len(self.icd_predicted) if self.icd_predicted else 0.0

    @property
    def cpt_recall(self) -> float:
        return len(self.cpt_truth & self.cpt_predicted) / len(self.cpt_truth) if self.cpt_truth else 1.0

    @property
    def cpt_precision(self) -> float:
        return len(self.cpt_truth & self.cpt_predicted) / len(self.cpt_predicted) if self.cpt_predicted else 0.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _build_coder(name: str, cfg):
    if name == "lm_studio":
        return load_coder(
            "lm_studio",
            base_url=cfg.lm_studio_base_url,
            api_key=cfg.lm_studio_api_key,
            model=cfg.lm_studio_model,
            temperature=cfg.lm_studio_temperature,
            max_tokens=cfg.lm_studio_max_tokens,
            request_timeout=cfg.lm_studio_timeout_s,
        )
    return load_coder(name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score end-to-end coding against manifest.jsonl.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--extractor", default=None)
    parser.add_argument("--coder", default=None)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Drop predictions below this confidence before scoring (simulates human-review threshold).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load()
    extractor = load_extractor(args.extractor or cfg.extractor)
    coder = _build_coder(args.coder or cfg.coder, cfg)

    manifest_dir = args.manifest.parent
    scores: list[ChartScore] = []

    with args.manifest.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if args.limit is not None and i >= args.limit:
                break
            record = json.loads(line)
            tiff_path = manifest_dir / record["tiff"]
            if not tiff_path.is_file():
                log.warning("Missing TIFF %s, skipping", tiff_path)
                continue

            log.info("[%d] %s", i, tiff_path.name)
            pages = extractor.extract(tiff_path)
            chart = normalize(pages, source_path=tiff_path, extractor_name=extractor.name)
            result = coder.code(chart)

            icd_truth = {d["code"].upper() for d in record.get("diagnoses", [])}
            cpt_truth = {p["code"].upper() for p in record.get("procedures", [])}
            icd_pred = {
                s.code for s in result.icd_suggestions if s.confidence >= args.min_confidence
            }
            cpt_pred = {
                s.code for s in result.cpt_suggestions if s.confidence >= args.min_confidence
            }

            score = ChartScore(
                chart_id=tiff_path.stem,
                icd_truth=icd_truth,
                icd_predicted=icd_pred,
                cpt_truth=cpt_truth,
                cpt_predicted=cpt_pred,
            )
            scores.append(score)

            log.info(
                "  ICD truth=%s pred=%s | CPT truth=%s pred=%s",
                sorted(icd_truth),
                sorted(icd_pred),
                sorted(cpt_truth),
                sorted(cpt_pred),
            )

    if not scores:
        print("No charts processed.", file=sys.stderr)
        return 1

    icd_p = sum(s.icd_precision for s in scores) / len(scores)
    icd_r = sum(s.icd_recall for s in scores) / len(scores)
    cpt_p = sum(s.cpt_precision for s in scores) / len(scores)
    cpt_r = sum(s.cpt_recall for s in scores) / len(scores)

    print()
    print(f"Charts processed: {len(scores)}")
    print(f"Extractor: {extractor.name} | Coder: {coder.name} | min_conf={args.min_confidence}")
    print()
    print(f"ICD-10  precision={icd_p:.3f}  recall={icd_r:.3f}  f1={_f1(icd_p, icd_r):.3f}")
    print(f"CPT     precision={cpt_p:.3f}  recall={cpt_r:.3f}  f1={_f1(cpt_p, cpt_r):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
