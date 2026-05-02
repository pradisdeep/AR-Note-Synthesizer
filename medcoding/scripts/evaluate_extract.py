"""Evaluate the extractor against the synthetic chart generator's manifest.

Reads the manifest.jsonl produced by synthetic-chart-generator and, for each
chart, runs extraction + normalization and scores ICD-10 / CPT recall + precision
against the ground-truth codes.

Usage:
    python scripts/evaluate_extract.py path/to/manifest.jsonl
    python scripts/evaluate_extract.py path/to/manifest.jsonl --limit 5 --verbose
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

from medcoding import load, load_extractor, normalize  # noqa: E402

log = logging.getLogger("evaluate_extract")


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


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Score extractor recall/precision against ground truth.")
    parser.add_argument("manifest", type=Path, help="Path to manifest.jsonl produced by the generator.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N charts.")
    parser.add_argument("--extractor", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load()
    extractor_name = args.extractor or cfg.extractor
    extractor = load_extractor(extractor_name)

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

            icd_truth = {d["code"].upper() for d in record.get("diagnoses", [])}
            cpt_truth = {p["code"].upper() for p in record.get("procedures", [])}
            icd_pred = {r.code for r in chart.icd_rows}
            cpt_pred = {r.code for r in chart.cpt_rows}

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
    print(f"Extractor: {extractor.name}")
    print()
    print(f"ICD-10  precision={icd_p:.3f}  recall={icd_r:.3f}  f1={_f1(icd_p, icd_r):.3f}")
    print(f"CPT     precision={cpt_p:.3f}  recall={cpt_r:.3f}  f1={_f1(cpt_p, cpt_r):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
