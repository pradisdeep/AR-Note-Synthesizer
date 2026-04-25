"""
Phase 3 - Operational enrichment.

Reads data/categorized_output.csv and adds three columns the dashboard needs:

  - Current Balance ($)  : random per unique Account Number, $50 - $2000
  - Total Touches        : per account, 1 - 12; COB_ISSUE accounts skew higher
                           (mean ~9 vs ~6.5) so the Burnout Zone surfaces them
  - Labor Waste ($)      : Total Touches * $5.00

The transform is pure and deterministic given a seed so the dashboard renders
the same numbers across reloads. Importable as a function (used by app.py)
or runnable standalone to dump a CSV for inspection.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "categorized_output.csv"
OUTPUT_PATH = ROOT / "data" / "transformed_output.csv"

LABOR_RATE_PER_TOUCH = 5.00
BALANCE_MIN = 50
BALANCE_MAX = 2000
TOUCHES_MIN = 1
TOUCHES_MAX = 12

# COB_ISSUE accounts get a higher floor on touches so they cluster in the
# Burnout Zone (touches > 5 AND balance < $300) the dashboard highlights.
COB_TOUCHES_MIN = 6
COB_LABEL = "COB_ISSUE"

DEFAULT_SEED = 42


def _touch_count(label: str, rng: random.Random) -> int:
    """Sample a touch count, biasing COB_ISSUE accounts higher."""
    if label == COB_LABEL:
        return rng.randint(COB_TOUCHES_MIN, TOUCHES_MAX)
    return rng.randint(TOUCHES_MIN, TOUCHES_MAX)


def transform(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Append Current Balance, Total Touches, and Labor Waste ($) columns."""
    if "LLM_Root_Cause" not in df.columns:
        raise ValueError("Input dataframe is missing 'LLM_Root_Cause' - run Phase 2 first.")

    rng = random.Random(seed)
    out = df.copy()

    # Per-account values (one row per Account Number in this POC, but coded
    # as a per-account map so multi-touch accounts would still get one number).
    accounts = out["Account Number"].drop_duplicates().tolist()
    label_by_account = (
        out.drop_duplicates("Account Number")
        .set_index("Account Number")["LLM_Root_Cause"]
        .to_dict()
    )

    balances = {acct: round(rng.uniform(BALANCE_MIN, BALANCE_MAX), 2) for acct in accounts}
    touches = {acct: _touch_count(label_by_account.get(acct, ""), rng) for acct in accounts}

    out["Current Balance"] = out["Account Number"].map(balances)
    out["Total Touches"] = out["Account Number"].map(touches)
    out["Labor Waste ($)"] = (out["Total Touches"] * LABOR_RATE_PER_TOUCH).round(2)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 - operational enrichment")
    p.add_argument("--input", type=Path, default=INPUT_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"ERROR: {args.input} not found. Run src/claude_processor.py first.")

    df = pd.read_csv(args.input)
    enriched = transform(df, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)

    print(f"Wrote {len(enriched)} rows -> {args.output.relative_to(ROOT)}")
    print("\nMean touches by root cause:")
    summary = (
        enriched.groupby("LLM_Root_Cause")["Total Touches"]
        .agg(["mean", "count"])
        .round(2)
        .sort_values("mean", ascending=False)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
