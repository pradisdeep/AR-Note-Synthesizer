"""
Phase 3 - Operational enrichment (multi-note edition).

Reads data/categorized_output.csv (per-account, with terminal + journey labels)
and adds the columns the dashboard needs:

  - Current Balance ($)  : random per account, $50 - $2000
  - Total Touches        : derived from journey length AND terminal cause
                            * each stage adds 2-4 base touches
                            * COB_ISSUE terminals add an extra 1-2 (always
                              the most labor-intensive workflow in practice)
                            * capped at 12 to keep the X-axis comparable
                            * single-cause simple claims stay low (2-5)
                            * 4-stage cascades cluster near the cap (10-12)
  - Labor Waste ($)      : Total Touches * $5.00

Touch generation is deterministic given a seed so the dashboard renders the
same numbers across reloads. Importable as transform() (used by app.py) or
runnable standalone to dump a CSV for inspection.
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

PER_STAGE_TOUCHES_MIN = 2
PER_STAGE_TOUCHES_MAX = 4
COB_TERMINAL_BONUS_MIN = 1
COB_TERMINAL_BONUS_MAX = 2

COB_LABEL = "COB_ISSUE"
DEFAULT_SEED = 42


def _touch_count(journey_length: int, terminal: str, rng: random.Random) -> int:
    """Touches scale with journey length; COB terminals add extra labor."""
    stages = max(1, int(journey_length))
    total = sum(rng.randint(PER_STAGE_TOUCHES_MIN, PER_STAGE_TOUCHES_MAX) for _ in range(stages))
    if terminal == COB_LABEL:
        total += rng.randint(COB_TERMINAL_BONUS_MIN, COB_TERMINAL_BONUS_MAX)
    return max(TOUCHES_MIN, min(TOUCHES_MAX, total))


def transform(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Append Current Balance, Total Touches, and Labor Waste ($) columns.

    Expects a per-account dataframe with at least:
        Account Number, LLM_Terminal_Root_Cause, LLM_Journey_Length
    """
    required = {"Account Number", "LLM_Terminal_Root_Cause", "LLM_Journey_Length"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe is missing columns: {sorted(missing)} - run Phase 2 first.")

    rng = random.Random(seed)
    out = df.copy()

    out["Current Balance"] = [
        round(rng.uniform(BALANCE_MIN, BALANCE_MAX), 2) for _ in range(len(out))
    ]
    out["Total Touches"] = [
        _touch_count(int(out.at[i, "LLM_Journey_Length"]), out.at[i, "LLM_Terminal_Root_Cause"], rng)
        for i in out.index
    ]
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

    print(f"Wrote {len(enriched)} accounts -> {args.output.relative_to(ROOT)}")
    print("\nMean touches by terminal root cause:")
    by_cause = (
        enriched.groupby("LLM_Terminal_Root_Cause")["Total Touches"]
        .agg(["mean", "count"])
        .round(2)
        .sort_values("mean", ascending=False)
    )
    print(by_cause.to_string())
    print("\nMean touches by journey length:")
    by_len = (
        enriched.groupby("LLM_Journey_Length")["Total Touches"]
        .agg(["mean", "count"])
        .round(2)
        .sort_index()
    )
    print(by_len.to_string())


if __name__ == "__main__":
    main()
