"""
Phase 2 - Claude Haiku per-account denial-journey synthesis.

Reads data/synthetic_pms_extract.csv (multi-note edition), groups all notes for
each Account Number in chronological order, and sends the full sequence to
claude-haiku-4-5 (model claude-3-haiku-20240307 was retired Apr 2026) in a
single call per account. The model returns:

  * LLM_Terminal_Root_Cause - the cause currently blocking the claim
  * LLM_Denial_Journey      - distinct causes the claim has cycled through,
                              ordered by first appearance
  * LLM_Journey_Length      - count of distinct causes in the journey

Output: data/categorized_output.csv at one row per account, with the original
fields collapsed to per-account values (latest Status, latest User Name, etc.)
plus a Notes column containing the full chronological note history joined
with date markers.

Usage:
    python src/claude_processor.py
    python src/claude_processor.py --limit 5      # cheap smoke test
    python src/claude_processor.py --resume       # skip already-labeled accounts
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic, APIError, APIStatusError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "synthetic_pms_extract.csv"
OUTPUT_PATH = ROOT / "data" / "categorized_output.csv"

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 80
TEMPERATURE = 0.0

VALID_LABELS = {
    "MISSING_AUTH",
    "COB_ISSUE",
    "CLINICAL_REQUIRED",
    "CREDENTIALING_ERROR",
    "TIMELY_FILING",
    "UNCLEAR_NOTE",
}
FALLBACK_LABEL = "UNCLEAR_NOTE"

SYSTEM_PROMPT = """<role>
You are a healthcare Revenue Cycle Management (RCM) classification engine.
You read the FULL chronological note history for one account and identify
the denial journey the claim has been through.
</role>

<input_format>
The user message contains all biller notes for ONE account, in chronological
order. Each note is on its own line, prefixed with [YYYY-MM-DD]. Notes use
biller shorthand (DOS, EOB, COB, TFL, retro auth, peer-to-peer, etc).
</input_format>

<task>
Identify TWO things:

1. TERMINAL root cause: what is blocking this claim RIGHT NOW. Read the most
   recent (last) note - that is the current blocker. If a problem was resolved
   in an earlier note and a new problem surfaced after it, the new problem is
   terminal, not the resolved one.

2. JOURNEY: the ordered list of DISTINCT root causes the claim has cycled
   through, in order of first appearance. If the same cause appears in two
   notes (recurring or unresolved), include it once. Always include the
   terminal cause as the last entry of the journey.
</task>

<taxonomy>
<label name="MISSING_AUTH">Prior authorization / pre-cert was not obtained, expired, or invalid. Includes retro-auth requests, CO-197, peer-to-peer reviews.</label>
<label name="COB_ISSUE">Coordination of Benefits is outdated, incorrect, or unresolved. Includes other-primary-insurance, suspended for COB, pt needs to update payor.</label>
<label name="CLINICAL_REQUIRED">Payor needs medical records, op notes, progress notes, or chart notes to adjudicate or support an appeal.</label>
<label name="CREDENTIALING_ERROR">Provider not credentialed / not par on payor panel, NPI or taxonomy mismatch, enrollment pending.</label>
<label name="TIMELY_FILING">Claim denied or at risk for exceeding the payor's filing window. Includes CO-29 and TFL appeals with proof of timely filing.</label>
<label name="UNCLEAR_NOTE">Notes are ambiguous and do not map to any category above.</label>
</taxonomy>

<output_format>
Output EXACTLY one line in this format and nothing else:

TERMINAL|CAUSE1,CAUSE2,...

Where TERMINAL is one taxonomy token and CAUSE1,CAUSE2,... is the journey -
distinct causes in chronological order of first appearance, ending with the
terminal. Use only the six taxonomy tokens. Uppercase, underscores. No quotes,
no explanation, no extra whitespace.
</output_format>

<examples>
<example>
<input>[2026-01-15] called UHC re DOS 1/3 - rep says no auth on file for E11.9, denl CO-197. need retro auth, sent rqst to prov ofc</input>
<output>MISSING_AUTH|MISSING_AUTH</output>
</example>
<example>
<input>[2025-12-01] denl - Aetna needs COB info refreshed, suspended pending other ins verif. f/u 5d J45.909
[2025-12-22] after pt updated other ins, Aetna now wants auth for J45.909 - none was obtained, requesting retro</input>
<output>MISSING_AUTH|COB_ISSUE,MISSING_AUTH</output>
</example>
<example>
<input>[2025-11-08] Medicare Primary - no precert obtained for DOS 11/2 E11.9, attempting peer to peer review
[2025-12-04] peer to peer w/ Medicare Primary req clinical justification for E11.9 - chasing prov for chart notes
[2026-01-20] auth obtained but cred check shows prov terming on Medicare Primary panel - escalated to cred dept E11.9
[2026-02-25] cred fix took 90d - Medicare Primary now denying claim as TFL. building appeal w/ acceptance report E11.9</input>
<output>TIMELY_FILING|MISSING_AUTH,CLINICAL_REQUIRED,CREDENTIALING_ERROR,TIMELY_FILING</output>
</example>
<example>
<input>[2026-02-10] left vm pt, no answer, will try again tomorrow</input>
<output>UNCLEAR_NOTE|UNCLEAR_NOTE</output>
</example>
</examples>"""


def _format_account_notes(account_df: pd.DataFrame) -> str:
    """Concatenate one account's notes chronologically with date prefixes."""
    ordered = account_df.sort_values("User Touch Date")
    return "\n".join(
        f"[{row['User Touch Date']}] {row['Notes']}"
        for _, row in ordered.iterrows()
    )


def _parse_response(raw: str) -> tuple[str, list[str]]:
    """Parse 'TERMINAL|CAUSE1,CAUSE2,...' into (terminal, journey list)."""
    cleaned = raw.strip().upper().strip(".,:;\"'<>/ \t\n")
    if "|" not in cleaned:
        return FALLBACK_LABEL, [FALLBACK_LABEL]

    terminal_raw, journey_raw = cleaned.split("|", 1)
    terminal = terminal_raw.strip()
    journey = [c.strip() for c in journey_raw.split(",") if c.strip()]

    terminal = terminal if terminal in VALID_LABELS else FALLBACK_LABEL
    journey = [c if c in VALID_LABELS else FALLBACK_LABEL for c in journey]
    if not journey:
        journey = [terminal]

    return terminal, journey


def classify_account(client: Anthropic, account_notes: str) -> tuple[str, list[str]]:
    """Send one account's full note history to Claude Haiku."""
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": account_notes}],
        )
    except (APIError, APIStatusError) as exc:
        print(f"  ! API error: {exc}", file=sys.stderr)
        return FALLBACK_LABEL, [FALLBACK_LABEL]

    raw = "".join(b.text for b in resp.content if b.type == "text")
    return _parse_response(raw)


def aggregate_to_account_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the multi-note dataframe into one row per Account Number."""
    df = df.sort_values(["Account Number", "User Touch Date"])
    grouped = df.groupby("Account Number", sort=False)

    accounts = []
    for account_id, group in grouped:
        latest = group.iloc[-1]
        first = group.iloc[0]
        notes_combined = "\n".join(
            f"[{row['User Touch Date']}] {row['Notes']}"
            for _, row in group.iterrows()
        )
        accounts.append({
            "Account Number": account_id,
            "Payor Name": latest["Payor Name"],
            "Primary DX": latest["Primary DX"],
            "First Touch Date": first["User Touch Date"],
            "Last Touch Date": latest["User Touch Date"],
            "Total Notes": len(group),
            "Latest Status": latest["Status"],
            "Latest Sub-status": latest["Sub-status"],
            "Latest User": latest["User Name"],
            "Next Followup Date": latest["Next Followup Date"],
            "Notes": notes_combined,
        })

    return pd.DataFrame(accounts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 - per-account denial-journey synthesis")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N accounts (smoke testing).")
    p.add_argument("--resume", action="store_true", help="Skip accounts already labeled in categorized_output.csv.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("sk-ant-your-key"):
        sys.exit("ERROR: ANTHROPIC_API_KEY missing or placeholder. Edit .env and try again.")

    if not INPUT_PATH.exists():
        sys.exit(f"ERROR: {INPUT_PATH} not found. Run src/generate_pms_data.py first.")

    raw_df = pd.read_csv(INPUT_PATH)
    account_df = aggregate_to_account_rows(raw_df)

    # Initialise label columns
    account_df["LLM_Terminal_Root_Cause"] = pd.NA
    account_df["LLM_Denial_Journey"] = pd.NA
    account_df["LLM_Journey_Length"] = pd.NA

    if args.resume and OUTPUT_PATH.exists():
        prior = pd.read_csv(OUTPUT_PATH)
        if {"Account Number", "LLM_Terminal_Root_Cause"}.issubset(prior.columns):
            merged = account_df.merge(
                prior[["Account Number", "LLM_Terminal_Root_Cause", "LLM_Denial_Journey", "LLM_Journey_Length"]],
                on="Account Number",
                how="left",
                suffixes=("", "_prior"),
            )
            for col in ["LLM_Terminal_Root_Cause", "LLM_Denial_Journey", "LLM_Journey_Length"]:
                merged[col] = merged[f"{col}_prior"].combine_first(merged[col])
                merged.drop(columns=[f"{col}_prior"], inplace=True)
            account_df = merged
            print(f"Resume: loaded {account_df['LLM_Terminal_Root_Cause'].notna().sum()} prior labels")

    target_idx = account_df.head(args.limit).index if args.limit else account_df.index

    client = Anthropic(api_key=api_key)
    started = time.time()
    processed = 0

    for i, idx in enumerate(target_idx, start=1):
        if args.resume and pd.notna(account_df.at[idx, "LLM_Terminal_Root_Cause"]):
            continue
        notes_blob = account_df.at[idx, "Notes"]
        terminal, journey = classify_account(client, notes_blob)

        account_df.at[idx, "LLM_Terminal_Root_Cause"] = terminal
        account_df.at[idx, "LLM_Denial_Journey"] = ",".join(journey)
        account_df.at[idx, "LLM_Journey_Length"] = len(journey)
        processed += 1

        if i % 10 == 0 or i == len(target_idx):
            elapsed = time.time() - started
            print(f"  [{i:>3}/{len(target_idx)}] last_terminal={terminal:<20} elapsed={elapsed:5.1f}s")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    account_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nProcessed {processed} account(s). Wrote -> {OUTPUT_PATH.relative_to(ROOT)}")
    print("\nTerminal root cause distribution:")
    for label, n in account_df["LLM_Terminal_Root_Cause"].value_counts(dropna=False).items():
        print(f"  {label:<20} {n}")
    print("\nJourney length distribution:")
    for length, n in account_df["LLM_Journey_Length"].value_counts(dropna=False).sort_index().items():
        print(f"  length {length:<3} {n}")


if __name__ == "__main__":
    main()
