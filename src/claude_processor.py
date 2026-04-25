"""
Phase 2 - Claude-3-Haiku root-cause synthesis.

Reads data/synthetic_pms_extract.csv, sends each Notes value to the Anthropic
Messages API (claude-3-haiku-20240307) with a strict, few-shot XML system
prompt, and writes the result back as an LLM_Root_Cause column in
data/categorized_output.csv.

The model is locked down with max_tokens=10 and temperature=0.0 and is required
to emit exactly one taxonomy token. Anything off-taxonomy is coerced to
UNCLEAR_NOTE so the downstream dashboard never sees a stray label.

Usage:
    python src/claude_processor.py
    python src/claude_processor.py --limit 5     # smoke test on 5 rows
    python src/claude_processor.py --resume      # skip rows already labeled
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

MODEL = "claude-3-haiku-20240307"
MAX_TOKENS = 10
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
You read short, abbreviation-heavy biller notes and assign one root-cause label.
</role>

<task>
Identify the TERMINAL root cause of why the claim is currently stalled.
Ignore interim follow-up actions (calls, faxes, voicemails, escalations).
The terminal root cause is the underlying payor/provider problem that must be
fixed for the claim to move forward.
</task>

<taxonomy>
<label name="MISSING_AUTH">Prior authorization / pre-certification was not obtained, expired, or is invalid. Includes retro-auth requests and CO-197 denials.</label>
<label name="COB_ISSUE">Coordination of Benefits is outdated, incorrect, or unresolved. Includes "other primary insurance", suspended for COB, or pt needs to update payor.</label>
<label name="CLINICAL_REQUIRED">Payor needs medical records, clinical documentation, op notes, progress notes, or chart notes to adjudicate.</label>
<label name="CREDENTIALING_ERROR">Provider is not credentialed / not par on payor panel, NPI or taxonomy mismatch, enrollment pending, or rendering provider not loaded.</label>
<label name="TIMELY_FILING">Claim was denied or at risk because it exceeded the payor's filing window. Includes CO-29 denials and proof-of-timely-filing appeals.</label>
<label name="UNCLEAR_NOTE">The note is ambiguous and does not clearly map to any category above.</label>
</taxonomy>

<rules>
- Output EXACTLY ONE token from the taxonomy above.
- Uppercase letters and underscores only.
- No quotes, no punctuation, no explanation, no XML tags, no whitespace before or after.
</rules>

<examples>
<example>
<note>called UHC re DOS 3/14 - rep says no auth on file for E11.9, denl CO-197. need retro auth, sent rqst to prov ofc</note>
<answer>MISSING_AUTH</answer>
</example>
<example>
<note>Aetna rep states COB outdated, last update 2yrs ago. pt has new MCR primary now. M54.5 svc on hold</note>
<answer>COB_ISSUE</answer>
</example>
<example>
<note>BCBS req medical records for DOS 2/8 J45.909 - faxed rqst to clinic, awaiting recd</note>
<answer>CLINICAL_REQUIRED</answer>
</example>
<example>
<note>Medicare Primary denl - prov NPI not effective on DOS. cred dept says enrollment still pending I10</note>
<answer>CREDENTIALING_ERROR</answer>
</example>
<example>
<note>Medicaid denl CO-29 timely filing exceeded for DOS 1/3 E11.9. checking proof of timely submission</note>
<answer>TIMELY_FILING</answer>
</example>
<example>
<note>left vm pt, no answer, will try again tomorrow</note>
<answer>UNCLEAR_NOTE</answer>
</example>
</examples>"""


def classify(client: Anthropic, note: str) -> str:
    """Send one note to Claude Haiku and return a normalized taxonomy label."""
    user_msg = f"<note>{note}</note>"
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except (APIError, APIStatusError) as exc:
        print(f"  ! API error: {exc}", file=sys.stderr)
        return FALLBACK_LABEL

    raw = "".join(block.text for block in resp.content if block.type == "text")
    label = raw.strip().upper().strip(".,:;\"'<>/ \t\n")
    return label if label in VALID_LABELS else FALLBACK_LABEL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 - Claude Haiku root-cause synthesis")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for smoke testing).")
    p.add_argument(
        "--resume",
        action="store_true",
        help="If categorized_output.csv already exists, skip rows whose LLM_Root_Cause is set.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("sk-ant-your-key"):
        sys.exit("ERROR: ANTHROPIC_API_KEY is missing or still the placeholder. Edit .env and try again.")

    if not INPUT_PATH.exists():
        sys.exit(f"ERROR: {INPUT_PATH} not found. Run src/generate_pms_data.py first.")

    df = pd.read_csv(INPUT_PATH)

    if args.resume and OUTPUT_PATH.exists():
        prior = pd.read_csv(OUTPUT_PATH)
        if "LLM_Root_Cause" in prior.columns and len(prior) == len(df):
            df["LLM_Root_Cause"] = prior["LLM_Root_Cause"]
            print(f"Resume: loaded {df['LLM_Root_Cause'].notna().sum()} prior labels")
        else:
            df["LLM_Root_Cause"] = pd.NA
    else:
        df["LLM_Root_Cause"] = pd.NA

    if args.limit:
        target_idx = df.head(args.limit).index
    else:
        target_idx = df.index

    client = Anthropic(api_key=api_key)
    started = time.time()
    processed = 0

    for i, idx in enumerate(target_idx, start=1):
        if args.resume and pd.notna(df.at[idx, "LLM_Root_Cause"]):
            continue
        note = str(df.at[idx, "Notes"])
        label = classify(client, note)
        df.at[idx, "LLM_Root_Cause"] = label
        processed += 1
        if i % 10 == 0 or i == len(target_idx):
            elapsed = time.time() - started
            print(f"  [{i:>3}/{len(target_idx)}] last={label:<20} elapsed={elapsed:5.1f}s")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    counts = df["LLM_Root_Cause"].value_counts(dropna=False)
    print(f"\nProcessed {processed} note(s). Wrote -> {OUTPUT_PATH.relative_to(ROOT)}")
    print("Label distribution:")
    for label, n in counts.items():
        print(f"  {label:<20} {n}")


if __name__ == "__main__":
    main()
