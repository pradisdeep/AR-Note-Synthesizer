"""
Phase 2 - Claude Haiku per-account journey synthesis (physician POC).

Input: per-account timelines from ingest.build_account_timelines() - a
chronological interleaving of ledger events (Charges, Payment, Denial,
Adjustment with $ amounts and CPT codes) and biller notes.

Output: one structured record per account with:
  * terminal_root_cause - what is blocking the claim RIGHT NOW
  * denial_journey      - distinct causes the claim cycled through
  * anomaly_flags       - operational red flags (incorrect write-offs, etc.)
  * narrative           - one-sentence executive summary

Differs from the root-level POC processor:
  * Multi-source input (ledger + notes), not notes-only.
  * Broader taxonomy reflecting physician-billing reality.
  * Anomaly flag dimension - the ledger lets us catch operational issues
    that pure-notes input cannot (e.g. an Adjustment equal to Charges
    posted before any Payment = likely incorrect write-off).
  * JSON output instead of pipe-format - we need to return more fields
    than the original two.

Usage:
    # Dry run - format prompts and print the first one, no API call.
    python physician_poc/src/claude_processor.py \
        --events physician_poc/data/events_redacted.csv \
        --notes  physician_poc/data/notes_redacted.csv \
        --dry-run --limit 3

    # Live run - hits Anthropic API.
    python physician_poc/src/claude_processor.py \
        --events physician_poc/data/events_redacted.csv \
        --notes  physician_poc/data/notes_redacted.csv \
        --output physician_poc/data/synthesized.csv

    # Self-test against synthetic fixtures (offline, no API call).
    python physician_poc/src/claude_processor.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# anthropic + dotenv are only needed for live runs; selftest and --dry-run
# must work without them installed.
try:
    from anthropic import Anthropic, APIError, APIStatusError  # type: ignore
except ImportError:  # pragma: no cover
    Anthropic = None  # type: ignore[assignment]
    APIError = APIStatusError = Exception  # type: ignore[assignment, misc]

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore[no-redef]
        return False

# Local import - support running as `python physician_poc/src/claude_processor.py`
# and as `python -m physician_poc.src.claude_processor`.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from ingest import TimelineEvent, build_account_timelines  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 400
TEMPERATURE = 0.0

# --- Taxonomy --------------------------------------------------------------
# Terminal / journey labels. Physician-billing-specific extension of the
# original POC's six-label set. Tokens are stable identifiers; rename only
# with a migration of saved synthesis output.
VALID_LABELS = frozenset({
    "REGISTRATION_ELIGIBILITY",   # eligibility not verified, member-id wrong
    "COB_ISSUE",                  # other primary insurance / suspended for COB
    "MEDICAL_NECESSITY",          # payor disputes clinical necessity / LOC
    "BUNDLED_DENIAL",             # CPT bundled into another, NCCI edits
    "TIMELY_FILING",              # CO-29 / past filing window
    "ADDITIONAL_DOCUMENTATION_NEEDED",  # records / op-notes / itemized bill
    "CODING_ERROR",               # wrong CPT/ICD/modifier, needs rebill
    "AUTH_REQUIRED",              # prior auth / pre-cert missing or expired
    "UNDERPAYMENT",               # paid but below contracted rate
    "DRG_DOWNGRADE",              # payor downgraded DRG vs billed
    "WC_DENIED_BY_ADJUSTOR",      # workers' comp adjustor denied claim
    "DUPLICATE_CLAIM",            # CO-18 / duplicate submission
    "CLAIM_RECOUPED",             # payor took back a previous payment
    "BILLED_TO_WRONG_PAYOR",      # claim sent to wrong primary
    "AWAITING_RESPONSE",          # claim pending; no denial yet, biller waiting
    "RESOLVED_PAID",              # claim was paid; account is closed/closing
    "RESOLVED_WRITEOFF",          # legitimate adjustment / write-off
    "UNCLEAR_NOTE",               # cannot map confidently
})
FALLBACK_LABEL = "UNCLEAR_NOTE"

# Anomaly flags - operational red flags surfaced from the ledger+notes mix.
# These are independent of the journey labels; an account can have several.
VALID_FLAGS = frozenset({
    "INCORRECT_WRITEOFF",        # Adjustment >= Charges posted before any Payment
    "WRITEOFF_LIKELY_PR",        # Adjustment booked but balance was patient-resp
    "MASS_REGELIG_DENIAL",       # multiple CPT lines all denied Reg/Eligibility
    "ADJUSTMENT_REVERSAL_NOISE", # offsetting +/- Adjustments, suggests rework
    "EOB_DATA_INTEGRITY",        # EOB amount disagrees with note narrative
    "AGED_OPEN",                 # open >120d with no recent biller activity
    "CASCADE_PATTERN",           # 3+ distinct denial causes in journey
    "NO_HUMAN_TOUCH",            # only system-generated notes; no biller worked it
})

# --- Prompt ----------------------------------------------------------------

SYSTEM_PROMPT = """<role>
You are a physician-billing Revenue Cycle Management (RCM) synthesis engine.
You read the FULL chronological timeline for one patient account - both
ledger events (Charges, Payment, Denial, Adjustment with dollar amounts and
CPT codes) AND free-text biller notes - and produce a structured summary.
</role>

<input_format>
The user message contains a single account's timeline in chronological order.
Each line is one event, prefixed with [YYYY-MM-DD]. Three event sources:

  EVENT  ledger entry. Format:
           [date] EVENT type=Denial amount=1728.00 cpt=99285 cat=Registration/Eligibility
           [date] EVENT type=Charges amount=4271.16
           [date] EVENT type=Payment amount=850.00
           [date] EVENT type=Adjustment amount=4136.22
  BILLER human biller note. Format:
           [date] BILLER user=synbiller1: Claim billed to UMR... underpayment appeal sent
  SYSTEM auto-posted note from a process (CLAIMAUTOUPLOAD, PM_ResponseInterface,
         SHSIndexXMLCreation_*). Treat as low-signal; do not treat as biller work.
</input_format>

<task>
Return FOUR things as a single JSON object:

1. terminal_root_cause: what is blocking this claim RIGHT NOW. Read the most
   recent ledger event AND the most recent biller note. If the latest signal
   is a Payment or a closing Adjustment with the balance going to zero, use
   RESOLVED_PAID or RESOLVED_WRITEOFF. If the latest signal is a denial or an
   open biller question, use the matching cause label.

2. denial_journey: ordered list of DISTINCT causes the claim has cycled
   through, in order of first appearance. Always end with terminal_root_cause.
   If the claim never hit a denial, the journey can be a single label like
   ["AWAITING_RESPONSE"] or ["RESOLVED_PAID"].

3. anomaly_flags: zero or more operational red flags. Examples:
   - INCORRECT_WRITEOFF: an Adjustment whose amount equals (or nearly equals)
     the original Charges was posted before any Payment was received - looks
     like a contractual write-off was applied to a balance that should still
     be collected.
   - WRITEOFF_LIKELY_PR: an Adjustment was posted but the notes mention the
     balance was patient-responsibility (deductible/coinsurance), so the
     write-off was probably wrong.
   - MASS_REGELIG_DENIAL: 2+ separate CPT lines on the same DOS all denied
     with category Registration/Eligibility - smells like a registration
     batch problem, not a one-off.
   - ADJUSTMENT_REVERSAL_NOISE: offsetting positive and negative Adjustments
     for similar amounts - suggests posting errors and rework.
   - EOB_DATA_INTEGRITY: notes claim a payment/denial amount that disagrees
     with the ledger amount on the same DOS.
   - AGED_OPEN: latest ledger or biller event > 120 days before today AND no
     resolved/payment signal.
   - CASCADE_PATTERN: 3 or more distinct cause labels in denial_journey.
   - NO_HUMAN_TOUCH: timeline contains ledger events and possibly SYSTEM
     notes, but zero BILLER notes - account is not being worked.

4. narrative: ONE sentence in plain English summarising what happened on
   this account. Concrete and non-redundant; do not just restate the labels.

Use ONLY the taxonomy and flag tokens listed below. Uppercase, underscores.
</task>

<taxonomy>
REGISTRATION_ELIGIBILITY  eligibility not verified, member ID wrong, plan terminated
COB_ISSUE                 other primary insurance, COB suspended, pt needs to update
MEDICAL_NECESSITY         payor disputes clinical necessity or level of care
BUNDLED_DENIAL            CPT bundled into another, NCCI edit, mutually exclusive
TIMELY_FILING             past payor filing window, CO-29
ADDITIONAL_DOCUMENTATION_NEEDED  records / op notes / itemised bill / W-9 requested
CODING_ERROR              wrong CPT/ICD/modifier, needs corrected rebill
AUTH_REQUIRED             prior auth / pre-cert missing, expired, peer-to-peer
UNDERPAYMENT              paid below contracted rate, underpayment appeal needed
DRG_DOWNGRADE             payor downgraded DRG vs billed
WC_DENIED_BY_ADJUSTOR     workers' comp adjustor denied or compensability disputed
DUPLICATE_CLAIM           CO-18 duplicate of another claim
CLAIM_RECOUPED            payor took back a previously paid amount
BILLED_TO_WRONG_PAYOR     claim sent to wrong primary, needs rebill to correct payor
AWAITING_RESPONSE         claim still pending, no denial yet, biller is waiting
RESOLVED_PAID             claim is paid in full or near-full; balance closing
RESOLVED_WRITEOFF         legitimate contractual or small-balance write-off
UNCLEAR_NOTE              ambiguous or insufficient information
</taxonomy>

<output_format>
Return EXACTLY one JSON object on a single line, nothing before or after:

{"terminal_root_cause":"<TOKEN>","denial_journey":["<TOKEN>",...],"anomaly_flags":["<FLAG>",...],"narrative":"<one sentence>"}

No markdown fences. No prose. Empty anomaly_flags is the empty list [].
</output_format>"""


# --- Timeline -> prompt formatter -----------------------------------------


def _fmt_amount(amt: float | None) -> str:
    if amt is None:
        return ""
    return f"{amt:.2f}"


def _fmt_cpt(cpt) -> str:
    """Render CPT robustly. Pandas may have parsed it as float ('99285.0')."""
    if cpt is None:
        return ""
    s = str(cpt).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def format_timeline_for_prompt(timeline: Iterable[TimelineEvent]) -> str:
    """Render a per-account timeline as the LLM-facing string.

    One event per line. Date prefix in [YYYY-MM-DD]. Events from
    different sources get different leading tokens (EVENT / BILLER /
    SYSTEM) so the LLM can disambiguate ledger from notes.
    """
    lines: list[str] = []
    for ev in timeline:
        date = ev.timestamp.strftime("%Y-%m-%d") if pd.notna(ev.timestamp) else "????-??-??"
        if ev.source == "event":
            parts = [f"type={ev.event_kind}"]
            if ev.amount is not None:
                parts.append(f"amount={_fmt_amount(ev.amount)}")
            if ev.cpt_code:
                parts.append(f"cpt={_fmt_cpt(ev.cpt_code)}")
            if ev.denial_category:
                parts.append(f"cat={ev.denial_category}")
            lines.append(f"[{date}] EVENT " + " ".join(parts))
        elif ev.source == "note" and ev.is_system:
            text = (ev.note_text or "").strip().replace("\n", " ")
            actor = ev.actor or "system"
            if text:
                lines.append(f"[{date}] SYSTEM proc={actor}: {text}")
            else:
                lines.append(f"[{date}] SYSTEM proc={actor}")
        elif ev.source == "note":
            text = (ev.note_text or "").strip().replace("\n", " ")
            actor = ev.actor or "biller"
            lines.append(f"[{date}] BILLER user={actor}: {text}")
    return "\n".join(lines)


# --- Response parsing ------------------------------------------------------


@dataclass
class Synthesis:
    """One LLM synthesis result for a single account."""
    visit: str
    terminal_root_cause: str
    denial_journey: list[str]
    anomaly_flags: list[str]
    narrative: str
    journey_length: int

    def to_row(self) -> dict:
        return {
            "visit": self.visit,
            "terminal_root_cause": self.terminal_root_cause,
            "denial_journey": ",".join(self.denial_journey),
            "anomaly_flags": ",".join(self.anomaly_flags) if self.anomaly_flags else "",
            "journey_length": self.journey_length,
            "narrative": self.narrative,
        }


def _coerce_label(token: str) -> str:
    cleaned = token.strip().upper().replace("-", "_").replace(" ", "_")
    return cleaned if cleaned in VALID_LABELS else FALLBACK_LABEL


def _coerce_flag(token: str) -> str | None:
    cleaned = token.strip().upper().replace("-", "_").replace(" ", "_")
    return cleaned if cleaned in VALID_FLAGS else None


def parse_response(raw: str, visit: str) -> Synthesis:
    """Parse the model's JSON response into a Synthesis.

    On any malformation, returns a Synthesis with FALLBACK_LABEL and a
    diagnostic narrative so downstream code never crashes on a bad row.
    """
    text = raw.strip()
    # Strip accidental markdown fences if the model leaks them.
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    # Sometimes the model adds prose before/after; grab the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return Synthesis(
            visit=visit,
            terminal_root_cause=FALLBACK_LABEL,
            denial_journey=[FALLBACK_LABEL],
            anomaly_flags=[],
            narrative=f"PARSE_FAILED: no JSON object in response: {raw[:120]!r}",
            journey_length=1,
        )
    blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return Synthesis(
            visit=visit,
            terminal_root_cause=FALLBACK_LABEL,
            denial_journey=[FALLBACK_LABEL],
            anomaly_flags=[],
            narrative=f"PARSE_FAILED: {exc.msg} in {blob[:120]!r}",
            journey_length=1,
        )

    terminal = _coerce_label(str(data.get("terminal_root_cause", "")))
    journey_raw = data.get("denial_journey") or []
    if not isinstance(journey_raw, list):
        journey_raw = [journey_raw]
    journey = [_coerce_label(str(t)) for t in journey_raw if str(t).strip()]
    if not journey:
        journey = [terminal]
    if journey[-1] != terminal:
        # Enforce the contract: terminal must be last.
        journey = [c for c in journey if c != terminal] + [terminal]

    flags_raw = data.get("anomaly_flags") or []
    if not isinstance(flags_raw, list):
        flags_raw = [flags_raw]
    flags = [f for f in (_coerce_flag(str(t)) for t in flags_raw) if f]
    # Dedupe while preserving order.
    flags = list(dict.fromkeys(flags))

    narrative = str(data.get("narrative", "")).strip().replace("\n", " ")
    if not narrative:
        narrative = "(no narrative returned)"

    return Synthesis(
        visit=visit,
        terminal_root_cause=terminal,
        denial_journey=journey,
        anomaly_flags=flags,
        narrative=narrative,
        journey_length=len(journey),
    )


# --- LLM call --------------------------------------------------------------


def synthesize_account(client, visit: str, timeline: list[TimelineEvent]) -> Synthesis:
    """Send one account's timeline to Claude Haiku and parse the response."""
    user_msg = format_timeline_for_prompt(timeline)
    if not user_msg.strip():
        return Synthesis(
            visit=visit,
            terminal_root_cause=FALLBACK_LABEL,
            denial_journey=[FALLBACK_LABEL],
            anomaly_flags=[],
            narrative="Empty timeline.",
            journey_length=1,
        )

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except (APIError, APIStatusError) as exc:
        print(f"  ! API error on {visit}: {exc}", file=sys.stderr)
        return Synthesis(
            visit=visit,
            terminal_root_cause=FALLBACK_LABEL,
            denial_journey=[FALLBACK_LABEL],
            anomaly_flags=[],
            narrative=f"API_ERROR: {exc}",
            journey_length=1,
        )

    raw = "".join(b.text for b in resp.content if b.type == "text")
    return parse_response(raw, visit=visit)


def synthesize_per_account(
    timelines: dict[str, list[TimelineEvent]],
    client,
    limit: int | None = None,
    progress_every: int = 10,
) -> list[Synthesis]:
    """Run synthesis for every account in `timelines` (or first `limit`)."""
    visits = list(timelines.keys())
    if limit:
        visits = visits[:limit]

    out: list[Synthesis] = []
    started = time.time()
    for i, visit in enumerate(visits, start=1):
        result = synthesize_account(client, visit, timelines[visit])
        out.append(result)
        if i % progress_every == 0 or i == len(visits):
            elapsed = time.time() - started
            print(
                f"  [{i:>4}/{len(visits)}] last={result.terminal_root_cause:<28} "
                f"flags={len(result.anomaly_flags)} elapsed={elapsed:5.1f}s"
            )
    return out


# --- Self-test (offline) ---------------------------------------------------


def selftest() -> int:
    """Validate prompt formatting and parser without hitting the API.

    Uses the synthetic fixtures committed under tests/fixtures/ so this
    runs in any environment, including CI without ANTHROPIC_API_KEY.
    Returns 0 on success, non-zero on failure.
    """
    fixtures = ROOT / "tests" / "fixtures"
    events_path = fixtures / "events_synthetic.csv"
    notes_path = fixtures / "notes_synthetic.csv"
    if not events_path.exists() or not notes_path.exists():
        print(f"selftest: fixtures missing under {fixtures}", file=sys.stderr)
        return 2

    timelines = build_account_timelines(events_path, notes_path)
    print(f"\nLoaded {len(timelines)} synthetic timelines")

    failures = 0

    # Test 1: every timeline renders to a non-empty string.
    for visit, tl in timelines.items():
        prompt = format_timeline_for_prompt(tl)
        if not prompt.strip():
            print(f"  FAIL: empty prompt for {visit}", file=sys.stderr)
            failures += 1
        else:
            print(f"\n--- {visit} ({len(tl)} events) ---")
            print(prompt)

    # Test 2: parser handles a well-formed response.
    good = '{"terminal_root_cause":"REGISTRATION_ELIGIBILITY","denial_journey":["REGISTRATION_ELIGIBILITY"],"anomaly_flags":["MASS_REGELIG_DENIAL"],"narrative":"All CPT lines denied for eligibility on 2024-05-13."}'
    s = parse_response(good, visit="TEST1")
    assert s.terminal_root_cause == "REGISTRATION_ELIGIBILITY", s
    assert s.denial_journey == ["REGISTRATION_ELIGIBILITY"], s
    assert s.anomaly_flags == ["MASS_REGELIG_DENIAL"], s
    assert s.journey_length == 1, s

    # Test 3: parser tolerates markdown fences.
    fenced = "```json\n" + good + "\n```"
    s2 = parse_response(fenced, visit="TEST2")
    assert s2.terminal_root_cause == "REGISTRATION_ELIGIBILITY", s2

    # Test 4: parser coerces unknown labels to UNCLEAR_NOTE.
    bad_label = '{"terminal_root_cause":"NONSENSE_TOKEN","denial_journey":["NONSENSE_TOKEN","COB_ISSUE"],"anomaly_flags":["WHATEVER"],"narrative":"x"}'
    s3 = parse_response(bad_label, visit="TEST3")
    assert s3.terminal_root_cause == FALLBACK_LABEL, s3
    assert FALLBACK_LABEL in s3.denial_journey, s3
    assert "COB_ISSUE" in s3.denial_journey, s3
    assert s3.anomaly_flags == [], s3  # WHATEVER is not a valid flag

    # Test 5: parser enforces terminal-is-last contract.
    out_of_order = '{"terminal_root_cause":"AUTH_REQUIRED","denial_journey":["AUTH_REQUIRED","COB_ISSUE"],"anomaly_flags":[],"narrative":"x"}'
    s4 = parse_response(out_of_order, visit="TEST4")
    assert s4.denial_journey[-1] == "AUTH_REQUIRED", s4
    assert "COB_ISSUE" in s4.denial_journey, s4

    # Test 6: parser handles a totally broken response without crashing.
    s5 = parse_response("the LLM returned prose only, no JSON", visit="TEST5")
    assert s5.terminal_root_cause == FALLBACK_LABEL, s5
    assert s5.narrative.startswith("PARSE_FAILED"), s5

    if failures:
        print(f"\nselftest FAILED with {failures} prompt-formatting issue(s)", file=sys.stderr)
        return 1
    print("\nselftest OK: prompts render, parser handles good/bad/fenced/wrong-order/garbage.")
    return 0


# --- CLI -------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 - per-account journey synthesis (physician POC)."
    )
    p.add_argument("--events", type=Path, help="Events CSV path.")
    p.add_argument("--notes", type=Path, help="Notes (redacted) CSV path.")
    p.add_argument("--output", type=Path, help="Where to write synthesis CSV.")
    p.add_argument("--limit", type=int, default=None, help="Process first N accounts only.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Format prompts and print the first one; no API call.",
    )
    p.add_argument(
        "--selftest",
        action="store_true",
        help="Run offline self-test against synthetic fixtures.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.selftest:
        return selftest()

    if not args.events or not args.notes:
        print("ERROR: --events and --notes are required (or use --selftest).", file=sys.stderr)
        return 2
    if not args.events.exists():
        print(f"ERROR: events not found: {args.events}", file=sys.stderr)
        return 2
    if not args.notes.exists():
        print(f"ERROR: notes not found: {args.notes}", file=sys.stderr)
        return 2

    timelines = build_account_timelines(args.events, args.notes)
    print(f"\nBuilt {len(timelines)} timelines.")

    if args.limit:
        keys = list(timelines.keys())[: args.limit]
        timelines = {k: timelines[k] for k in keys}

    if args.dry_run:
        for i, (visit, tl) in enumerate(timelines.items()):
            print(f"\n--- {visit} ({len(tl)} events) ---")
            print(format_timeline_for_prompt(tl))
            if i >= 2:
                print(f"\n[...{len(timelines) - 3} more accounts not shown]")
                break
        return 0

    if Anthropic is None:
        print("ERROR: anthropic SDK not installed. pip install anthropic", file=sys.stderr)
        return 2

    load_dotenv(ROOT.parent / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("sk-ant-your-key"):
        print("ERROR: ANTHROPIC_API_KEY missing or placeholder.", file=sys.stderr)
        return 2

    client = Anthropic(api_key=api_key)
    results = synthesize_per_account(timelines, client)

    out_path = args.output or (ROOT / "data" / "synthesized.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.to_row() for r in results]).to_csv(out_path, index=False)
    print(f"\nWrote {len(results)} rows -> {out_path}")

    print("\nTerminal root cause distribution:")
    df = pd.DataFrame([asdict(r) for r in results])
    for label, n in df["terminal_root_cause"].value_counts().items():
        print(f"  {label:<32} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
