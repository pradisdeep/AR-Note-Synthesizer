"""
PHI redactor for physician_poc note data.

Targets known patterns in the production notes:

  1. Patient names embedded in CLAIMAUTOUPLOAD-generated rows in the form
     "Insurance Name LASTNAME,FIRSTNAME ..." where LASTNAME and FIRSTNAME are
     in all caps. Replaced with "Insurance Name [REDACTED-NAME]".

  2. Member IDs / health plan beneficiary numbers - these are PHI under HIPAA
     Safe Harbor (45 CFR 164.514). Replaced with "[REDACTED-MEMBER-ID]".

  3. SSN-style 9-digit numbers (defensive - not seen in samples but worth
     guarding against). Replaced with "[REDACTED-SSN]".

The redactor is intentionally conservative: it errs on the side of redacting
something that turns out to be benign rather than letting PHI through. Run
report() after redact_dataframe() to get a count of how many substitutions
were made of each type so you can spot-check the result.

This module touches only the free-text Notes column. It does not modify
account numbers, dates, payor names, denial codes, or other structured
fields - those are not PHI under Safe Harbor for de-identified records as
long as patient identifiers are removed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

# --- Patterns ---------------------------------------------------------------

# 1. "Insurance Name LASTNAME,FIRSTNAME ..." in CLAIMAUTOUPLOAD rows.
#    Shape (synthetic illustrative examples):
#      "Insurance Name DOE,JOHN denied the claim"
#      "Insurance Name ROE,JANE denied"
#      "Insurance Name TESTPATIENT,FAKE for DOS"
#    Captures: 2-30 char ALL-CAPS surnames (allowing apostrophes, hyphens,
#    spaces for compound names) followed by comma + 2-30 char given name.
# NOTE: case-sensitive on the name portion. With re.IGNORECASE the [A-Z]
# class would match lowercase too, and the greedy second-name group would
# swallow words like "denied the claim" after the name. Patient names in
# CLAIMAUTOUPLOAD rows are reliably ALL-CAPS, so case-sensitive matching is
# both correct and safer. The "Insurance Name" prefix is matched with an
# inline (?i:...) so we tolerate prefix-casing variants without breaking
# the strict-uppercase name match.
INSURANCE_NAME_PATTERN = re.compile(
    r"(?i:Insurance Name)\s+[A-Z][A-Z'\-\s]{1,29},\s*[A-Z][A-Z'\-\s]{1,29}(?=\s+(?:denied|for|stating|paid|approved|processed|rejected))"
)

# 2. Member IDs - alphanumeric, 6-15 chars, when preceded by member-id-style label.
#    Shape (synthetic illustrative examples):
#      "member id#XYZ000111222"
#      "member id # A0000000001"
#      "Member ID ZZZ999888777"
MEMBER_ID_PATTERN = re.compile(
    r"\b(member\s*id|memberid|member\s*number|policy\s*id|policy\s*number|subscriber\s*id)\s*#?\s*[A-Z0-9]{6,15}\b",
    re.IGNORECASE,
)

# 3. SSN pattern (defensive). Not observed in the sample but cheap insurance.
#    Skips obvious phone numbers: requires SSN format with dashes.
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# 4. Date-of-birth-style mentions ("DOB 01/01/1980", "born 1/1/1980").
#    Service dates are not PHI on their own under Safe Harbor at year-only
#    granularity, but a DOB is. Conservatively scrub.
DOB_PATTERN = re.compile(
    r"\b(?:DOB|D\.O\.B\.?|born\s+on|date\s+of\s+birth)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
    re.IGNORECASE,
)

# 5. (Optional, off by default) Workers-comp employer names. Not PHI under
#    Safe Harbor on their own, but for very small employers in a narrow
#    geography, employer + DOS can narrow identity. Provide a configurable
#    list rather than an open-ended NER pass to keep precision high and
#    avoid scrubbing payor or facility names.
#
#    DEFAULT_EMPLOYER_TERMS is intentionally EMPTY in source. Operational
#    employer lists are themselves sensitive (committing them to git would
#    record which employers send WC patients to this practice). Populate
#    at runtime instead - either by editing this list locally before
#    running, or by passing employer_terms=[...] to redact_dataframe()
#    from a config file kept outside the repo.
#
#    Example shape of values you would pass:
#      employer_terms=["EXAMPLE FOODS INC", "EXAMPLE TRUCKING", "EXAMPLE PD"]
DEFAULT_EMPLOYER_TERMS: list[str] = []


def _build_employer_pattern(terms: list[str]) -> re.Pattern | None:
    """Compile a case-insensitive alternation matching whole-word employer terms."""
    if not terms:
        return None
    # Sort longest first so multi-word matches don't get pre-empted by a single token.
    safe = sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)
    if not safe:
        return None
    alternation = "|".join(re.escape(t) for t in safe)
    return re.compile(rf"(?<![A-Z0-9]){alternation}(?![A-Z0-9])", re.IGNORECASE)


@dataclass
class RedactionStats:
    insurance_names: int = 0
    member_ids: int = 0
    ssns: int = 0
    dobs: int = 0
    employers: int = 0
    rows_touched: int = 0
    samples: list[tuple[str, str]] = field(default_factory=list)  # (before, after) pairs

    def total(self) -> int:
        return (
            self.insurance_names
            + self.member_ids
            + self.ssns
            + self.dobs
            + self.employers
        )

    def report(self) -> str:
        lines = [
            "Redaction summary",
            "=================",
            f"  Insurance Name (patient name) hits : {self.insurance_names}",
            f"  Member ID hits                     : {self.member_ids}",
            f"  SSN hits                           : {self.ssns}",
            f"  DOB hits                           : {self.dobs}",
            f"  Employer name hits (optional)      : {self.employers}",
            f"  Total substitutions                : {self.total()}",
            f"  Rows with at least one substitution: {self.rows_touched}",
        ]
        if self.samples:
            lines.append("")
            lines.append("Sample diffs (first few touched rows):")
            for i, (before, after) in enumerate(self.samples[:5], start=1):
                lines.append(f"  [{i}] BEFORE: {before[:160]}{'...' if len(before) > 160 else ''}")
                lines.append(f"      AFTER : {after[:160]}{'...' if len(after) > 160 else ''}")
        return "\n".join(lines)


def redact_text(
    text: str,
    stats: RedactionStats,
    employer_pattern: re.Pattern | None = None,
) -> str:
    """Apply all redaction patterns to a single string. Updates stats in place.

    employer_pattern is optional. Pass None (default) to skip employer
    scrubbing; pass a compiled pattern from _build_employer_pattern() to
    enable it.
    """
    if not isinstance(text, str) or not text:
        return text

    out = text

    new, n = INSURANCE_NAME_PATTERN.subn("Insurance Name [REDACTED-NAME]", out)
    stats.insurance_names += n
    out = new

    new, n = MEMBER_ID_PATTERN.subn(lambda m: f"{m.group(1)} [REDACTED-MEMBER-ID]", out)
    stats.member_ids += n
    out = new

    new, n = SSN_PATTERN.subn("[REDACTED-SSN]", out)
    stats.ssns += n
    out = new

    new, n = DOB_PATTERN.subn("[REDACTED-DOB]", out)
    stats.dobs += n
    out = new

    if employer_pattern is not None:
        new, n = employer_pattern.subn("[REDACTED-EMPLOYER]", out)
        stats.employers += n
        out = new

    return out


def redact_dataframe(
    df: pd.DataFrame,
    note_col: str = "notescurrentvalue",
    redact_employers: bool = False,
    employer_terms: list[str] | None = None,
) -> tuple[pd.DataFrame, RedactionStats]:
    """Return a copy of df with the note column redacted, plus stats.

    redact_employers: opt-in flag for employer-name scrubbing. Default False
    because employer names are not PHI under Safe Harbor on their own;
    enable when policy or context requires it (e.g. small workers-comp
    employers where employer + DOS could narrow identity).

    employer_terms: optional override for the employer term list. Defaults
    to DEFAULT_EMPLOYER_TERMS when redact_employers=True and no list
    is supplied. Pass an empty list to effectively disable even if the
    flag is True.
    """
    if note_col not in df.columns:
        raise KeyError(f"Note column '{note_col}' not found in dataframe (cols: {list(df.columns)})")

    employer_pattern: re.Pattern | None = None
    if redact_employers:
        terms = employer_terms if employer_terms is not None else DEFAULT_EMPLOYER_TERMS
        employer_pattern = _build_employer_pattern(terms)

    stats = RedactionStats()
    out = df.copy()
    redacted_values = []

    for original in out[note_col].tolist():
        new_value = redact_text(original, stats, employer_pattern=employer_pattern)
        if isinstance(original, str) and new_value != original:
            stats.rows_touched += 1
            stats.samples.append((original, new_value))
        redacted_values.append(new_value)

    out[note_col] = redacted_values
    return out, stats


# --- Test harness for the patterns I observed in real data ------------------


# Synthetic test fixtures only. These are NOT real names, IDs, or employers
# - they exercise the same regex patterns observed in production note data
# without embedding any real PHI or operational specifics in the source
# code. Format intentionally matches the real strings: ALL-CAPS Last,First
# inside CLAIMAUTOUPLOAD-style sentences and alphanumeric member-id
# substrings of typical length.
SYNTHETIC_TEST_FIXTURES = [
    # 1. CLAIMAUTOUPLOAD rows with synthetic names matching the real shape
    "As Per UHC online status, Insurance Name DOE,JOHN denied the claim stating non covered As Per UHC patient plan.  Claim Processed Date 13-01-2025.",
    "As Per UHC online status, Insurance Name ROE,JANE denied the claim stating non covered As Per UHC patient plan.  Claim Processed Date 23-03-2025.",
    "As Per CIG online status, patient is not eligible with the Insurance Name TESTPATIENT,FAKE for DOS 15-09-2024.  Claim Processed Date 24-12-2024.",
    # 2. Member IDs - synthetic alphanumeric strings of typical length
    "Verified member ID found claims pays up to 150% of Medicare allowable charges and as member didn't meet his Deductible, member id#XYZ000111222 active on dos.",
    "AMBETTER with member id # A0000000001 is primary for the patient.",
    "Member does not have active coverage under member id #ZZZ999888777.",
    # 3. Defensive SSN test (test-fake number)
    "Patient SSN 000-00-0000 verified during call.",
    # 4. Defensive DOB test
    "Patient DOB 01/01/1900 confirmed with insurance rep.",
    # 5. Notes that should NOT be touched
    "called UHC re DOS 3/14 - rep says no auth on file, denl CO-197. need retro auth, sent rqst to prov ofc",
    "claim billed to BCBS for the DOS 02/02/2024-02/14/2024, paid $7,367.00, ptr $2,221.00",
    "PO BOX 14465 LEXINGTON KY 40512-4465",   # insurance address
    "called employer @580-436-1500",          # employer phone, not PHI
    "spoke with rep ben",                     # insurance rep first name, not PHI
    "Spandana.shs",                           # biller signature, not PHI
]

# Employer-toggle test fixtures. SYNTHETIC names only - "ACME WIDGETS",
# "BETA TRUCKING", "GAMMA POLICE DEPT" - chosen to look obviously fake.
# The selftest passes these via employer_terms= to verify the toggle works
# without committing the actual operational employer list.
SYNTHETIC_EMPLOYER_TERMS = ["ACME WIDGETS", "BETA TRUCKING", "GAMMA POLICE DEPT"]
SYNTHETIC_EMPLOYER_FIXTURES = [
    "claim billed to ACME WIDGETS, no eob received yet",
    "called employer BETA TRUCKING for WC injury report status",
    "?As per agent comments, claim billed to GAMMA POLICE DEPT and out of scope",
    # Negative case - similar text, no fixture employer present
    "called employer at 800-699-4115, left voicemail",
]


def selftest() -> None:
    """Run the redactor against synthetic fixtures shaped like real-data patterns."""
    print("Phase 1 - default mode (employer redaction OFF)")
    print("=" * 60)
    df = pd.DataFrame({"notescurrentvalue": SYNTHETIC_TEST_FIXTURES})
    redacted, stats = redact_dataframe(df)
    print(stats.report())
    assert stats.employers == 0, "Employer redaction should be off by default"

    print()
    print("Phase 2 - employer redaction ON, synthetic terms")
    print("=" * 60)
    df2 = pd.DataFrame({"notescurrentvalue": SYNTHETIC_EMPLOYER_FIXTURES})
    redacted2, stats2 = redact_dataframe(
        df2, redact_employers=True, employer_terms=SYNTHETIC_EMPLOYER_TERMS
    )
    print(stats2.report())
    print()
    for before, after in zip(df2["notescurrentvalue"], redacted2["notescurrentvalue"]):
        marker = "REDACTED" if before != after else "unchanged"
        print(f"  [{marker}]")
        print(f"    {before}")
        if before != after:
            print(f"  -> {after}")
    assert stats2.employers == 3, f"Expected 3 employer hits, got {stats2.employers}"
    assert stats2.rows_touched == 3, f"Expected 3 rows touched, got {stats2.rows_touched}"

    print()
    print("OK - both modes pass.")


if __name__ == "__main__":
    selftest()
