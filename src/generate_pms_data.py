"""
Phase 1 - Synthetic PMS extract generator (multi-note edition).

Produces ~150 unique accounts, each with 1 to 4 chronologically-ordered notes,
following realistic denial-cascade patterns:

  * 40% single-note (one root cause, claim still active or recently resolved)
  * 35% two-stage cascade (e.g. COB resolved -> Auth surfaces)
  * 20% three-stage cascade (deeper rework cycles)
  *  5% four-stage cascade (severe Burnout Zone candidates)

The cascades reflect patterns common in healthcare RCM:
  - COB updates expire the auth window -> Missing Auth follows
  - Clinical-records delays push the claim past the filing limit -> Timely Filing
  - Auth peer-to-peer needs records -> Clinical Required follows
  - Provider not par on the new primary payor -> Credentialing follows COB
  - Appeals on TFL need supporting clinical evidence -> Clinical follows TFL

Notes within an account share Payor and Primary DX (the same claim) but have
their own User Touch Date, biller (User Name), Status / Sub-status, and Notes.

Output: data/synthetic_pms_extract.csv (~300 rows for ~150 accounts)
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "synthetic_pms_extract.csv"
N_ACCOUNTS = 150
SEED = 42

PAYORS = ["UHC", "Aetna", "BCBS", "Medicare Primary", "Medicaid"]
DX_CODES = ["E11.9", "J45.909", "I10", "M54.5"]
STATUSES = ["Open", "Pending", "Denied", "Appeal", "Follow-up"]
SUB_STATUSES = [
    "Awaiting Payor",
    "Need Info",
    "Reviewed",
    "Sent to Provider",
    "Escalated",
    "Worked",
]

ROOT_CAUSES = [
    "Missing Auth",
    "COB Issue",
    "Clinical Records Required",
    "Credentialing Error",
    "Timely Filing",
]

# Realistic denial cascades. Each list is one chronological journey for an
# account. The first item is the original blocker; subsequent items are what
# surfaced after the previous blocker was (partially) resolved.
LENGTH_2_CASCADES = [
    ["COB Issue", "Missing Auth"],                 # COB delay expired auth
    ["Clinical Records Required", "Timely Filing"], # docs took too long
    ["Missing Auth", "Clinical Records Required"],  # peer-to-peer needs docs
    ["Credentialing Error", "Timely Filing"],       # cred fix delayed claim
    ["COB Issue", "Credentialing Error"],           # new primary, prov not par
    ["Missing Auth", "COB Issue"],                  # auth call surfaced new ins
    ["Timely Filing", "Clinical Records Required"], # appeal needs records
]

LENGTH_3_CASCADES = [
    ["COB Issue", "Missing Auth", "Clinical Records Required"],
    ["Missing Auth", "Credentialing Error", "Timely Filing"],
    ["Credentialing Error", "COB Issue", "Missing Auth"],
    ["Clinical Records Required", "Timely Filing", "Clinical Records Required"],
    ["COB Issue", "Missing Auth", "Timely Filing"],
]

LENGTH_4_CASCADES = [
    ["COB Issue", "Missing Auth", "Clinical Records Required", "Timely Filing"],
    ["Credentialing Error", "COB Issue", "Missing Auth", "Timely Filing"],
    ["Missing Auth", "Clinical Records Required", "Credentialing Error", "Timely Filing"],
]

JOURNEY_LENGTH_WEIGHTS = [40, 35, 20, 5]  # for lengths 1, 2, 3, 4

# Templates for stage-aware notes. Each cascade stage gets its own flavour so
# the LLM has temporal context to work with - "retro auth submitted" reads
# differently from "called UHC re DOS, no auth on file".
STAGE_TEMPLATES = {
    "Missing Auth": {
        "first": [
            "called {payor} re DOS {dos} - rep says no auth on file for {dx}, denl CO-197. need retro auth, sent rqst to prov ofc",
            "{payor} EOB shows denl - auth required prior to svc {dx}. retro auth rqst submitted, awaiting determ",
            "spoke to {payor}, no precert obtained for DOS {dos} {dx}. attempting peer to peer review",
        ],
        "follow": [
            "{payor} - prior COB now resolved but auth window expired during delay. re-attempting retro auth {dx}",
            "after pt updated other ins, {payor} now wants auth for {dx} - none was obtained, requesting retro",
            "cred file fixed, claim reprocessed - now denl for missing auth on {dx}. submitting retro auth rqst",
        ],
    },
    "COB Issue": {
        "first": [
            "{payor} bounced clm - says pt has other primary ins. need updated COB from pt, called pt LVM",
            "{payor} rep states COB outdated, last update 2yrs ago. pt has new MCR primary now. {dx} svc on hold",
            "denl - {payor} needs COB info refreshed, suspended pending other ins verif. f/u 5d {dx}",
        ],
        "follow": [
            "while working auth, pt mentioned different primary ins - need to update COB w/ {payor} for {dx}",
            "during cred review, found pt switched ins last yr. {payor} now secondary. need fresh COB {dx}",
            "{payor} rep notes other ins on file - need pt to update before re-adjudication of {dx} clm",
        ],
    },
    "Clinical Records Required": {
        "first": [
            "{payor} req medical records for DOS {dos} {dx} - faxed rqst to clinic, awaiting recd",
            "denl from {payor} - clinical docs needed to support {dx} dx. ordered chart notes from prov",
            "{payor} addl info req - operative report and progress notes for {dx}. submitted, no resp",
        ],
        "follow": [
            "peer to peer w/ {payor} req clinical justification for {dx} - chasing prov for chart notes",
            "TFL appeal needs supporting documentation - faxed records request to provider for {dx}",
            "after auth retro denied, {payor} now requesting full clinical pkg for {dx} appeal",
        ],
    },
    "Credentialing Error": {
        "first": [
            "{payor} denl - prov NPI not effective on DOS. cred dept says enrollment still pending {dx}",
            "{payor} rep says rendering prov not par on DOS, need to check w/ cred team. {dx} svc held",
            "{payor} clm rejection - taxonomy/NPI mismatch, cred file out of date. fix in PM and rebill {dx}",
        ],
        "follow": [
            "after COB updated to new primary {payor}, found prov not credentialed on this panel. cred coord notified {dx}",
            "auth obtained but cred check shows prov terming on {payor} panel - escalated to cred dept {dx}",
            "{payor} reprocessing held - cred file out of sync w/ NPI registry. fixing taxonomy {dx}",
        ],
    },
    "Timely Filing": {
        "first": [
            "{payor} denl CO-29 timely filing exceeded for DOS {dos} {dx}. checking proof of timely submission",
            "TFL denl from {payor}, orig clm sent in window but no record on payor side. resubmit w/ POTF",
            "{payor} rejected as TFL exceeded. orig submission was electronic, pulling 277CA for proof",
        ],
        "follow": [
            "after lengthy clinical review, {payor} now denying as TFL exceeded - drafting appeal w/ POTF for {dx}",
            "cred fix took 90d - {payor} now denying claim as TFL. building appeal w/ acceptance report {dx}",
            "appeal in progress - {payor} TFL denial {dx}, have clearinghouse acceptance from DOS+12d",
        ],
    },
}


def _random_dos(fake: Faker) -> str:
    """A date-of-service string in messy short form (e.g. 3/14)."""
    d = fake.date_between(start_date="-180d", end_date="-15d")
    return f"{d.month}/{d.day}"


def _build_note(fake: Faker, payor: str, dx: str, root_cause: str, is_first_stage: bool) -> str:
    bucket = "first" if is_first_stage else "follow"
    template = random.choice(STAGE_TEMPLATES[root_cause][bucket])
    return template.format(payor=payor, dx=dx, dos=_random_dos(fake))


def _next_followup(touch_date: date) -> date:
    return touch_date + timedelta(days=random.randint(3, 21))


def _pick_cascade() -> list[str]:
    length = random.choices([1, 2, 3, 4], weights=JOURNEY_LENGTH_WEIGHTS)[0]
    if length == 1:
        return [random.choice(ROOT_CAUSES)]
    if length == 2:
        return list(random.choice(LENGTH_2_CASCADES))
    if length == 3:
        return list(random.choice(LENGTH_3_CASCADES))
    return list(random.choice(LENGTH_4_CASCADES))


def _generate_account(account_id: str, fake: Faker) -> list[dict]:
    """Generate 1-4 chronologically-ordered notes for one account."""
    payor = random.choice(PAYORS)
    dx = random.choice(DX_CODES)
    cascade = _pick_cascade()

    rows: list[dict] = []
    base_date = fake.date_between(start_date="-180d", end_date="-30d")
    current_date = base_date

    for i, root_cause in enumerate(cascade):
        if i > 0:
            current_date = current_date + timedelta(days=random.randint(7, 35))

        rows.append(
            {
                "Account Number": account_id,
                "User Touch Date": current_date.isoformat(),
                "Payor Name": payor,
                "Primary DX": dx,
                "Status": random.choice(STATUSES),
                "Sub-status": random.choice(SUB_STATUSES),
                "User Name": fake.user_name(),
                "Next Followup Date": _next_followup(current_date).isoformat(),
                "Notes": _build_note(fake, payor, dx, root_cause, is_first_stage=(i == 0)),
            }
        )

    return rows


def generate_rows(n_accounts: int = N_ACCOUNTS) -> list[dict]:
    fake = Faker()
    Faker.seed(SEED)
    random.seed(SEED)

    rows: list[dict] = []
    for _ in range(n_accounts):
        account_id = f"ACC{fake.unique.random_number(digits=7, fix_len=True)}"
        rows.extend(_generate_account(account_id, fake))
    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows()

    fieldnames = [
        "Account Number",
        "User Touch Date",
        "Payor Name",
        "Primary DX",
        "Status",
        "Sub-status",
        "User Name",
        "Next Followup Date",
        "Notes",
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_accounts = len({r["Account Number"] for r in rows})
    print(f"Wrote {len(rows)} rows across {n_accounts} accounts -> {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
