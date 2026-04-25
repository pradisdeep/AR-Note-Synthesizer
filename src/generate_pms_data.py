"""
Phase 1 - Synthetic PMS extract generator.

Produces 150 rows of PHI-free Practice Management System data with messy,
biller-style free-text notes that implicitly encode one of five root causes:
Missing Auth, COB Issue, Clinical Records Required, Credentialing Error,
Timely Filing.

Output: data/synthetic_pms_extract.csv
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "synthetic_pms_extract.csv"
N_ROWS = 150
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

# Note templates per root cause. Each template intentionally uses biller
# shorthand (DOS, EOB, pt, f/u, RX, prov, denl, recd, auth#, TFL, etc.)
# and embeds {payor} and {dx} so the LLM has to read the note holistically.
NOTE_TEMPLATES = {
    "Missing Auth": [
        "called {payor} re DOS {dos} - rep says no auth on file for {dx}, denl CO-197. need retro auth, sent rqst to prov ofc",
        "pt called, claim denied no auth. {payor} wants auth# for {dx} svcs. f/u w/ provider 7d",
        "{payor} EOB shows denl - auth required prior to svc {dx}. retro auth rqst submitted, awaiting determ",
        "spoke to {payor}, no precert obtained for DOS {dos} {dx}. attempting peer to peer review",
        "denl recd from {payor} - svc requires prior auth, none on file. {dx} pt - working w/ ofc on retro",
    ],
    "COB Issue": [
        "{payor} bounced clm - says pt has other primary ins. need updated COB from pt, called pt LVM",
        "called pt re COB - {payor} secondary, pt confirms term'd primary. need {payor} to update COB on file {dx}",
        "{payor} rep states COB outdated, last update 2yrs ago. pt has new MCR primary now. {dx} svc on hold",
        "denl - {payor} needs COB info refreshed, suspended pending other ins verif. f/u 5d {dx}",
        "called {payor} - clm pending COB update, pt needs to call payor directly to update. left vm pt {dx}",
    ],
    "Clinical Records Required": [
        "{payor} req medical records for DOS {dos} {dx} - faxed rqst to clinic, awaiting recd",
        "denl from {payor} - clinical docs needed to support {dx} dx. ordered chart notes from prov",
        "{payor} addl info req - operative report and progress notes for {dx}. submitted 10d ago, no resp",
        "review pending {payor} - clinical recs received but missing op note. resubmitting w/ complete pkg {dx}",
        "{payor} downcoded {dx} clm - appeal w/ medical records in progress, prov dictation pending",
    ],
    "Credentialing Error": [
        "{payor} denl - prov NPI not effective on DOS. cred dept says enrollment still pending {dx}",
        "called {payor}, prov not loaded in system. cred app submitted but not approved yet, pt seen {dx}",
        "{payor} rep says rendering prov not par on DOS, need to check w/ cred team. {dx} svc held",
        "denl recd - prov terming/cred issue on {payor} panel. forwarded to cred coord {dx}",
        "{payor} clm rejection - taxonomy/NPI mismatch, cred file out of date. fix in PM and rebill {dx}",
    ],
    "Timely Filing": [
        "{payor} denl CO-29 timely filing exceeded for DOS {dos} {dx}. checking proof of timely submission",
        "TFL denl from {payor}, orig clm sent in window but no record on payor side. resubmit w/ POTF",
        "denl - past timely filing limit {payor} {dx}. appeal drafted w/ EDI acceptance report attached",
        "{payor} rejected as TFL exceeded. orig submission was electronic, pulling 277CA for proof",
        "appeal in progress - {payor} TFL denial {dx}, have clearinghouse acceptance from DOS+12d",
    ],
}


def _random_dos(fake: Faker) -> str:
    """A date-of-service string in messy short form (e.g. 3/14)."""
    d = fake.date_between(start_date="-180d", end_date="-15d")
    return f"{d.month}/{d.day}"


def _build_note(fake: Faker, payor: str, dx: str, root_cause: str) -> str:
    template = random.choice(NOTE_TEMPLATES[root_cause])
    return template.format(payor=payor, dx=dx, dos=_random_dos(fake))


def _next_followup(touch_date: date) -> date:
    return touch_date + timedelta(days=random.randint(3, 21))


def generate_rows(n: int = N_ROWS) -> list[dict]:
    fake = Faker()
    Faker.seed(SEED)
    random.seed(SEED)

    rows: list[dict] = []
    for _ in range(n):
        payor = random.choice(PAYORS)
        dx = random.choice(DX_CODES)
        root_cause = random.choice(ROOT_CAUSES)
        touch = fake.date_between(start_date="-60d", end_date="today")

        rows.append(
            {
                "Account Number": f"ACC{fake.unique.random_number(digits=7, fix_len=True)}",
                "User Touch Date": touch.isoformat(),
                "Payor Name": payor,
                "Primary DX": dx,
                "Status": random.choice(STATUSES),
                "Sub-status": random.choice(SUB_STATUSES),
                "User Name": fake.user_name(),
                "Next Followup Date": _next_followup(touch).isoformat(),
                "Notes": _build_note(fake, payor, dx, root_cause),
            }
        )
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

    print(f"Wrote {len(rows)} rows -> {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
