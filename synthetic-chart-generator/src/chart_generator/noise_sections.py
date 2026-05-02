"""Provider-variant noise sections (orders, prescriptions, referrals).

Real-world charts vary by EHR / provider. This module fabricates that
variation so the noise filter has realistic ground truth to test against.

Each `NoiseSection` describes one chunk of non-coding-relevant content the
chart will include: a header label that mimics a particular EHR's
convention, a layout style (table-with-columns vs bullet-list), and a
list of generated rows.

The medcoding pipeline must drop these sections before they reach phi-4.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Each "provider profile" mimics a real EHR's section style. None of these
# are exact reproductions — they're plausible variants that surface the
# same coding hazard (orders/prescriptions that aren't billable services).

PROVIDER_PROFILES = [
    {
        "name": "epic_like",
        "header": "Orders Placed This Visit",
        "columns": ["Order", "Status", "Authorized By", "Date"],
        "statuses": ["Sent", "Pending", "Acknowledged", "In Process"],
        "kind": "orders_table",
    },
    {
        "name": "cerner_like",
        "header": "Provider Orders",
        "columns": ["Type", "Description", "Frequency", "Status"],
        "statuses": ["Active", "Discontinued", "On Hold"],
        "kind": "orders_table",
    },
    {
        "name": "athena_like",
        "header": "Outgoing Orders",
        "columns": [],
        "statuses": [],
        "kind": "orders_bulleted",
    },
    {
        "name": "allscripts_like",
        "header": "Rx / Prescriptions",
        "columns": ["Drug", "Sig", "Refills", "Pharmacy"],
        "statuses": [],
        "kind": "prescriptions_table",
    },
    {
        "name": "generic_referrals",
        "header": "Referrals",
        "columns": ["To", "Reason", "Urgency", "Status"],
        "statuses": ["Routine", "Urgent", "STAT"],
        "kind": "referrals_table",
    },
]

ORDERABLE_LABS = [
    "Comprehensive Metabolic Panel",
    "CBC with differential",
    "Hemoglobin A1C",
    "Lipid Panel",
    "TSH",
    "Urinalysis",
    "PT/INR",
    "Vitamin D 25-OH",
    "Iron studies",
    "Magnesium",
]

ORDERABLE_IMAGING = [
    "Chest X-ray 2 views",
    "MRI lumbar spine without contrast",
    "CT abdomen and pelvis with contrast",
    "Echocardiogram transthoracic",
    "Ultrasound right upper quadrant",
    "Mammogram screening bilateral",
    "DEXA scan",
]

PRESCRIPTION_DRUGS = [
    ("Amoxicillin 500mg", "1 cap PO TID x 10 days", 0),
    ("Azithromycin 250mg", "2 tabs PO day 1 then 1 tab PO daily x 4 days", 0),
    ("Prednisone 20mg", "1 tab PO daily x 5 days then taper", 0),
    ("Lisinopril 10mg", "1 tab PO daily", 5),
    ("Metformin 500mg", "1 tab PO BID", 5),
    ("Atorvastatin 40mg", "1 tab PO QHS", 11),
    ("Albuterol HFA", "2 puffs inhaled Q4-6h PRN wheezing", 2),
    ("Hydrochlorothiazide 25mg", "1 tab PO daily", 5),
]

REFERRAL_SPECIALTIES = [
    ("Cardiology", "Evaluation of palpitations and abnormal EKG"),
    ("Endocrinology", "Diabetes management"),
    ("Gastroenterology", "Abdominal pain workup"),
    ("Orthopedics", "Knee pain failing conservative management"),
    ("Pulmonology", "Persistent cough and abnormal CXR"),
    ("Neurology", "Recurrent headache evaluation"),
    ("Dermatology", "Suspicious skin lesion biopsy"),
    ("Physical Therapy", "Strengthening and ROM after injury"),
]


@dataclass
class NoiseRow:
    """One row of an order/prescription/referral table or bulleted list."""

    cells: list[str] = field(default_factory=list)
    bullet_text: str = ""


@dataclass
class NoiseSection:
    """One non-coding-relevant block of content to inject into a chart."""

    profile_name: str  # which provider profile this came from
    header: str
    kind: str  # orders_table | orders_bulleted | prescriptions_table | referrals_table
    columns: list[str]
    rows: list[NoiseRow]


def _faker_pharmacy(rng: random.Random) -> str:
    chains = [
        "CVS Pharmacy",
        "Walgreens",
        "Rite Aid",
        "Walmart Pharmacy",
        "Costco Pharmacy",
        "Local Compounding Pharmacy",
        "Mail Order Express Scripts",
    ]
    return rng.choice(chains)


def _generate_orders_table_rows(
    profile: dict, rng: random.Random, n: int
) -> list[NoiseRow]:
    rows: list[NoiseRow] = []
    pool = ORDERABLE_LABS + ORDERABLE_IMAGING
    if profile["name"] == "cerner_like":
        # Cerner-style includes a Type column distinguishing Lab/Imaging/Med
        types = ["Lab", "Lab", "Imaging", "Medication"]
        for _ in range(n):
            t = rng.choice(types)
            if t == "Lab":
                desc = rng.choice(ORDERABLE_LABS)
                freq = rng.choice(["Once", "Daily", "Weekly", "Q3 months"])
            elif t == "Imaging":
                desc = rng.choice(ORDERABLE_IMAGING)
                freq = "Once"
            else:
                drug, sig, _ = rng.choice(PRESCRIPTION_DRUGS)
                desc = drug
                freq = sig
            rows.append(
                NoiseRow(cells=[t, desc, freq, rng.choice(profile["statuses"])])
            )
    else:  # epic_like default
        for _ in range(n):
            order = rng.choice(pool)
            status = rng.choice(profile["statuses"])
            authorizer = f"Dr. {chr(rng.randint(65, 90))}{chr(rng.randint(97, 122))}{chr(rng.randint(97, 122))}{chr(rng.randint(97, 122))}"
            rows.append(
                NoiseRow(cells=[order, status, authorizer, _random_recent_date(rng)])
            )
    return rows


def _generate_orders_bulleted(rng: random.Random, n: int) -> list[NoiseRow]:
    rows: list[NoiseRow] = []
    pool = ORDERABLE_LABS + ORDERABLE_IMAGING
    targets = [
        "Send to: Quest Diagnostics",
        "Send to: LabCorp",
        "Send to: Hospital Imaging Center",
        "Send to: Outpatient Lab",
    ]
    for _ in range(n):
        order = rng.choice(pool)
        target = rng.choice(targets)
        rows.append(NoiseRow(bullet_text=f"{order} -- {target}"))
    return rows


def _generate_prescriptions_rows(rng: random.Random, n: int) -> list[NoiseRow]:
    rows: list[NoiseRow] = []
    for _ in range(n):
        drug, sig, refills = rng.choice(PRESCRIPTION_DRUGS)
        rows.append(NoiseRow(cells=[drug, sig, str(refills), _faker_pharmacy(rng)]))
    return rows


def _generate_referrals_rows(profile: dict, rng: random.Random, n: int) -> list[NoiseRow]:
    rows: list[NoiseRow] = []
    for _ in range(n):
        spec, reason = rng.choice(REFERRAL_SPECIALTIES)
        urgency = rng.choice(profile["statuses"])
        status = rng.choice(["Pending Auth", "Sent", "Scheduled", "Awaiting Response"])
        rows.append(NoiseRow(cells=[spec, reason, urgency, status]))
    return rows


def _random_recent_date(rng: random.Random) -> str:
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{month:02d}/{day:02d}/2025"


def generate_noise_sections(
    rng: random.Random, *, count: int = 1
) -> list[NoiseSection]:
    """Pick `count` distinct provider profiles and generate noise content."""

    if count <= 0:
        return []
    chosen_profiles = rng.sample(
        PROVIDER_PROFILES, k=min(count, len(PROVIDER_PROFILES))
    )
    sections: list[NoiseSection] = []
    for profile in chosen_profiles:
        n_rows = rng.randint(2, 5)
        if profile["kind"] == "orders_table":
            rows = _generate_orders_table_rows(profile, rng, n_rows)
        elif profile["kind"] == "orders_bulleted":
            rows = _generate_orders_bulleted(rng, n_rows)
        elif profile["kind"] == "prescriptions_table":
            rows = _generate_prescriptions_rows(rng, n_rows)
        elif profile["kind"] == "referrals_table":
            rows = _generate_referrals_rows(profile, rng, n_rows)
        else:
            rows = []
        sections.append(
            NoiseSection(
                profile_name=profile["name"],
                header=profile["header"],
                kind=profile["kind"],
                columns=list(profile["columns"]),
                rows=rows,
            )
        )
    return sections
