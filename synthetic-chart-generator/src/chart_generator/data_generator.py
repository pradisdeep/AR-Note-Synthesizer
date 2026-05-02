"""Generate synthetic chart payloads using Faker plus clinical dictionaries.

All identifiers are fabricated. The MRN, account numbers, and provider details
are random; nothing here references real patient data.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from faker import Faker

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class ChartData:
    patient: dict[str, Any]
    encounter: dict[str, Any]
    provider: dict[str, Any]
    facility: dict[str, Any]
    insurance: dict[str, Any]
    chief_complaint: str
    hpi: str
    ros: list[str]
    physical_exam: list[str]
    vitals: dict[str, Any]
    diagnoses: list[dict[str, str]]
    procedures: list[dict[str, str]]
    medications: list[str]
    assessment_plan: list[str]
    addenda: list[dict[str, str]] = field(default_factory=list)


def _load_csv_dict(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _format(template: str, **values: Any) -> str:
    # str.format would crash on stray braces in clinical text, so do a
    # light placeholder substitution instead.
    out = template
    for key, val in values.items():
        out = out.replace("{" + key + "}", str(val))
    return out


def _random_dob(faker: Faker, min_age: int = 25, max_age: int = 88) -> date:
    today = date.today()
    age = random.randint(min_age, max_age)
    start = today.replace(year=today.year - age - 1) + timedelta(days=1)
    end = today.replace(year=today.year - age)
    return faker.date_between(start_date=start, end_date=end)


def _vitals(rng: random.Random) -> dict[str, Any]:
    return {
        "bp_sys": rng.randint(108, 158),
        "bp_dia": rng.randint(62, 96),
        "hr": rng.randint(58, 102),
        "rr": rng.randint(12, 22),
        "temp": round(rng.uniform(97.4, 99.6), 1),
        "spo2": rng.randint(94, 100),
        "weight_lb": rng.randint(120, 280),
        "height_in": rng.randint(60, 76),
    }


def generate_chart_data(
    seed: int | None = None,
    *,
    n_diagnoses: int = 2,
    n_procedures: int = 2,
    n_medications: int = 3,
    include_addendum: bool = False,
) -> ChartData:
    """Build a single synthetic chart payload.

    `seed` makes a chart reproducible; pass None for fresh randomness.
    """

    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    faker = Faker("en_US")
    rng = random.Random(seed)

    icd = _load_csv_dict(DATA_DIR / "icd10_codes.csv")
    cpt = _load_csv_dict(DATA_DIR / "cpt_codes.csv")
    templates = _load_json(DATA_DIR / "clinical_templates.json")

    sex = rng.choice(["male", "female"])
    dob = _random_dob(faker)
    age = (date.today() - dob).days // 365
    encounter_dt = faker.date_time_between(start_date="-90d", end_date="now")

    patient = {
        "name": faker.name_male() if sex == "male" else faker.name_female(),
        "dob": dob.strftime("%m/%d/%Y"),
        "age": age,
        "sex": sex.capitalize(),
        "mrn": f"MRN-{rng.randint(100000, 999999)}",
        "address": faker.address().replace("\n", ", "),
        "phone": faker.phone_number(),
    }

    facility_name = rng.choice(templates["facilities"])
    facility = {
        "name": facility_name,
        "address": faker.address().replace("\n", ", "),
        "phone": faker.phone_number(),
        "fax": faker.phone_number(),
        "npi": str(rng.randint(1000000000, 1999999999)),
    }

    provider = {
        "name": f"Dr. {faker.last_name()}, {rng.choice(['MD', 'DO'])}",
        "npi": str(rng.randint(1000000000, 1999999999)),
        "specialty": rng.choice(templates["specialties"]).title(),
    }

    insurance = {
        "payer": rng.choice(templates["insurance_payers"]),
        "member_id": f"{rng.choice(['XJB', 'YHU', 'KLM', 'PRA'])}{rng.randint(100000000, 999999999)}",
        "group": f"GRP{rng.randint(10000, 99999)}",
        "subscriber": patient["name"],
    }

    encounter = {
        "datetime": encounter_dt.strftime("%m/%d/%Y %H:%M"),
        "type": rng.choice(["Office Visit", "Follow-up", "New Patient", "Urgent Care"]),
        "account": f"ACC{rng.randint(10000000, 99999999)}",
        "encounter_id": f"ENC{rng.randint(10000, 99999)}",
    }

    symptom = rng.choice(templates["symptoms"])
    chronic = rng.choice(templates["chronic_conditions"])
    chief = _format(
        rng.choice(templates["chief_complaints"]),
        duration=rng.randint(1, 14),
        symptom=symptom,
        age=age,
        sex=sex,
        chronic_condition=chronic,
    )

    hpi_lines = []
    for tmpl in rng.sample(templates["hpi_phrases"], k=min(3, len(templates["hpi_phrases"]))):
        hpi_lines.append(
            _format(
                tmpl,
                quality=rng.choice(templates["pain_quality"]),
                severity=rng.randint(2, 9),
                pattern=rng.choice(templates["pain_pattern"]),
                trigger=rng.choice(templates["triggers"]),
                reliever=rng.choice(templates["relievers"]),
                denied_symptom=rng.choice(templates["denied_symptoms"]),
            )
        )
    hpi = " ".join(hpi_lines)

    ros = rng.sample(templates["ros_systems"], k=min(6, len(templates["ros_systems"])))
    vitals = _vitals(rng)
    pe = []
    for line in templates["physical_exam"]:
        pe.append(_format(line, **vitals))

    diag_picks = rng.sample(icd, k=min(n_diagnoses, len(icd)))
    diagnoses = [{"code": d["code"], "description": d["description"]} for d in diag_picks]

    proc_picks = rng.sample(cpt, k=min(n_procedures, len(cpt)))
    procedures = [{"code": p["code"], "description": p["description"]} for p in proc_picks]

    medications = rng.sample(templates["medications"], k=min(n_medications, len(templates["medications"])))

    primary_dx = diagnoses[0]["description"] if diagnoses else chronic
    assessment_plan = []
    for tmpl in rng.sample(templates["assessment_phrases"], k=min(3, len(templates["assessment_phrases"]))):
        assessment_plan.append(
            _format(
                tmpl,
                condition=primary_dx,
                followup_weeks=rng.choice([2, 4, 6, 8, 12]),
                medication=rng.choice(templates["medications"]).split()[0],
                specialty=rng.choice(templates["specialties"]),
            )
        )

    addenda: list[dict[str, str]] = []
    if include_addendum:
        addenda.append(
            {
                "datetime": (encounter_dt + timedelta(days=rng.randint(1, 5))).strftime("%m/%d/%Y %H:%M"),
                "author": provider["name"],
                "text": (
                    "Lab results reviewed. "
                    + rng.choice(
                        [
                            "Hemoglobin A1C returned at 7.4% — recommend medication adjustment.",
                            "Lipid panel within target range; continue statin therapy.",
                            "TSH elevated at 6.2 — increase levothyroxine and recheck in 6 weeks.",
                            "BMP unremarkable. No further action needed at this time.",
                        ]
                    )
                ),
            }
        )

    return ChartData(
        patient=patient,
        encounter=encounter,
        provider=provider,
        facility=facility,
        insurance=insurance,
        chief_complaint=chief,
        hpi=hpi,
        ros=ros,
        physical_exam=pe,
        vitals=vitals,
        diagnoses=diagnoses,
        procedures=procedures,
        medications=medications,
        assessment_plan=assessment_plan,
        addenda=addenda,
    )
