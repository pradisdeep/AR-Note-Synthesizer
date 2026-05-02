"""Unit tests for the normalizer's section detection and code extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medcoding.models import Page  # noqa: E402
from medcoding.normalizer import (  # noqa: E402
    _CPT_CODE,
    _ICD_CODE,
    _classify_line,
    normalize,
)


def _page(page_number: int, text: str) -> Page:
    return Page(page_number=page_number, width=1700, height=2200, blocks=[], full_text=text)


def test_classify_line_recognizes_canonical_headers():
    assert _classify_line("History of Present Illness") == ("history_of_present_illness", "History of Present Illness")
    assert _classify_line("HISTORY OF PRESENT ILLNESS")[0] == "history_of_present_illness"
    assert _classify_line("Assessment / Diagnoses (ICD-10)")[0] == "diagnoses"
    assert _classify_line("Procedures / Services (CPT)")[0] == "procedures"
    assert _classify_line("Active Medications")[0] == "medications"


def test_classify_line_ignores_non_headers():
    assert _classify_line("Patient presents with fatigue and shortness of breath.") is None
    assert _classify_line("BP 137/96, HR 65, RR 18, Temp 97.6 F, SpO2 96%") is None
    assert _classify_line("") is None


def test_icd_code_pattern():
    assert _ICD_CODE.findall("Diagnosis E11.9 noted") == ["E11.9"]
    assert _ICD_CODE.findall("M25.561 and I10 confirmed") == ["M25.561", "I10"]
    # Should not match U-prefixed special codes
    assert _ICD_CODE.findall("U07.1") == []


def test_cpt_code_pattern_matches_cpt_and_hcpcs():
    assert _CPT_CODE.findall("CPT 99213 billed") == ["99213"]
    assert _CPT_CODE.findall("HCPCS G0438 included") == ["G0438"]


def test_normalize_extracts_codes_from_table_section():
    pages = [
        _page(
            1,
            "\n".join(
                [
                    "Heritage Community Hospital",
                    "Patient: Jane Doe",
                    "Chief Complaint",
                    "Patient presents with a 9-day history of fever.",
                    "Assessment / Diagnoses (ICD-10)",
                    "ICD-10 Description",
                    "R53.83 Other fatigue",
                    "I10 Essential hypertension",
                    "Procedures / Services (CPT)",
                    "CPT Description",
                    "99213 Office visit established patient low complexity",
                    "Plan",
                    "Continue current management.",
                ]
            ),
        )
    ]
    chart = normalize(pages, source_path="x.tiff", extractor_name="test")

    icd_codes = {r.code for r in chart.icd_rows}
    cpt_codes = {r.code for r in chart.cpt_rows}
    assert icd_codes == {"R53.83", "I10"}
    assert cpt_codes == {"99213"}

    section_names = {s.name for s in chart.sections}
    assert {"chief_complaint", "diagnoses", "procedures", "plan"}.issubset(section_names)
    assert "## Assessment / Diagnoses" in chart.markdown
    assert "| R53.83 |" in chart.markdown


def test_normalize_falls_back_to_full_document_when_table_missing():
    pages = [_page(1, "I10 Essential hypertension noted in passing.")]
    chart = normalize(pages, source_path="x.tiff", extractor_name="test")
    assert "I10" in {r.code for r in chart.icd_rows}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
