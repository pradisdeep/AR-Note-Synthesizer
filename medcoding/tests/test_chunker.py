"""Tests for section-based chunking."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medcoding.chunker import chunk_for_coding  # noqa: E402
from medcoding.models import ExtractedChart, Section  # noqa: E402


def _chart(*sections: Section) -> ExtractedChart:
    from datetime import datetime, timezone

    return ExtractedChart(
        source_path="x.tiff",
        extracted_at=datetime.now(timezone.utc),
        extractor="test",
        pages=[],
        sections=list(sections),
    )


def test_chunk_emits_one_per_purpose_when_small():
    chart = _chart(
        Section("chief_complaint", "Chief Complaint", "Patient with fever."),
        Section("diagnoses", "Assessment / Diagnoses", "R50.9 Fever unspecified"),
        Section("procedures", "Procedures / Services", "99213 Office visit"),
        Section("plan", "Plan", "Follow up in 1 week."),
    )
    chunks = chunk_for_coding(chart)
    purposes = [c.purpose for c in chunks]
    assert purposes.count("icd") == 1
    assert purposes.count("cpt") == 1


def test_chunk_includes_only_relevant_sections():
    chart = _chart(
        Section("chief_complaint", "Chief Complaint", "Headache."),
        Section("diagnoses", "Assessment / Diagnoses", "R51.9 Headache"),
        Section("procedures", "Procedures / Services", "99213 Office visit"),
        Section("insurance", "Insurance", "Aetna PPO"),
    )
    chunks = chunk_for_coding(chart)
    icd_chunk = next(c for c in chunks if c.purpose == "icd")
    cpt_chunk = next(c for c in chunks if c.purpose == "cpt")
    # Insurance shouldn't appear in either prompt — it's not coding evidence.
    assert "insurance" not in icd_chunk.sections_included
    assert "insurance" not in cpt_chunk.sections_included
    # ICD prompt has diagnoses, CPT prompt has procedures.
    assert "diagnoses" in icd_chunk.sections_included
    assert "procedures" in cpt_chunk.sections_included


def test_chunk_skips_purpose_when_no_relevant_text():
    chart = _chart(Section("insurance", "Insurance", "Aetna PPO"))
    chunks = chunk_for_coding(chart)
    # Insurance alone shouldn't yield any chunks at all.
    assert len(chunks) == 0


def test_chunk_splits_when_over_budget():
    long_text = "x" * 8000
    chart = _chart(
        Section("history_of_present_illness", "History of Present Illness", long_text),
        Section("physical_examination", "Physical Examination", long_text),
        Section("diagnoses", "Assessment / Diagnoses", "R51.9 Headache"),
    )
    chunks = chunk_for_coding(chart, char_budget=10_000)
    icd_chunks = [c for c in chunks if c.purpose == "icd"]
    # Three large sections should not fit in one 10k-char chunk.
    assert len(icd_chunks) >= 2
