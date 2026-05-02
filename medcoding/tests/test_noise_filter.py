"""Tests for the NoiseFilter and its three layered checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medcoding.models import Section  # noqa: E402
from medcoding.noise_filter import NoiseFilter  # noqa: E402


def _section(name: str, title: str, text: str) -> Section:
    return Section(name=name, title=title, text=text)


# ---- Layer 1: header patterns -------------------------------------------


def test_orders_header_classified_as_noise():
    f = NoiseFilter()
    s = _section("noise_orders", "Orders Placed This Visit", "any text")
    # Sections already tagged at parse time bypass the filter — direct
    # check uses a generic name to exercise the header pattern itself.
    s2 = _section("unknown", "Provider Orders", "Order: CMP\nStatus: Pending")
    assert f.classify(s2).kind == "noise"
    assert f.classify(s2).reason == "header_pattern"


def test_prescriptions_header_classified_as_noise():
    f = NoiseFilter()
    s = _section("unknown", "Rx / Prescriptions", "Drug: Lisinopril\nSig: 10mg daily")
    assert f.classify(s).kind == "noise"


def test_referrals_header_classified_as_noise():
    f = NoiseFilter()
    s = _section("unknown", "Referrals", "Cardiology — Eval palpitations")
    assert f.classify(s).kind == "noise"


def test_fax_cover_classified_as_noise():
    f = NoiseFilter()
    s = _section("unknown", "Fax Cover Sheet", "To: 555-1234")
    assert f.classify(s).kind == "noise"


# ---- Layer 2: table-structure cues --------------------------------------


def test_table_header_tokens_trip_layer_2():
    f = NoiseFilter()
    # Header isn't a known noise phrase, but the column header line is.
    text = "Order Status Authorized Date\nCMP Pending Dr. Smith 01/02/2025"
    s = _section("unknown", "Visit Items", text)
    result = f.classify(s)
    assert result.kind == "noise"
    assert result.reason == "table_header_tokens"


def test_status_density_trips_layer_2():
    f = NoiseFilter()
    text = (
        "Some random heading line\n"
        "Lab order pending\n"
        "Imaging order sent\n"
        "Pharmacy order acknowledged\n"
    )
    s = _section("unknown", "Action Items", text)
    result = f.classify(s)
    assert result.kind == "noise"
    assert result.reason in {"order_status_density", "line_pattern_density"}


# ---- Layer 3: line-pattern density --------------------------------------


def test_line_pattern_density_trips_layer_3():
    f = NoiseFilter()
    text = (
        "Today's items:\n"
        "Send to: Quest Diagnostics\n"
        "Refer to: Cardiology\n"
        "Order: TSH\n"
    )
    s = _section("unknown", "Today's Action", text)
    result = f.classify(s)
    assert result.kind == "noise"


# ---- Protected sections survive even when text is suspicious ------------


def test_protected_section_never_classified_as_noise():
    f = NoiseFilter()
    # An HPI mentioning that something was ordered should stay relevant.
    s = _section(
        "history_of_present_illness",
        "History of Present Illness",
        "Patient was previously ordered to take Lisinopril; reports compliance.",
    )
    assert f.classify(s).kind == "relevant"


def test_diagnoses_section_never_classified_as_noise():
    f = NoiseFilter()
    s = _section(
        "diagnoses",
        "Assessment / Diagnoses",
        "ICD-10 R53.83 Other fatigue",
    )
    assert f.classify(s).kind == "relevant"


# ---- Innocuous sections kept by default ---------------------------------


def test_signature_section_kept():
    f = NoiseFilter()
    s = _section("signature", "Signature", "Electronically signed by Dr. X")
    assert f.classify(s).kind == "relevant"


def test_unknown_section_with_no_red_flags_returns_uncertain():
    f = NoiseFilter()
    s = _section("custom_section", "Some New Header", "Free text without order language.")
    result = f.classify(s)
    # Default-keep covers known-innocuous names; a brand-new name without
    # red flags should be flagged uncertain so a human can extend patterns.
    assert result.kind == "uncertain"


# ---- End-to-end via normalizer ------------------------------------------


def test_normalizer_tags_orders_section_via_header_recognition():
    from medcoding.models import Page
    from medcoding.normalizer import normalize

    pages = [
        Page(
            page_number=1,
            width=1700,
            height=2200,
            blocks=[],
            full_text="\n".join(
                [
                    "Chief Complaint",
                    "Patient with cough.",
                    "Assessment / Diagnoses (ICD-10)",
                    "ICD-10 Description",
                    "J06.9 Acute upper respiratory infection",
                    "Provider Orders",
                    "Order Status Authorized Date",
                    "CMP Pending Dr. Smith 01/02/2025",
                    "Plan",
                    "Reassess in 1 week.",
                ]
            ),
        )
    ]
    chart = normalize(pages, source_path="x.tiff", extractor_name="test")
    by_name = {s.name: s for s in chart.sections}

    # Orders section was recognized by the parse-time pattern.
    assert "noise_orders" in by_name
    assert by_name["noise_orders"].noise_classification == "noise"

    # Diagnoses survived as relevant.
    assert by_name["diagnoses"].noise_classification == "relevant"

    # Markdown should NOT contain the orders content.
    assert "Provider Orders" not in chart.markdown
    assert "CMP Pending" not in chart.markdown
    # But it SHOULD contain the legitimate diagnoses table.
    assert "J06.9" in chart.markdown
