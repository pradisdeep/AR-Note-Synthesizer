"""Tests for the coder protocol implementations."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medcoding.coders import load_coder  # noqa: E402
from medcoding.coders.lm_studio import _coerce_suggestions, _parse_json_response  # noqa: E402
from medcoding.coders.mock import MockCoder  # noqa: E402
from medcoding.models import CodeRow, ExtractedChart  # noqa: E402


def _chart_with_codes() -> ExtractedChart:
    return ExtractedChart(
        source_path="x.tiff",
        extracted_at=datetime.now(timezone.utc),
        extractor="test",
        pages=[],
        sections=[],
        icd_rows=[
            CodeRow(code="E11.9", description="Type 2 diabetes", page=1, raw_line="E11.9 Type 2 diabetes")
        ],
        cpt_rows=[
            CodeRow(code="99213", description="Office visit", page=1, raw_line="99213 Office visit")
        ],
    )


def test_load_coder_resolves_known_backends():
    assert load_coder("mock").name == "mock"


def test_load_coder_rejects_unknown_backend():
    with pytest.raises(ValueError):
        load_coder("definitely-not-a-coder")


def test_mock_echo_returns_normalizer_codes():
    chart = _chart_with_codes()
    coder = MockCoder()
    result = coder.code(chart)
    icd_codes = {s.code for s in result.icd_suggestions}
    cpt_codes = {s.code for s in result.cpt_suggestions}
    assert icd_codes == {"E11.9"}
    assert cpt_codes == {"99213"}
    assert all(0.9 <= s.confidence <= 1.0 for s in result.icd_suggestions + result.cpt_suggestions)


def test_mock_scripted_overrides_normalizer(tmp_path: Path):
    scripted = {
        "x.tiff": {
            "diagnoses": [
                {"code": "R51.9", "description": "Headache", "evidence": "patient with HA", "confidence": 0.8}
            ],
            "procedures": [
                {"code": "99214", "description": "Office visit moderate", "evidence": "MDM moderate", "confidence": 0.75}
            ],
        }
    }
    fp = tmp_path / "scripted.json"
    fp.write_text(json.dumps(scripted))
    coder = MockCoder(scripted_responses_path=fp)
    result = coder.code(_chart_with_codes())
    assert {s.code for s in result.icd_suggestions} == {"R51.9"}
    assert {s.code for s in result.cpt_suggestions} == {"99214"}


def test_parse_json_response_handles_clean_json():
    parsed = _parse_json_response('{"diagnoses": [{"code": "I10"}]}')
    assert parsed == {"diagnoses": [{"code": "I10"}]}


def test_parse_json_response_extracts_object_from_prose():
    text = 'Here you go:\n```json\n{"diagnoses": [{"code": "I10"}]}\n```'
    parsed = _parse_json_response(text)
    assert parsed == {"diagnoses": [{"code": "I10"}]}


def test_parse_json_response_returns_none_on_garbage():
    assert _parse_json_response("totally not json") is None
    assert _parse_json_response("") is None


def test_coerce_suggestions_clamps_confidence_and_drops_empty_codes():
    rows = [
        {"code": "I10", "description": "HTN", "evidence": "BP elevated", "confidence": 1.5},
        {"code": "", "description": "blank", "evidence": "", "confidence": 0.5},
        {"code": "E11.9", "description": "DM2", "evidence": "diabetes", "confidence": -0.3},
        "not a dict",
    ]
    out = _coerce_suggestions(rows, "ICD-10", ["diagnoses"])
    codes = [s.code for s in out]
    assert codes == ["I10", "E11.9"]
    assert out[0].confidence == 1.0  # clamped down
    assert out[1].confidence == 0.0  # clamped up
