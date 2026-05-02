"""Mock coder for testing without a real LLM.

Two operating modes:
1. **Echo mode (default):** returns the structured codes that the normalizer
   already extracted, with a fixed confidence. Useful as a pipeline-wiring
   test and as a "baseline" for evaluating whether the LLM actually adds
   value over deterministic table parsing.
2. **Scripted mode:** load canned responses from a JSON file. Useful for
   replaying a real LM Studio session against tests / CI.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import CodeSuggestion, CodingResult, ExtractedChart


class MockCoder:
    name = "mock"

    def __init__(
        self,
        scripted_responses_path: str | Path | None = None,
        echo_confidence: float = 0.95,
    ) -> None:
        self._echo_confidence = echo_confidence
        self._scripted: dict[str, Any] | None = None
        if scripted_responses_path:
            with Path(scripted_responses_path).open(encoding="utf-8") as fh:
                self._scripted = json.load(fh)

    def code(self, chart: ExtractedChart) -> CodingResult:
        if self._scripted and chart.source_path in self._scripted:
            return self._from_scripted(chart, self._scripted[chart.source_path])
        return self._from_echo(chart)

    def _from_echo(self, chart: ExtractedChart) -> CodingResult:
        icd = [
            CodeSuggestion(
                code_type="ICD-10",
                code=row.code,
                description=row.description,
                evidence_quote=row.raw_line,
                confidence=self._echo_confidence,
                source_section="diagnoses",
            )
            for row in chart.icd_rows
        ]
        cpt = [
            CodeSuggestion(
                code_type="CPT",
                code=row.code,
                description=row.description,
                evidence_quote=row.raw_line,
                confidence=self._echo_confidence,
                source_section="procedures",
            )
            for row in chart.cpt_rows
        ]
        return CodingResult(
            chart_source=chart.source_path,
            coder=self.name,
            coded_at=datetime.now(timezone.utc),
            icd_suggestions=icd,
            cpt_suggestions=cpt,
            raw_responses={"mode": "echo"},
            coder_metadata={"mode": "echo"},
        )

    def _from_scripted(self, chart: ExtractedChart, payload: dict) -> CodingResult:
        def _build(rows: list[dict], code_type: str) -> list[CodeSuggestion]:
            out: list[CodeSuggestion] = []
            for row in rows or []:
                out.append(
                    CodeSuggestion(
                        code_type=code_type,  # type: ignore[arg-type]
                        code=str(row.get("code", "")).upper(),
                        description=str(row.get("description", "")),
                        evidence_quote=str(row.get("evidence", "")),
                        confidence=float(row.get("confidence", 0.0)),
                        source_section="scripted",
                    )
                )
            return out

        return CodingResult(
            chart_source=chart.source_path,
            coder=self.name,
            coded_at=datetime.now(timezone.utc),
            icd_suggestions=_build(payload.get("diagnoses", []), "ICD-10"),
            cpt_suggestions=_build(payload.get("procedures", []), "CPT"),
            raw_responses={"mode": "scripted"},
            coder_metadata={"mode": "scripted"},
        )
