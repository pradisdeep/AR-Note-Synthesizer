"""Core dataclasses for the medcoding pipeline.

The pipeline produces stage-by-stage immutable records so each step can be
serialized, audited, and replaced independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

CodeType = Literal["ICD-10", "CPT"]


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space bounding box on a single page."""

    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class TextBlock:
    """One OCR'd or VLM-emitted text region with its provenance."""

    text: str
    bbox: BoundingBox
    confidence: float
    page: int


@dataclass
class Page:
    """A single page after extraction. Holds the raw OCR text plus the
    layout-aware blocks that downstream stages use to reconstruct sections.
    """

    page_number: int
    width: int
    height: int
    blocks: list[TextBlock] = field(default_factory=list)
    full_text: str = ""


# Canonical section names. Keep aligned with the synthetic chart generator's
# section headings; the normalizer uses these as targets when classifying lines.
SECTION_NAMES = (
    "header",
    "patient",
    "encounter",
    "insurance",
    "chief_complaint",
    "history_of_present_illness",
    "vital_signs",
    "review_of_systems",
    "physical_examination",
    "diagnoses",
    "procedures",
    "medications",
    "plan",
    "addendum",
    "signature",
    "unknown",
)


@dataclass
class Section:
    """A logical section of a chart (HPI, ROS, Diagnoses, etc.).

    `noise_classification` is set by the noise filter during normalization:
    "relevant" / "noise" / "uncertain". Downstream stages (chunker, coder)
    use this to decide whether to send the section to the LLM.
    """

    name: str
    title: str
    text: str
    pages: list[int] = field(default_factory=list)
    noise_classification: str = "relevant"
    noise_reason: str = ""


@dataclass
class CodeRow:
    """One ICD-10 or CPT row extracted from a structured table."""

    code: str
    description: str
    page: int
    raw_line: str


@dataclass
class ExtractedChart:
    """Output of stage 2 (extraction) + stage 3 (normalization).

    Production ingestion writes one of these per TIFF; downstream coding
    operates only on this structure.
    """

    source_path: str
    extracted_at: datetime
    extractor: str
    pages: list[Page]
    sections: list[Section]
    icd_rows: list[CodeRow] = field(default_factory=list)
    cpt_rows: list[CodeRow] = field(default_factory=list)
    markdown: str = ""
    extractor_metadata: dict = field(default_factory=dict)

    def section(self, name: str) -> Optional[Section]:
        for s in self.sections:
            if s.name == name:
                return s
        return None


@dataclass
class CodeSuggestion:
    """One LLM-produced code with provenance.

    `evidence_quote` must be a verbatim span from the chart so reviewers can
    audit the suggestion. `confidence` is the LLM's self-reported probability
    on a 0-1 scale; downstream policy decides the rejection threshold.
    """

    code_type: CodeType
    code: str
    description: str
    evidence_quote: str
    confidence: float
    source_section: str = ""


@dataclass
class CodingResult:
    """Per-chart output of the coding stage."""

    chart_source: str
    coder: str
    coded_at: datetime
    icd_suggestions: list[CodeSuggestion] = field(default_factory=list)
    cpt_suggestions: list[CodeSuggestion] = field(default_factory=list)
    raw_responses: dict[str, str] = field(default_factory=dict)
    coder_metadata: dict = field(default_factory=dict)
