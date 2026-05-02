"""Convert raw extracted pages into a section-tagged ExtractedChart.

The normalizer is heuristic on purpose: it should be cheap, predictable, and
produce a structure phi-4 can consume. If the heuristics fail on a chart
type, plug in a different normalizer rather than making this one smarter.

Two responsibilities:
1. Identify section headers in OCR text and group following lines under them.
2. Pull structured ICD-10 / CPT rows out of tables (they have a known shape:
   "<CODE>  <DESCRIPTION>").
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import CodeRow, ExtractedChart, Page, Section

log = logging.getLogger(__name__)


# Map of canonical section name -> patterns the OCR might emit for the header.
# Patterns are lowercase, whitespace-flexible. Order matters only when a line
# could match multiple (the longer/more specific pattern should be checked first).
_SECTION_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("history_of_present_illness", "History of Present Illness",
     re.compile(r"^\s*history\s+of\s+present\s+illness\s*$", re.I)),
    ("review_of_systems", "Review of Systems",
     re.compile(r"^\s*review\s+of\s+systems\s*$", re.I)),
    ("physical_examination", "Physical Examination",
     re.compile(r"^\s*physical\s+examination\s*$", re.I)),
    ("chief_complaint", "Chief Complaint",
     re.compile(r"^\s*chief\s+complaint\s*$", re.I)),
    ("vital_signs", "Vital Signs",
     re.compile(r"^\s*vital\s+signs\s*$", re.I)),
    ("diagnoses", "Assessment / Diagnoses",
     re.compile(r"^\s*(assessment\s*/\s*)?diagnoses?(\s*\(?\s*icd[\s\-]?10\)?)?\s*$", re.I)),
    ("procedures", "Procedures / Services",
     re.compile(r"^\s*procedures?(\s*/\s*services?)?(\s*\(?\s*cpt\)?)?\s*$", re.I)),
    ("medications", "Active Medications",
     re.compile(r"^\s*(active\s+)?medications?\s*$", re.I)),
    ("plan", "Plan",
     re.compile(r"^\s*plan\s*$", re.I)),
    ("addendum", "Addendum",
     re.compile(r"^\s*addendum\s*$", re.I)),
    ("insurance", "Insurance",
     re.compile(r"^\s*insurance\s*$", re.I)),
    ("signature", "Signature",
     re.compile(r"^\s*(electronically\s+signed|signed\s+by)\b.*$", re.I)),
]


# ICD-10 format: letter + 2 digits, optionally a dot + up to 4 alphanumerics.
# Examples: I10, E11.9, M25.561, Z79.899, R06.02
_ICD_CODE = re.compile(r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b")

# CPT format: 5 digits, OR HCPCS Level II (1 letter + 4 digits, e.g. G0438).
_CPT_CODE = re.compile(r"\b((?:[0-9]{5})|(?:[A-V][0-9]{4}))\b")


def _classify_line(line: str) -> tuple[str, str] | None:
    """Return (section_name, canonical_title) if `line` is a section header."""

    stripped = line.strip()
    # Short heuristic guard: section headers in our charts are 1-5 words.
    word_count = len(stripped.split())
    if not stripped or word_count > 6:
        return None
    for name, title, pattern in _SECTION_PATTERNS:
        if pattern.search(stripped):
            return name, title
    return None


def _extract_code_rows(
    section_text: str,
    pattern: re.Pattern[str],
    page_of: dict[str, int],
) -> list[CodeRow]:
    """Pull (code, description) pairs from a section's text.

    OCR'd table rows look like: "I10  Essential (primary) hypertension".
    We anchor on the code, then take the rest of the line as description.
    """

    rows: list[CodeRow] = []
    seen: set[tuple[str, str]] = set()
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = pattern.search(line)
        if not m:
            continue
        code = m.group(1).upper()
        # Description is everything after the code; if it leads with the
        # column header word, drop it.
        after = line[m.end():].strip(" -|\t:")
        if after.lower().startswith(("description", "desc")):
            after = after.split(maxsplit=1)[1] if " " in after else ""
        # If description is empty (header row, or code alone), skip — likely
        # a stray code embedded in a paragraph rather than a real table row.
        if not after:
            continue
        key = (code, after.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            CodeRow(
                code=code,
                description=after,
                page=page_of.get(line, 0),
                raw_line=raw_line,
            )
        )
    return rows


def _build_page_index(pages: list[Page]) -> dict[str, int]:
    """Map OCR line text -> page number. Used to attribute code rows."""

    index: dict[str, int] = {}
    for page in pages:
        for line in page.full_text.splitlines():
            stripped = line.strip()
            if stripped and stripped not in index:
                index[stripped] = page.page_number
    return index


def normalize(
    pages: list[Page],
    *,
    source_path: Path | str,
    extractor_name: str,
    extractor_metadata: dict | None = None,
) -> ExtractedChart:
    """Group OCR lines into sections, pull code rows, render Markdown."""

    # Concatenate all pages with a page marker so we can attribute line -> page.
    line_to_page: list[tuple[str, int]] = []
    for page in pages:
        for line in page.full_text.splitlines():
            line_to_page.append((line, page.page_number))

    # Walk lines, switching sections whenever we see a header.
    current_name = "header"
    current_title = "Header"
    section_pages: dict[str, list[int]] = {}
    section_lines: dict[str, list[str]] = {}
    section_titles: dict[str, str] = {current_name: current_title}

    for line, page_num in line_to_page:
        classified = _classify_line(line)
        if classified is not None:
            current_name, current_title = classified
            section_titles[current_name] = current_title
            section_lines.setdefault(current_name, [])
            section_pages.setdefault(current_name, [])
            if page_num not in section_pages[current_name]:
                section_pages[current_name].append(page_num)
            continue
        section_lines.setdefault(current_name, []).append(line)
        section_pages.setdefault(current_name, [])
        if page_num not in section_pages[current_name]:
            section_pages[current_name].append(page_num)

    sections: list[Section] = []
    for name, lines in section_lines.items():
        sections.append(
            Section(
                name=name,
                title=section_titles.get(name, name.replace("_", " ").title()),
                text="\n".join(lines).strip(),
                pages=section_pages.get(name, []),
            )
        )

    page_of = _build_page_index(pages)

    # Code extraction: prefer the dedicated section, but fall back to a
    # whole-document scan so we don't miss codes the section detector lost.
    icd_section = next((s for s in sections if s.name == "diagnoses"), None)
    cpt_section = next((s for s in sections if s.name == "procedures"), None)
    full_text = "\n".join(p.full_text for p in pages)

    icd_rows = _extract_code_rows(
        icd_section.text if icd_section else full_text, _ICD_CODE, page_of
    )
    cpt_rows = _extract_code_rows(
        cpt_section.text if cpt_section else full_text, _CPT_CODE, page_of
    )

    markdown = _render_markdown(sections, icd_rows, cpt_rows)

    return ExtractedChart(
        source_path=str(source_path),
        extracted_at=datetime.now(timezone.utc),
        extractor=extractor_name,
        pages=pages,
        sections=sections,
        icd_rows=icd_rows,
        cpt_rows=cpt_rows,
        markdown=markdown,
        extractor_metadata=extractor_metadata or {},
    )


def _render_markdown(
    sections: list[Section],
    icd_rows: list[CodeRow],
    cpt_rows: list[CodeRow],
) -> str:
    """Produce a deterministic Markdown rendering for downstream LLM prompts.

    Sections are emitted in canonical clinical order regardless of OCR order,
    so phi-4 sees a consistent layout across charts.
    """

    by_name = {s.name: s for s in sections}
    canonical_order = [
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
    ]

    out: list[str] = []
    out.append("# Clinical Chart")
    for name in canonical_order:
        section = by_name.get(name)
        if not section or not section.text.strip():
            if name == "diagnoses" and icd_rows:
                pass  # render below from rows even if section text is empty
            elif name == "procedures" and cpt_rows:
                pass
            else:
                continue
        title = section.title if section else name.replace("_", " ").title()
        out.append(f"\n## {title}\n")
        if name == "diagnoses" and icd_rows:
            out.append("| ICD-10 | Description |")
            out.append("| --- | --- |")
            for row in icd_rows:
                out.append(f"| {row.code} | {row.description} |")
        elif name == "procedures" and cpt_rows:
            out.append("| CPT | Description |")
            out.append("| --- | --- |")
            for row in cpt_rows:
                out.append(f"| {row.code} | {row.description} |")
        elif section:
            out.append(section.text)

    # Capture anything the heuristic misclassified rather than dropping it.
    other = by_name.get("unknown")
    if other and other.text.strip():
        out.append("\n## Other (Unclassified)\n")
        out.append(other.text)

    return "\n".join(out).strip() + "\n"
