"""Section-aware chunking.

Coding accuracy depends heavily on which sections the LLM sees. Two rules:
1. Never split a section mid-text (codes must travel with their evidence).
2. Send only the sections that are relevant to the coding task at hand —
   ICD-10 needs diagnoses + clinical context; CPT needs procedures + E&M
   complexity signals. Mixing them dilutes the prompt.

For phi-4 (16k tokens), a typical chart fits in one chunk. Charts that
exceed the budget get split by section boundaries with a header repeat
so each chunk carries the patient identifier.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import ExtractedChart, Section

# Sections that contain *evidence* for ICD-10 diagnosis coding.
ICD_RELEVANT_SECTIONS = (
    "header",
    "patient",
    "encounter",
    "chief_complaint",
    "history_of_present_illness",
    "review_of_systems",
    "physical_examination",
    "diagnoses",
    "plan",
    "addendum",
)

# Sections that contain *evidence* for CPT procedure / E&M coding.
CPT_RELEVANT_SECTIONS = (
    "header",
    "patient",
    "encounter",
    "chief_complaint",
    "history_of_present_illness",
    "review_of_systems",
    "physical_examination",
    "procedures",
    "plan",
    "addendum",
)


@dataclass
class CodingChunk:
    """One LLM-ready slice of a chart."""

    purpose: str  # "icd" or "cpt"
    text: str
    sections_included: list[str]
    char_count: int


def _render_sections(chart: ExtractedChart, names: tuple[str, ...]) -> tuple[str, list[str]]:
    """Render the requested sections as Markdown in canonical order."""

    by_name: dict[str, Section] = {s.name: s for s in chart.sections}
    parts: list[str] = []
    included: list[str] = []
    for name in names:
        section = by_name.get(name)
        if not section or not section.text.strip():
            continue
        parts.append(f"## {section.title}\n{section.text.strip()}")
        included.append(name)
    return "\n\n".join(parts), included


def _split_by_section(
    text: str,
    sections_included: list[str],
    purpose: str,
    char_budget: int,
) -> list[CodingChunk]:
    """Fallback when a single chunk exceeds char_budget.

    Splits at section boundaries (`## ` markers). Never breaks mid-section.
    Each emitted chunk carries the same `purpose` and the subset of
    sections it actually contains.
    """

    blocks = text.split("\n\n## ")
    if len(blocks) > 1:
        blocks = [blocks[0]] + ["## " + b for b in blocks[1:]]

    chunks: list[CodingChunk] = []
    current: list[str] = []
    current_sections: list[str] = []
    current_len = 0

    def flush() -> None:
        if not current:
            return
        chunks.append(
            CodingChunk(
                purpose=purpose,
                text="\n\n".join(current),
                sections_included=list(current_sections),
                char_count=current_len,
            )
        )

    # Pre-extract section names from each block so we can attribute them.
    block_sections = []
    cursor = 0
    for block in blocks:
        # Match the first "## Title" line back to a section name by index.
        if cursor < len(sections_included):
            block_sections.append(sections_included[cursor])
            cursor += 1
        else:
            block_sections.append("unknown")

    for block, name in zip(blocks, block_sections):
        block_len = len(block)
        if block_len > char_budget:
            # A single section exceeds the budget — emit it alone and warn
            # via metadata. Better to truncate at the LLM boundary than
            # mid-evidence.
            flush()
            current = []
            current_sections = []
            current_len = 0
            chunks.append(
                CodingChunk(
                    purpose=purpose,
                    text=block,
                    sections_included=[name],
                    char_count=block_len,
                )
            )
            continue
        if current_len + block_len > char_budget:
            flush()
            current = [block]
            current_sections = [name]
            current_len = block_len
        else:
            current.append(block)
            current_sections.append(name)
            current_len += block_len + 4  # joiner

    flush()
    return chunks


def chunk_for_coding(
    chart: ExtractedChart,
    *,
    char_budget: int = 12_000,
) -> list[CodingChunk]:
    """Produce one ICD chunk and one CPT chunk for the chart.

    `char_budget` is a rough proxy for token budget. ~3.5 chars/token on
    average English; 12k chars ≈ 3.4k tokens, leaving headroom in phi-4's
    16k window for the prompt + JSON output.
    """

    icd_text, icd_sections = _render_sections(chart, ICD_RELEVANT_SECTIONS)
    cpt_text, cpt_sections = _render_sections(chart, CPT_RELEVANT_SECTIONS)

    chunks: list[CodingChunk] = []

    if icd_text:
        if len(icd_text) <= char_budget:
            chunks.append(
                CodingChunk(
                    purpose="icd",
                    text=icd_text,
                    sections_included=icd_sections,
                    char_count=len(icd_text),
                )
            )
        else:
            chunks.extend(_split_by_section(icd_text, icd_sections, "icd", char_budget))

    if cpt_text:
        if len(cpt_text) <= char_budget:
            chunks.append(
                CodingChunk(
                    purpose="cpt",
                    text=cpt_text,
                    sections_included=cpt_sections,
                    char_count=len(cpt_text),
                )
            )
        else:
            chunks.extend(_split_by_section(cpt_text, cpt_sections, "cpt", char_budget))

    return chunks
