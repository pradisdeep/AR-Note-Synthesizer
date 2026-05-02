"""Identify and drop sections that contain orders / Rx / referrals / forms.

These sections are NOT coding evidence. ICD coding rules forbid coding
suspected/planned conditions, and CPT requires services rendered (not
ordered). Letting these sections reach the LLM causes both compliance
failures and accuracy drops.

Three layered checks; the first one to fire wins:
1. **Header pattern** — fast, deterministic, configurable per-EHR.
2. **Table-structure** — looks for column headers typical of orders tables
   (Status / Pending / Sent / Send to:) regardless of the section's name.
3. **Content pattern** — line-level scan for order language even when
   neither the header nor a table betrays the section.

Result is `Classification` (relevant | noise | uncertain). Anything
`uncertain` is logged so a human can extend the patterns rather than
silently letting noise through or silently dropping clinical content.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from .models import Section

log = logging.getLogger(__name__)

ClassificationKind = Literal["relevant", "noise", "uncertain"]


# --- Layer 1: header patterns ---------------------------------------------

# Headers that are unambiguously NOT coding evidence. Case-insensitive.
# Extend per-EHR — keep one phrase per regex so logs show which one matched.
NOISE_HEADER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("orders_generic", re.compile(r"^\s*(provider\s+)?orders?\s*(placed.*)?$", re.I)),
    ("orders_outgoing", re.compile(r"^\s*outgoing\s+orders?\s*$", re.I)),
    ("orders_standing", re.compile(r"^\s*standing\s+orders?\s*$", re.I)),
    ("rx_prescriptions", re.compile(r"^\s*(rx|prescriptions?|meds?\s+prescribed.*)\s*/?\s*(prescriptions?)?\s*$", re.I)),
    ("referrals", re.compile(r"^\s*referrals?\s*$", re.I)),
    ("lab_orders", re.compile(r"^\s*lab(oratory)?\s+orders?\s*$", re.I)),
    ("imaging_orders", re.compile(r"^\s*imaging\s+orders?\s*$", re.I)),
    ("prior_auth", re.compile(r"^\s*prior\s+authoriz(ation|ations)?\s*$", re.I)),
    ("fax_cover", re.compile(r"^\s*fax\s+cover.*$", re.I)),
    ("routing_slip", re.compile(r"^\s*routing\s+slip\s*$", re.I)),
    ("patient_education", re.compile(r"^\s*patient\s+education.*$", re.I)),
    ("registration", re.compile(r"^\s*(registration|insurance\s+verification)\s+form?s?\s*$", re.I)),
)


# --- Layer 2: table-structure cues ----------------------------------------

# Column-header words that, when seen together near the top of a section,
# strongly suggest the section is an orders/Rx/referral table.
TABLE_HEADER_TOKENS = {
    "order",
    "orders",
    "status",
    "pending",
    "sent",
    "acknowledged",
    "discontinued",
    "drug",
    "sig",
    "refills",
    "pharmacy",
    "to",
    "reason",
    "urgency",
    "frequency",
    "authorized",
    "type",
}

ORDER_STATUS_TOKENS = {
    "pending",
    "sent",
    "acknowledged",
    "in process",
    "active",
    "discontinued",
    "on hold",
    "scheduled",
    "stat",
    "routine",
    "urgent",
    "awaiting response",
    "pending auth",
}


# --- Layer 3: content-pattern cues ----------------------------------------

LINE_ORDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsend\s+to\s*:", re.I),
    re.compile(r"\brefer\s+to\s*:", re.I),
    re.compile(r"\border\s*:", re.I),
    re.compile(r"\b(rx|prescription)\s*:", re.I),
    re.compile(r"\bauthorized\s+by\s*:", re.I),
    re.compile(r"\bquest\s+diagnostics\b", re.I),
    re.compile(r"\blabcorp\b", re.I),
    re.compile(r"\b(cvs|walgreens|rite\s*aid)\s+pharmacy\b", re.I),
    re.compile(r"\b(refills?|sig)\s*:", re.I),
)


@dataclass
class Classification:
    kind: ClassificationKind
    reason: str  # which check fired (or which one was inconclusive)
    matched_pattern: str = ""


@dataclass
class NoiseFilter:
    """Configurable filter; tweak `noise_header_patterns` per-EHR.

    Defaults are tuned for the synthetic chart generator's provider variants
    plus the most common real-world headers. Add to the list rather than
    replacing — the test suite covers the defaults.
    """

    noise_header_patterns: tuple[tuple[str, re.Pattern[str]], ...] = NOISE_HEADER_PATTERNS
    table_header_threshold: int = 3  # min table-token matches to trip layer 2
    line_pattern_density_threshold: float = 0.3  # fraction of lines matching → noise

    # Sections we never want to drop, even if they happen to mention an order
    # word (e.g., "ordered" appearing in a Plan).
    protected_sections: frozenset[str] = frozenset(
        {
            "chief_complaint",
            "history_of_present_illness",
            "review_of_systems",
            "physical_examination",
            "diagnoses",
            "procedures",
            "vital_signs",
        }
    )

    def classify(self, section: Section) -> Classification:
        # Protected sections always pass through, even if their text trips
        # layer 3. We trust the section taxonomy here — phi-4 needs to see
        # them to do its job.
        if section.name in self.protected_sections:
            return Classification(kind="relevant", reason="protected_section")

        # Layer 1: header pattern.
        title = (section.title or "").strip()
        for label, pattern in self.noise_header_patterns:
            if pattern.search(title):
                return Classification(
                    kind="noise", reason="header_pattern", matched_pattern=label
                )

        # Layer 2: table-structure heuristic.
        lines = [ln.strip() for ln in section.text.splitlines() if ln.strip()]
        if lines:
            head = lines[0].lower()
            head_tokens = set(re.findall(r"\b[a-z]+\b", head))
            if len(head_tokens & TABLE_HEADER_TOKENS) >= self.table_header_threshold:
                return Classification(kind="noise", reason="table_header_tokens")

            status_hits = 0
            for line in lines[: max(1, min(20, len(lines)))]:
                low = line.lower()
                for token in ORDER_STATUS_TOKENS:
                    if token in low:
                        status_hits += 1
                        break
            if status_hits >= 2 and status_hits / max(len(lines), 1) >= 0.2:
                return Classification(kind="noise", reason="order_status_density")

        # Layer 3: content-pattern density.
        if lines:
            matched = 0
            for line in lines:
                for pattern in LINE_ORDER_PATTERNS:
                    if pattern.search(line):
                        matched += 1
                        break
            density = matched / len(lines)
            if density >= self.line_pattern_density_threshold:
                return Classification(
                    kind="noise", reason="line_pattern_density"
                )

        # If the section name was already classified as something innocuous,
        # accept it. Otherwise flag uncertain so a human can extend patterns.
        if section.name in {"header", "patient", "encounter", "insurance", "medications", "plan", "addendum", "signature", "unknown"}:
            return Classification(kind="relevant", reason="default_keep")
        return Classification(kind="uncertain", reason="no_check_fired")
