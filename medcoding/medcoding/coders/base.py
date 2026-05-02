"""Coder protocol. Every coding backend (LM Studio, vLLM, hosted API) implements this."""

from __future__ import annotations

from typing import Protocol

from ..models import CodingResult, ExtractedChart


class Coder(Protocol):
    """Convert an ExtractedChart into a CodingResult.

    The protocol is deliberately minimal: input is a normalized chart, output
    is a list of code suggestions with evidence and confidence. Backends own
    chunking, prompting, retry, and parsing internally — callers only see
    the structured result.
    """

    name: str

    def code(self, chart: ExtractedChart) -> CodingResult: ...
