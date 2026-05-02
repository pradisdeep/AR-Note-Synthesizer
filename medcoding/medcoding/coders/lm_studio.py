"""LM Studio (OpenAI-compatible) coder backend.

Talks to a locally-running LM Studio server hosting phi-4 or any other
chat-completion model exposed at the OpenAI Chat Completions endpoint.

Production swap: pointing `LM_STUDIO_BASE_URL` at vLLM, TGI, or any other
OpenAI-compatible server is the only change needed. No code edits.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from ..chunker import chunk_for_coding
from ..models import CodeSuggestion, CodingResult, ExtractedChart
from ..prompts import render_cpt_messages, render_icd_messages

log = logging.getLogger(__name__)

_CONNECTION_HINT = (
    "Could not reach LM Studio. Make sure LM Studio is running, the OpenAI "
    "endpoint is enabled (Server tab), the model is loaded, and "
    "LM_STUDIO_BASE_URL points at it (default: http://localhost:1234/v1)."
)


class LMStudioCoder:
    name = "lm_studio"

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "phi-4",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        request_timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for LMStudioCoder. "
                "Install it with: pip install openai"
            ) from e

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout,
            max_retries=0,  # we handle retries ourselves so we can log them
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._base_url = base_url

    def code(self, chart: ExtractedChart) -> CodingResult:
        chunks = chunk_for_coding(chart)
        icd_suggestions: list[CodeSuggestion] = []
        cpt_suggestions: list[CodeSuggestion] = []
        raw: dict[str, str] = {}

        for i, chunk in enumerate(chunks):
            messages = (
                render_icd_messages(chunk.text)
                if chunk.purpose == "icd"
                else render_cpt_messages(chunk.text)
            )
            response_text = self._chat(messages)
            key = f"{chunk.purpose}_chunk_{i}"
            raw[key] = response_text

            parsed = _parse_json_response(response_text)
            if parsed is None:
                log.warning("Failed to parse JSON from %s", key)
                continue

            if chunk.purpose == "icd":
                icd_suggestions.extend(
                    _coerce_suggestions(parsed.get("diagnoses", []), "ICD-10", chunk.sections_included)
                )
            else:
                cpt_suggestions.extend(
                    _coerce_suggestions(parsed.get("procedures", []), "CPT", chunk.sections_included)
                )

        return CodingResult(
            chart_source=chart.source_path,
            coder=self.name,
            coded_at=datetime.now(timezone.utc),
            icd_suggestions=icd_suggestions,
            cpt_suggestions=cpt_suggestions,
            raw_responses=raw,
            coder_metadata={
                "model": self._model,
                "base_url": self._base_url,
                "temperature": self._temperature,
                "chunks": len(chunks),
            },
        )

    def _chat(self, messages: list[dict]) -> str:
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
                return completion.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - we re-raise after retries
                last_err = e
                msg = str(e).lower()
                if any(s in msg for s in ("connection", "refused", "ecconn", "could not connect")):
                    log.error(_CONNECTION_HINT)
                if attempt < self._max_retries:
                    backoff = 2**attempt
                    log.warning("LM Studio call failed (attempt %d): %s; retrying in %ds", attempt + 1, e, backoff)
                    time.sleep(backoff)
                    continue
                break
        # If we exhausted retries, try once more without response_format —
        # some models / LM Studio versions don't support strict JSON mode.
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            log.warning("Recovered without response_format=json_object — model may not support JSON mode")
            return completion.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(_CONNECTION_HINT) from (last_err or e)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Be forgiving: try strict JSON first, then a regex-extracted object."""

    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _coerce_suggestions(
    rows: list[dict],
    code_type: str,
    sections_included: list[str],
) -> list[CodeSuggestion]:
    """Defensive coercion: drop malformed rows, clamp confidence to [0,1]."""

    out: list[CodeSuggestion] = []
    source = ",".join(sections_included)
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        code = (row.get("code") or "").strip().upper()
        desc = (row.get("description") or "").strip()
        evidence = (row.get("evidence") or "").strip()
        try:
            conf = float(row.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        if not code:
            continue
        out.append(
            CodeSuggestion(
                code_type=code_type,  # type: ignore[arg-type]
                code=code,
                description=desc,
                evidence_quote=evidence,
                confidence=max(0.0, min(1.0, conf)),
                source_section=source,
            )
        )
    return out
