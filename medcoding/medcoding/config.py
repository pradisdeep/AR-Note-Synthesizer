"""Pipeline configuration. Reads from environment with sane local-dev defaults.

Production deployments override via env vars (LM_STUDIO_BASE_URL, etc.) so
the same code runs locally and on infrastructure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Extraction
    extractor: str = os.environ.get("MEDCODING_EXTRACTOR", "tesseract")
    ocr_dpi: int = int(os.environ.get("MEDCODING_OCR_DPI", "300"))
    ocr_lang: str = os.environ.get("MEDCODING_OCR_LANG", "eng")

    # Coding stage
    coder: str = os.environ.get("MEDCODING_CODER", "lm_studio")
    lm_studio_base_url: str = os.environ.get(
        "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
    )
    lm_studio_model: str = os.environ.get("LM_STUDIO_MODEL", "phi-4")
    lm_studio_api_key: str = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
    lm_studio_temperature: float = float(os.environ.get("LM_STUDIO_TEMPERATURE", "0.0"))
    lm_studio_max_tokens: int = int(os.environ.get("LM_STUDIO_MAX_TOKENS", "2048"))
    lm_studio_timeout_s: float = float(os.environ.get("LM_STUDIO_TIMEOUT_S", "120"))

    # Logging
    log_level: str = os.environ.get("MEDCODING_LOG_LEVEL", "INFO")


def load() -> Config:
    return Config()
