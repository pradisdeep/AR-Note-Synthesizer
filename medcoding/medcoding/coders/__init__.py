"""Pluggable coding backends. Pick via Config.coder."""

from .base import Coder

__all__ = ["Coder", "load_coder"]


def load_coder(name: str, **kwargs) -> Coder:
    """Resolve a backend by name.

    Adding a new backend (vLLM, hosted Anthropic, etc.) means dropping a new
    module here and a new branch — no caller changes.
    """

    name = name.lower()
    if name == "lm_studio":
        from .lm_studio import LMStudioCoder

        return LMStudioCoder(**kwargs)
    if name == "mock":
        from .mock import MockCoder

        return MockCoder(**kwargs)
    raise ValueError(
        f"Unknown coder {name!r}. Available: lm_studio, mock."
    )
