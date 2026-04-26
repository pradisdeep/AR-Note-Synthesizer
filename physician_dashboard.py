"""
Physician AR Synthesizer page.

Page module for the multi-page app driven by app.py + st.navigation.
Calls into physician_poc/app.py's render() so the physician POC stays
self-contained.
"""
from __future__ import annotations

from physician_poc.app import (
    DEFAULT_ACCOUNTS_CSV,
    DEFAULT_CLUSTERS_CSV,
    render,
)

render(DEFAULT_ACCOUNTS_CSV, DEFAULT_CLUSTERS_CSV)
