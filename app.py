"""
Multi-page Streamlit entry.

Two pages:
  * RCM Touch-Count Dashboard (rcm_home.py) - the original POC's
    note-only synthesis on synthetic data.
  * Physician AR Synthesizer (physician_dashboard.py) - event-sourced
    AR analysis on real production data with claude-haiku-4-5.

Run: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="RCM Note Synthesizer",
    page_icon=None,
    layout="wide",
)

home = st.Page(
    "rcm_home.py",
    title="RCM Touch-Count Dashboard",
    default=True,
)
physician = st.Page(
    "physician_dashboard.py",
    title="Physician AR Synthesizer",
)

st.navigation([home, physician]).run()
