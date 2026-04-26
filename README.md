# RCM Note Synthesizer & Touch-Count Dashboard

A POC that bridges unstructured biller notes and objective operational BI to surface the
"Burnout Zone" — claims that cost more in labor to work than their collectible balance.

## Stack
- Python 3.10+
- Pandas, Faker
- Anthropic SDK (`claude-haiku-4-5` — original POC spec named `claude-3-haiku-20240307`, which was retired Apr 19, 2026)
- Streamlit + Plotly

## Pipeline

A claim is rarely stuck for one reason. Many accounts go through a *cascade* of
denials over time — COB resolves and Auth surfaces, Auth peer-to-peer needs
clinical records, the resulting delays push the claim past the filing window.
The pipeline captures that journey, not just the latest blocker.

| Phase | Script | Output |
| --- | --- | --- |
| 1. Synthetic PMS extract (multi-note, with realistic denial cascades) | `src/generate_pms_data.py` | `data/synthetic_pms_extract.csv` (~270 rows for 150 accounts) |
| 2. LLM per-account synthesis (terminal cause + denial journey) | `src/claude_processor.py` | `data/categorized_output.csv` (one row per account) |
| 3. Touch / balance / waste transform | `src/data_transformer.py` | in-memory dataframe |
| 4. Streamlit dashboard | `rcm_home.py` (page) / `app.py` (multi-page entry) | http://localhost:8501 |

For each account the LLM returns:
- `LLM_Terminal_Root_Cause` - the current blocker (most recent note's category)
- `LLM_Denial_Journey` - distinct causes the claim cycled through, in order
- `LLM_Journey_Length` - count of distinct causes (higher = deeper rework)

## Setup

The dashboard (`app.py`) only needs pandas, streamlit, and plotly. The Phase 1-2
pipeline scripts also need faker, anthropic, and python-dotenv.

```bash
# Dashboard only (also what Streamlit Cloud installs)
pip install -r requirements.txt

# Full pipeline (regenerate synthetic data and re-run the LLM step)
pip install -r requirements-dev.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Run

```bash
# Multi-page dashboard (RCM Touch-Count + Physician AR Synthesizer)
streamlit run app.py

# Or just one page standalone
streamlit run rcm_home.py
streamlit run physician_poc/app.py

# Re-run the pipeline end-to-end (requires requirements-dev.txt)
python src/generate_pms_data.py
python src/claude_processor.py
```

## Layout

```
app.py                    # multi-page entry (st.navigation)
rcm_home.py               # page 1: synthetic-data POC
physician_dashboard.py    # page 2: shim that calls physician_poc.app.render()

src/                      # original POC pipeline (synthetic data)
data/                     # synthetic_pms_extract.csv, categorized_output.csv

physician_poc/            # second POC: real production-data event-sourced AR
  src/
    redact.py             # PHI scrubber (patient names, member IDs)
    ingest.py             # events + notes timeline builder
    claude_processor.py   # LLM journey synthesis (claude-haiku-4-5)
    transform.py          # per-account features + cluster aggregation
  data/
    events_redacted.csv   # AR ledger
    notes_redacted.csv    # biller notes
    synthesized.csv       # LLM output (terminal cause, journey, anomaly flags)
    accounts.csv          # per-account features
    clusters.csv          # cluster aggregation
  app.py                  # dashboard (run standalone or via root app.py)
```
