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
| 4. Streamlit dashboard | `app.py` | http://localhost:8501 |

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
# Dashboard against the committed data/categorized_output.csv
streamlit run app.py

# Re-run the pipeline end-to-end (requires requirements-dev.txt)
python src/generate_pms_data.py
python src/claude_processor.py
```
