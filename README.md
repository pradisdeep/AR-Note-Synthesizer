# RCM Note Synthesizer & Touch-Count Dashboard

A POC that bridges unstructured biller notes and objective operational BI to surface the
"Burnout Zone" — claims that cost more in labor to work than their collectible balance.

## Stack
- Python 3.10+
- Pandas, Faker
- Anthropic SDK (`claude-haiku-4-5` — original POC spec named `claude-3-haiku-20240307`, which was retired Apr 19, 2026)
- Streamlit + Plotly

## Pipeline

| Phase | Script | Output |
| --- | --- | --- |
| 1. Synthetic PMS extract | `src/generate_pms_data.py` | `data/synthetic_pms_extract.csv` |
| 2. LLM root-cause synthesis | `src/claude_processor.py` | `data/categorized_output.csv` |
| 3. Touch / balance / waste transform | `src/data_transformer.py` | in-memory dataframe |
| 4. Streamlit dashboard | `app.py` | http://localhost:8501 |

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
