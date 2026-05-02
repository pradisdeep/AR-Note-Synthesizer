# RCM Note Synthesizer & Touch-Count Dashboard

A POC that bridges unstructured biller notes and objective operational BI to surface the
"Burnout Zone" — claims that cost more in labor to work than their collectible balance.

## Stack
- Python 3.10+
- Pandas, Faker
- Anthropic SDK (`claude-3-haiku-20240307`)
- Streamlit + Plotly

## Pipeline

| Phase | Script | Output |
| --- | --- | --- |
| 1. Synthetic PMS extract | `src/generate_pms_data.py` | `data/synthetic_pms_extract.csv` |
| 2. LLM root-cause synthesis | `src/claude_processor.py` | `data/categorized_output.csv` |
| 3. Touch / balance / waste transform | `src/data_transformer.py` | in-memory dataframe |
| 4. Streamlit dashboard | `app.py` | http://localhost:8501 |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Run

```bash
python src/generate_pms_data.py
python src/claude_processor.py
streamlit run app.py
```
