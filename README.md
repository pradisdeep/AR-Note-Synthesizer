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

---

## Synthetic Clinical Chart Generator

Companion pipeline that produces synthetic, HIPAA-safe medical charts as
degraded multi-page TIFFs. The output serves as a ground-truth testing
set for local OCR, RAG, and offline-LLM stacks: nothing in the generated
data references real patients.

### Components
- `data/icd10_codes.csv`, `data/cpt_codes.csv`, `data/clinical_templates.json` — static clinical dictionaries.
- `src/chart_generator/data_generator.py` — Faker + dictionary-driven payload synthesis.
- `src/chart_generator/document_builder.py` — ReportLab PDF layout (header, vitals, ICD/CPT tables, plan).
- `src/chart_generator/image_degrader.py` — OpenCV-based blur, noise, rotation, speckle, and JPEG-roundtrip degradation.
- `src/chart_generator/tiff_exporter.py` — pdf2image rasterization and multi-page TIFF packaging.
- `scripts/generate_small.py`, `generate_medium.py`, `generate_complex.py` — entry points for the three complexity tiers.

### System dependencies
`pdf2image` requires Poppler. On Debian/Ubuntu: `apt-get install poppler-utils`.

### Usage

```bash
pip install -r requirements.txt
python scripts/generate_small.py   --count 10 --seed 42
python scripts/generate_medium.py  --count 10 --seed 42
python scripts/generate_complex.py --count 10 --seed 42
```

Each run writes TIFFs and a `manifest.jsonl` (with seeds and ground-truth
ICD/CPT codes) to `output/<level>_charts/`. The `output/` directory and
all `*.tiff` files are gitignored.

### Complexity tiers

| Tier | Diagnoses | Procedures | Meds | Degradation |
| --- | --- | --- | --- | --- |
| Small | 1 | 1 | 2 | Light scan-style: mild blur, minor noise, ~0.4° skew |
| Medium | 3 | 3 | 5 | Moderate fax-style: streaks, JPEG quality 65, ~1.2° skew |
| Complex | 6 | 5 | 8 | Aggressive: heavy noise, JPEG quality 40, ~2.5° skew, addenda |
