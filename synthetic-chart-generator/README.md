# Synthetic Clinical Chart Generator

Pipeline that produces synthetic, HIPAA-safe medical charts as degraded
multi-page TIFFs across three complexity tiers. Output is ground-truth
test data for local OCR, RAG, and offline-LLM stacks; nothing in the
generated data references real patients.

## Stack
- Python 3.10+
- Faker (PII synthesis)
- ReportLab (PDF layout)
- pdf2image + Poppler (PDF → image)
- Pillow + OpenCV + NumPy (degradation, TIFF packaging)

## Layout
```
synthetic-chart-generator/
├── data/
│   ├── icd10_codes.csv
│   ├── cpt_codes.csv
│   └── clinical_templates.json
├── src/chart_generator/
│   ├── data_generator.py     # Faker + dictionaries → ChartData
│   ├── document_builder.py   # ChartData → ReportLab PDF
│   ├── image_degrader.py     # OpenCV blur/noise/rotation/JPEG round-trip
│   └── tiff_exporter.py      # PDF → multi-page degraded TIFF
├── scripts/
│   ├── generate_small.py
│   ├── generate_medium.py
│   └── generate_complex.py
├── requirements.txt
└── .gitignore
```

## System dependencies
`pdf2image` requires Poppler.

- Debian/Ubuntu: `apt-get install poppler-utils`
- macOS (Homebrew): `brew install poppler`
- Windows: install Poppler binaries and add the `bin/` directory to `PATH`.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run from the `synthetic-chart-generator/` directory:

```bash
python scripts/generate_small.py   --count 10 --seed 42
python scripts/generate_medium.py  --count 10 --seed 42
python scripts/generate_complex.py --count 10 --seed 42
```

CLI flags (all three scripts):

| Flag | Default | Description |
| --- | --- | --- |
| `--count` | 5 | Number of charts to generate. |
| `--seed` | None | Base RNG seed; per-chart seeds are `seed + index`. |
| `--output-dir` | `output/<level>_charts` | Where TIFFs and the manifest are written. |
| `--dpi` | 200 | Rasterization DPI for pdf2image. |
| `--keep-pdf` | off | Also keep the source PDF next to each TIFF. |

Each run writes:
- `<level>_chart_NNNN.tiff` — multi-page degraded TIFF.
- `manifest.jsonl` — one record per chart with seed, MRN, encounter ID, and ground-truth ICD/CPT codes (use this as the OCR/RAG evaluation key).

## Complexity tiers

| Tier | Diagnoses | Procedures | Meds | Degradation |
| --- | --- | --- | --- | --- |
| Small | 1 | 1 | 2 | Light scan-style: mild blur, minor noise, ~0.4° skew, JPEG q85 |
| Medium | 3 | 3 | 5 | Moderate fax-style: streaks, JPEG q65, ~1.2° skew, addenda |
| Complex | 6 | 5 | 8 | Aggressive: heavy noise, JPEG q40, ~2.5° skew, addenda |

Profile parameters live in `src/chart_generator/image_degrader.py` —
construct a custom `DegradationProfile` if you need finer control.

## Reproducibility

Pass `--seed N` to make a batch fully reproducible. The same `(seed, count)`
pair produces the same charts and the same per-page degradations.

## HIPAA / safety note

Patient names, DOBs, MRNs, addresses, phone numbers, NPIs, account numbers,
and member IDs are all synthesized by Faker or random number generation.
The clinical content is assembled from static templates and code lookups.
No real PHI is read, written, or referenced.
