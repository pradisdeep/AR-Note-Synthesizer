# MedCoding Pipeline

End-to-end pipeline that turns scanned medical charts (TIFF) into ICD-10 / CPT
codes via a local LLM (LM Studio + phi-4). Designed for CPU-only personal-machine
development with a clean abstraction so the same code runs against GPU
infrastructure in production.

## Pipeline stages

```
TIFF (multi-page scan)
   ↓ [Stage 2] Extractor    (Tesseract OCR today; PaddleOCR / VLM via the same protocol)
Page[]                       — text + bounding boxes per page
   ↓ [Stage 3] Normalizer    (heuristic section detection + code-row parsing)
ExtractedChart               — section-tagged Markdown + structured ICD/CPT rows
   ↓ [Stage 4] Chunker       (Phase B — coming next)
   ↓ [Stage 5] Coder         (Phase B — phi-4 via LM Studio)
   ↓ [Stage 6] Validator     (Phase C — code lookup + audit)
ICD-10 + CPT with evidence
```

This repo currently ships **Stages 2–3** plus an evaluation harness that scores
extraction recall/precision against ground-truth codes from the synthetic chart
generator.

## Layout

```
medcoding/
├── medcoding/
│   ├── __init__.py
│   ├── config.py              # env-driven settings (LM Studio URL, etc.)
│   ├── models.py              # Page, Section, ExtractedChart, CodeRow
│   ├── normalizer.py          # OCR text -> section-tagged Markdown
│   └── extractors/
│       ├── base.py            # Extractor protocol
│       └── tesseract.py       # baseline OCR backend
├── scripts/
│   ├── extract.py             # single TIFF -> Markdown (+ optional JSON)
│   └── evaluate_extract.py    # batch eval against generator manifest.jsonl
├── tests/
│   └── test_normalizer.py
├── requirements.txt
└── README.md
```

## System dependencies

- Python 3.10+
- **Tesseract OCR** binary (the baseline extractor wraps it):
  - Debian/Ubuntu: `apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: download installer from UB Mannheim build

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Extract a single TIFF to Markdown

```bash
python scripts/extract.py path/to/chart.tiff --output chart.md --json chart.json
```

The Markdown is laid out in canonical clinical order (Chief Complaint, HPI,
ROS, PE, Diagnoses, Procedures, Plan, …) regardless of OCR order, with ICD-10
and CPT codes rendered as Markdown tables. This is the input format for phi-4
in Phase B.

### Evaluate against synthetic-chart-generator output

If you've run the [synthetic chart generator](../synthetic-chart-generator/),
score the extractor against its ground truth:

```bash
python scripts/evaluate_extract.py \
    ../synthetic-chart-generator/output/small_charts/manifest.jsonl
```

Reports per-chart predictions and aggregate ICD/CPT precision, recall, and F1.

### Run unit tests

```bash
pytest tests/ -v
```

## Configuration

Environment variables (all optional):

| Var | Default | Purpose |
| --- | --- | --- |
| `MEDCODING_EXTRACTOR` | `tesseract` | Backend name; resolved by `extractors/__init__.py`. |
| `MEDCODING_OCR_DPI` | `300` | Rasterization DPI hint (currently informational). |
| `MEDCODING_OCR_LANG` | `eng` | Tesseract language. |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio OpenAI-compatible endpoint (Phase B). |
| `LM_STUDIO_MODEL` | `phi-4` | Model name registered in LM Studio. |
| `MEDCODING_LOG_LEVEL` | `INFO` | stdlib logging level. |

## Production-portability notes

- **Extractor protocol** — `medcoding/extractors/base.py` defines a one-method
  interface (`extract(tiff_path) -> list[Page]`). Adding PaddleOCR, DocTR, or
  a VLM is a new file in `extractors/` and a new branch in
  `extractors/__init__.py:load_extractor`. No callers change.
- **Stateless stages** — every stage takes immutable input and returns
  immutable output (`@dataclass(frozen=True)` where it matters). Trivial to
  parallelize across processes, queues, or Lambdas.
- **Config via env** — no hardcoded paths or hostnames. Same code runs
  locally, in Docker, or on Kubernetes by changing env vars.

## What's next

- **Phase B:** `medcoding/coder.py` — chunk Markdown by section, send to
  phi-4 via LM Studio, return structured `{code, description, evidence_quote, confidence}`.
- **Phase C:** `medcoding/validator.py` — verify each predicted code exists,
  is billable, and pairs legally with linked codes; route low-confidence to a
  human-review queue.
- **Phase D:** orchestration (queue, retries, audit trail) once A–C are stable.
