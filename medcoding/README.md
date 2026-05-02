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
   ↓             ↳ NoiseFilter drops orders/Rx/referrals/forms before they reach the LLM
ExtractedChart               — section-tagged Markdown + structured ICD/CPT rows
   ↓ [Stage 4] Chunker       (section-aware: ICD prompt sees diagnosis context, CPT prompt sees procedure context)
   ↓ [Stage 5] Coder         (phi-4 via LM Studio, OpenAI-compatible)
   ↓ [Stage 6] Validator     (Phase C — code lookup + audit)
ICD-10 + CPT with evidence
```

This repo currently ships **Stages 2–3 (extraction + normalization)** and
**Stages 4–5 (chunking + LLM coding)** plus evaluation harnesses that score
both stages independently against ground-truth codes from the synthetic chart
generator. Validation, audit log, and orchestration (Phases C/D) are next.

## Layout

```
medcoding/
├── medcoding/
│   ├── __init__.py
│   ├── config.py              # env-driven settings (LM Studio URL, etc.)
│   ├── models.py              # Page, Section, ExtractedChart, CodeRow,
│   │                          # CodeSuggestion, CodingResult
│   ├── normalizer.py          # OCR text -> section-tagged Markdown
│   ├── noise_filter.py        # drops orders / Rx / referrals / forms
│   ├── chunker.py             # section-based chunking (ICD vs CPT relevant)
│   ├── prompts.py             # ICD/CPT prompt templates
│   ├── extractors/
│   │   ├── base.py            # Extractor protocol
│   │   └── tesseract.py       # baseline OCR backend
│   └── coders/
│       ├── base.py            # Coder protocol
│       ├── lm_studio.py       # OpenAI-compatible client (LM Studio / vLLM)
│       └── mock.py            # echo + scripted modes for tests/CI
├── scripts/
│   ├── extract.py             # TIFF -> Markdown (+ optional JSON)
│   ├── evaluate_extract.py    # extraction-only eval against manifest.jsonl
│   ├── code_chart.py          # TIFF -> ICD-10 + CPT codes (full pipeline)
│   └── evaluate_coding.py     # end-to-end eval against manifest.jsonl
├── tests/
│   ├── test_normalizer.py
│   ├── test_chunker.py
│   ├── test_coders.py
│   └── test_noise_filter.py
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

### Code a chart end-to-end

Single chart through extraction + normalization + LLM coding:

```bash
# With LM Studio running (phi-4 loaded, server on http://localhost:1234)
python scripts/code_chart.py path/to/chart.tiff

# Without LM Studio — mock coder echoes the normalizer's table-parsed codes
python scripts/code_chart.py path/to/chart.tiff --coder mock
```

Output is a CodingResult with per-code `(code, description, evidence_quote, confidence)`.

### Evaluate end-to-end coding

```bash
# Mock coder (no LM Studio needed) — measures the deterministic baseline
python scripts/evaluate_coding.py \
    ../synthetic-chart-generator/output/small_charts/manifest.jsonl \
    --coder mock --min-confidence 0

# LM Studio coder — measures phi-4 lift over the deterministic baseline
python scripts/evaluate_coding.py \
    ../synthetic-chart-generator/output/small_charts/manifest.jsonl \
    --coder lm_studio --min-confidence 0.5
```

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
| `MEDCODING_CODER` | `lm_studio` | Coder backend; `lm_studio` or `mock`. |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible endpoint (LM Studio, vLLM, etc.). |
| `LM_STUDIO_MODEL` | `phi-4` | Model name registered with the server. |
| `LM_STUDIO_API_KEY` | `lm-studio` | API key (LM Studio ignores it; required by the SDK). |
| `LM_STUDIO_TEMPERATURE` | `0.0` | Coding wants deterministic output. |
| `LM_STUDIO_MAX_TOKENS` | `2048` | Per-response cap. |
| `LM_STUDIO_TIMEOUT_S` | `120` | Request timeout. CPU phi-4 is slow; budget accordingly. |
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

## Running with LM Studio (phi-4)

1. Open LM Studio, **Server** tab, load `phi-4-Q4_K_M-GGUF`, click *Start*.
2. Confirm the endpoint is reachable: `curl http://localhost:1234/v1/models`.
3. Run the pipeline:
   ```bash
   export LM_STUDIO_MODEL="phi-4"   # or whatever the model id is in LM Studio
   python scripts/code_chart.py path/to/chart.tiff
   ```
4. CPU inference of phi-4 14B is slow (~5–10 min per chart, two LLM calls per
   chart — one for ICD, one for CPT). For evaluation runs use `--limit 3` until
   you have a GPU.

## Noise filtering

Real-world charts contain a lot of content that is not coding evidence:
order tables, prescriptions, referrals, fax cover sheets, registration
forms, prior-auth requests, etc. ICD coding rules forbid coding suspected
or planned conditions; CPT requires services rendered, not ordered.
Letting these sections reach the LLM causes both compliance failures and
accuracy drops.

`medcoding/noise_filter.py` runs three layered checks per section:

1. **Header pattern** (fast, deterministic, configurable per-EHR).
   Recognizes `ORDERS`, `PROVIDER ORDERS`, `RX`, `PRESCRIPTIONS`,
   `REFERRALS`, `LAB ORDER`, `IMAGING ORDER`, `FAX COVER`,
   `PATIENT EDUCATION`, `PRIOR AUTHORIZATION`, etc.
2. **Table-structure heuristic.** Looks for column-header tokens
   (`Status`, `Pending`, `Sent`, `Refills`, `Pharmacy`) and order-status
   density even when the section's name didn't match a known pattern.
3. **Content-pattern density.** Last-line-of-defense regex scan for
   `Send to:`, `Refer to:`, `Order:`, `Authorized by:`, etc. If more than
   30% of lines match, the section is dropped.

A separate **whitelist** in `chunker.py` enforces that only known
clinical sections (HPI, ROS, PE, A&P, Plan, Addendum, etc.) make it
into the LLM prompt — even if all three filter checks fail.

Sections marked `uncertain` (no check fired but the name is non-canonical)
are logged so the patterns can be extended for new EHRs without silently
dropping clinical content.

The synthetic chart generator now injects provider-variant noise sections
into medium tier (1 per chart) and complex tier (3 per chart). The
manifest records what was injected, so you can verify the filter is
working with `python scripts/evaluate_coding.py manifest.jsonl --show-noise-dropped`.

## What's next

- **Phase C:** `medcoding/validator.py` — verify each predicted code exists,
  is billable, and pairs legally with linked codes; route low-confidence to a
  human-review queue. Reference data: `data/icd10_codes.csv` and
  `data/cpt_codes.csv` from synthetic-chart-generator are a starting point.
- **Phase D:** orchestration (queue, retries, audit trail, persistent storage)
  once A–C are stable.
- **Stronger extractor** for the heavily-degraded tier: drop in
  `medcoding/extractors/paddle.py` or a VLM backend behind the same protocol —
  no caller changes.
