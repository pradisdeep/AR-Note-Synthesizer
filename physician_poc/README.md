# Physician POC

Physician-billing AR analysis on real production data (de-identified).
Separate from the root-level POC (`/app.py`), which uses synthetic notes.

## Status

| Layer | Component | Status |
| --- | --- | --- |
| 0 | PHI redactor | done (`src/redact.py`) |
| 1 | Data ingest + join (events x notes) | in progress (`src/ingest.py`) |
| 2 | LLM per-account timeline synthesis | not started |
| 3 | Cluster aggregation + automation flags | not started |
| 4 | Streamlit page | not started |

## Data sources

Two CSVs, joined on visit/account number.

### `events.csv` (event-sourced AR ledger)

| Column | Notes |
| --- | --- |
| `visit` | Account number; join key. Must NOT be Excel-mangled to scientific notation. |
| `dos_from`, `dos_to` | Date-of-service range. Multi-day spans indicate hospitalist or radiation-oncology series. |
| `placementpayor`, `placementpayorfsc` | Payor classification (COMMERCIAL / MEDICARE / MEDICAID / WORKERS COMP / GOVERNMENT). |
| `placementdate`, `first_claim_submission_date` | Date account was placed for follow-up. (NOTE: these are usually equal in the production extract; not a true claim-submission date.) |
| `cpt_code` | Populated on `Denial` rows for line-item denials; otherwise blank. |
| `transaction_type` | `Charges` / `Payment` / `Denial` / `Adjustment`. |
| `transaction_date` | Event date. |
| `transaction_amount` | Dollar amount of the event. |
| `denial_category1` | High-level denial bucket (Registration/Eligibility, Coding, Bundled, Timely Filing, Additional Documentation Needed, Medical Necessity/Level of Care, Collector Review, Billing). |
| `latest_status_code` | Terminal status of the account (Closed = resolved; Hcbl/Cs/Sp/Cc/Cp/Insfu/Den = open in various worklists). |

### `notes_redacted.csv` (post-redaction biller notes)

| Column | Notes |
| --- | --- |
| `accountnumber` | Join key (= `visit` in `events.csv`). |
| `touchstartdate` | Note timestamp. |
| `notescurrentvalue` | Free text, post-redaction. PHI patterns replaced with `[REDACTED-NAME]`, `[REDACTED-MEMBER-ID]`, etc. |
| `eventcreatedby` | Biller user ID, or system process name (e.g. `CLAIMAUTOUPLOAD`, `SHSIndexXMLCreation_Interface_Hosp_PROD`, `PM_ResponseInterface`). |

### Files in `data/`

- `notes_raw.csv` (gitignored) - raw production notes; never committed.
- `events_raw.csv` (gitignored) - raw production events; never committed.
- `*_redacted.csv` (committable) - allowlisted in `.gitignore`.

## Pipeline (target)

```
events_raw.csv ─┐
                ├─> ingest.load_and_join() ─> per-account event timelines
notes_raw.csv ──┴─> redact.run_cli() ─> notes_redacted.csv ─┘
                                                              │
                       ┌──────────────────────────────────────┘
                       v
       claude_processor.synthesize_per_account()
                       │
                       │  per-account: terminal cause, denial journey,
                       │              anomaly flags, narrative
                       v
              transform.compute_clusters()
                       │
                       │  cluster: payor x specialty x denial_category x DOS qtr
                       │  metrics: recovery rate, avg cycle time, common paths
                       v
              app.py (Streamlit page)
```

## Run

```bash
# 0. Redact raw notes (one-time per data refresh)
python physician_poc/src/redact.py \
    physician_poc/data/notes_raw.csv \
    physician_poc/data/notes_redacted.csv

# 1. Ingest + join (in progress)
python physician_poc/src/ingest.py \
    --events physician_poc/data/events_raw.csv \
    --notes physician_poc/data/notes_redacted.csv

# 2-4: not built yet
```
