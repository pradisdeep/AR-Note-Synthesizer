"""
Phase 3 - per-account features + cluster aggregation (physician POC).

Two outputs:

  1. Per-account feature dataframe (one row per visit) combining:
       * Financials from events (charges, payments, adjustments, balance)
       * Touch counts from notes (biller vs system)
       * LLM-derived synthesis (terminal cause, journey, anomaly flags)
       * Cycle-time and labor-cost derived columns

  2. Cluster aggregation grouped by payor x terminal_root_cause x DOS quarter,
     surfacing recovery-rate, average cycle time, common journeys, and
     anomaly prevalence so the dashboard can rank actionable buckets.

The events file is the canonical account universe - notes-only orphans are
ignored here (they represent test artefacts or accounts deleted from the
PMS extract). The LLM synthesis is left-joined; visits with no synthesis
get terminal_root_cause = UNSYNTHESIZED so they remain visible but flagged.

Usage:
    python physician_poc/src/transform.py \
        --events physician_poc/data/events_redacted.csv \
        --notes  physician_poc/data/notes_redacted.csv \
        --synth  physician_poc/data/synthesized.csv \
        --out-accounts physician_poc/data/accounts.csv \
        --out-clusters physician_poc/data/clusters.csv

    python physician_poc/src/transform.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from ingest import SYSTEM_PROCESSES, _parse_dates_robust, load_events, load_notes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Placeholder labor cost. The real number depends on the practice's
# fully-loaded biller hourly rate and average minutes-per-touch; the
# Streamlit page can expose this as a slider. $5/touch matches the
# original POC's assumption (~3 minutes at ~$100/hr loaded).
DEFAULT_LABOR_COST_PER_TOUCH = 5.0

UNSYNTHESIZED = "UNSYNTHESIZED"


# --- Per-account features -------------------------------------------------


def _financial_aggs(events: pd.DataFrame) -> pd.DataFrame:
    """Sum transaction_amount by visit x transaction_type into wide columns."""
    pivot = (
        events.pivot_table(
            index="visit",
            columns="transaction_type",
            values="transaction_amount",
            aggfunc="sum",
            fill_value=0.0,
        )
        .rename(
            columns={
                "Charges": "charges_total",
                "Payment": "payments_total",
                "Adjustment": "adjustments_total",
                "Denial": "denials_total",
            }
        )
    )
    # Make sure all four columns exist even if a transaction_type is absent.
    for col in ["charges_total", "payments_total", "adjustments_total", "denials_total"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    return pivot[["charges_total", "payments_total", "adjustments_total", "denials_total"]]


def _per_account_meta(events: pd.DataFrame) -> pd.DataFrame:
    """Pull payor, DOS quarter, latest status from the events feed."""
    # Each visit has consistent payor / DOS / status; take the first row.
    meta = events.groupby("visit").agg(
        payor=("placementpayor", "first"),
        payor_fsc=("placementpayorfsc", "first"),
        dos_from=("dos_from", "first"),
        dos_to=("dos_to", "first"),
        placement_date=("placementdate", "first"),
        latest_status_code=("latest_status_code", "first"),
        first_event_date=("transaction_date", "min"),
        last_event_date=("transaction_date", "max"),
    )
    # Production extract uses mixed date formats; see ingest._parse_dates_robust.
    meta["dos_from"] = _parse_dates_robust(meta["dos_from"])
    meta["dos_to"] = _parse_dates_robust(meta["dos_to"])
    meta["placement_date"] = _parse_dates_robust(meta["placement_date"])
    meta["dos_quarter"] = meta["dos_from"].dt.to_period("Q").astype(str)
    return meta


def _touch_counts(notes: pd.DataFrame) -> pd.DataFrame:
    """Count biller vs system touches per account, plus first/last biller dates."""
    if notes.empty:
        return pd.DataFrame(
            columns=[
                "num_biller_touches",
                "num_system_touches",
                "first_biller_touch_date",
                "last_biller_touch_date",
            ]
        )
    is_system = notes["eventcreatedby"].astype(str).isin(SYSTEM_PROCESSES)
    biller = notes.loc[~is_system].groupby("accountnumber")
    system = notes.loc[is_system].groupby("accountnumber")

    agg = pd.DataFrame(
        {
            "num_biller_touches": biller.size(),
            "first_biller_touch_date": biller["touchstartdate"].min(),
            "last_biller_touch_date": biller["touchstartdate"].max(),
            "num_system_touches": system.size(),
        }
    )
    agg.index.name = "visit"
    agg["num_biller_touches"] = agg["num_biller_touches"].fillna(0).astype(int)
    agg["num_system_touches"] = agg["num_system_touches"].fillna(0).astype(int)
    return agg


def _join_synthesis(accounts: pd.DataFrame, synth: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join synthesis output. Missing rows -> UNSYNTHESIZED placeholders."""
    if synth is None or synth.empty:
        accounts["terminal_root_cause"] = UNSYNTHESIZED
        accounts["denial_journey"] = ""
        accounts["journey_length"] = 0
        accounts["anomaly_flags"] = ""
        accounts["narrative"] = ""
        return accounts

    synth = synth.copy()
    synth["visit"] = synth["visit"].astype(str)
    keep = [c for c in ["visit", "terminal_root_cause", "denial_journey",
                        "journey_length", "anomaly_flags", "narrative"]
            if c in synth.columns]
    out = accounts.merge(synth[keep], on="visit", how="left")
    out["terminal_root_cause"] = out["terminal_root_cause"].fillna(UNSYNTHESIZED)
    out["denial_journey"] = out["denial_journey"].fillna("")
    out["journey_length"] = out["journey_length"].fillna(0).astype(int)
    out["anomaly_flags"] = out["anomaly_flags"].fillna("")
    if "narrative" in out.columns:
        out["narrative"] = out["narrative"].fillna("")
    return out


def compute_account_features(
    events: pd.DataFrame,
    notes: pd.DataFrame,
    synth: pd.DataFrame | None = None,
    labor_cost_per_touch: float = DEFAULT_LABOR_COST_PER_TOUCH,
) -> pd.DataFrame:
    """One row per visit with financials, touches, cycle, and synthesis joined."""
    financials = _financial_aggs(events)
    meta = _per_account_meta(events)

    accounts = meta.join(financials, how="left").reset_index()
    accounts["visit"] = accounts["visit"].astype(str)

    touches = _touch_counts(notes).reset_index()
    if not touches.empty:
        touches["visit"] = touches["visit"].astype(str)
        accounts = accounts.merge(touches, on="visit", how="left")
    else:
        accounts["num_biller_touches"] = 0
        accounts["num_system_touches"] = 0
        accounts["first_biller_touch_date"] = pd.NaT
        accounts["last_biller_touch_date"] = pd.NaT

    accounts["num_biller_touches"] = accounts["num_biller_touches"].fillna(0).astype(int)
    accounts["num_system_touches"] = accounts["num_system_touches"].fillna(0).astype(int)
    accounts["num_total_touches"] = accounts["num_biller_touches"] + accounts["num_system_touches"]

    # Current balance: what's still outstanding on the AR ledger.
    accounts["current_balance"] = (
        accounts["charges_total"]
        - accounts["payments_total"]
        - accounts["adjustments_total"]
    )

    # Recovery rate: fraction of charges actually collected as cash. NaN
    # when there were no charges (shouldn't happen in clean data, but be safe).
    with np.errstate(divide="ignore", invalid="ignore"):
        accounts["recovery_rate"] = np.where(
            accounts["charges_total"] > 0,
            accounts["payments_total"] / accounts["charges_total"],
            np.nan,
        )
    accounts["recovered"] = accounts["payments_total"] > 0

    # Cycle time: from placement to the latest signal (event or biller touch).
    last_activity = accounts[["last_event_date", "last_biller_touch_date"]].max(axis=1)
    accounts["last_activity_date"] = last_activity
    accounts["cycle_time_days"] = (
        last_activity - accounts["placement_date"]
    ).dt.days

    # Labor: estimated cost AND a "is the recoverable balance worth the labor?"
    # guard. balance_per_labor_dollar < 1.0 means we are losing money chasing it.
    accounts["labor_cost_estimate"] = (
        accounts["num_biller_touches"] * labor_cost_per_touch
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        accounts["balance_per_labor_dollar"] = np.where(
            accounts["labor_cost_estimate"] > 0,
            accounts["current_balance"] / accounts["labor_cost_estimate"],
            np.nan,
        )

    accounts = _join_synthesis(accounts, synth)
    return accounts


# --- Cluster aggregation --------------------------------------------------


def _top_value(series: pd.Series) -> str:
    """Most-common non-empty string in a series, or empty if all blank."""
    s = series.dropna().astype(str)
    s = s[s != ""]
    if s.empty:
        return ""
    return s.value_counts().idxmax()


def _pct_with_anomaly(series: pd.Series) -> float:
    """Fraction of accounts in the group with at least one anomaly flag."""
    s = series.fillna("").astype(str)
    return float((s != "").mean()) if len(s) else 0.0


def compute_clusters(
    accounts: pd.DataFrame,
    by: tuple[str, ...] = ("payor", "terminal_root_cause", "dos_quarter"),
) -> pd.DataFrame:
    """Roll per-account features up into actionable clusters.

    Default grouping keys: payor x terminal_root_cause x DOS quarter. Override
    `by` to slice differently (e.g. add latest_status_code).
    """
    if accounts.empty:
        return pd.DataFrame()

    grp = accounts.groupby(list(by), dropna=False)
    out = grp.agg(
        account_count=("visit", "size"),
        total_charges=("charges_total", "sum"),
        total_payments=("payments_total", "sum"),
        total_adjustments=("adjustments_total", "sum"),
        total_open_balance=("current_balance", "sum"),
        avg_cycle_days=("cycle_time_days", "mean"),
        median_cycle_days=("cycle_time_days", "median"),
        avg_journey_length=("journey_length", "mean"),
        avg_biller_touches=("num_biller_touches", "mean"),
        total_labor_cost=("labor_cost_estimate", "sum"),
        top_journey=("denial_journey", _top_value),
        top_anomaly_flag=("anomaly_flags", _top_value),
        pct_with_anomaly=("anomaly_flags", _pct_with_anomaly),
    ).reset_index()

    # Cluster recovery rate: collected / billed at the cluster level. This
    # is more meaningful than averaging per-account rates because per-account
    # rates double-count tiny accounts.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["recovery_rate"] = np.where(
            out["total_charges"] > 0,
            out["total_payments"] / out["total_charges"],
            np.nan,
        )

    # Sort by impact: largest open dollar buckets first.
    out = out.sort_values("total_open_balance", ascending=False).reset_index(drop=True)
    return out


# --- Self-test ------------------------------------------------------------


def _stub_synthesis(visits: list[str]) -> pd.DataFrame:
    """Hand-crafted synthesis records to exercise the join + cluster paths."""
    rows = []
    for visit in visits:
        if visit == "SYN0000001":
            rows.append({
                "visit": visit,
                "terminal_root_cause": "UNDERPAYMENT",
                "denial_journey": "UNDERPAYMENT",
                "journey_length": 1,
                "anomaly_flags": "INCORRECT_WRITEOFF",
                "narrative": "Paid $0, then full charge written off as adjustment.",
            })
        elif visit == "SYN0000002":
            rows.append({
                "visit": visit,
                "terminal_root_cause": "DUPLICATE_CLAIM",
                "denial_journey": "DUPLICATE_CLAIM",
                "journey_length": 1,
                "anomaly_flags": "",
                "narrative": "Medicare denied as duplicate; rebill required.",
            })
        elif visit == "SYN0000003":
            rows.append({
                "visit": visit,
                "terminal_root_cause": "REGISTRATION_ELIGIBILITY",
                "denial_journey": "REGISTRATION_ELIGIBILITY",
                "journey_length": 1,
                "anomaly_flags": "MASS_REGELIG_DENIAL",
                "narrative": "All CPT lines denied for eligibility verification failure.",
            })
    return pd.DataFrame(rows)


def selftest() -> int:
    fixtures = ROOT / "tests" / "fixtures"
    events_path = fixtures / "events_synthetic.csv"
    notes_path = fixtures / "notes_synthetic.csv"
    if not events_path.exists() or not notes_path.exists():
        print(f"selftest: fixtures missing under {fixtures}", file=sys.stderr)
        return 2

    events = load_events(events_path)
    notes = load_notes(notes_path)
    notes = notes.rename(columns={"accountnumber": "accountnumber"})  # noqa: keep explicit
    synth = _stub_synthesis(["SYN0000001", "SYN0000002", "SYN0000003"])

    accounts = compute_account_features(events, notes, synth)
    print(f"\nAccounts: {len(accounts)} rows, {len(accounts.columns)} cols")
    print(accounts[[
        "visit", "payor", "dos_quarter", "charges_total",
        "payments_total", "current_balance", "num_biller_touches",
        "num_system_touches", "cycle_time_days", "terminal_root_cause",
        "anomaly_flags",
    ]].to_string(index=False))

    # SYN0000001: charges 4271.16, no payment, two adjustments totaling
    # 134.94 + 4136.22 = 4271.16 -> balance should be ~0.
    syn1 = accounts.loc[accounts["visit"] == "SYN0000001"].iloc[0]
    assert abs(syn1["current_balance"]) < 0.01, f"SYN1 balance wrong: {syn1['current_balance']}"
    assert syn1["payments_total"] == 0, syn1
    assert syn1["num_biller_touches"] == 2, syn1   # synbiller1, synbiller2
    assert syn1["num_system_touches"] == 2, syn1   # PM_ResponseInterface + SHSIndex...
    assert syn1["terminal_root_cause"] == "UNDERPAYMENT", syn1

    # SYN0000003: charges 1060.84, two denial events (line items) but
    # those should NOT count as charges (transaction_type=Denial).
    syn3 = accounts.loc[accounts["visit"] == "SYN0000003"].iloc[0]
    assert abs(syn3["charges_total"] - 1060.84) < 0.01, syn3
    assert syn3["denials_total"] > 0, syn3       # 1728 + 199
    assert syn3["num_biller_touches"] == 1, syn3

    # ORPHAN0000099 (notes-only) should NOT appear in accounts - events drives.
    assert "ORPHAN0000099" not in set(accounts["visit"]), "Orphan leaked into accounts"

    # Clusters: should have 3 rows (one per visit since each has unique
    # payor + terminal + DOS quarter).
    clusters = compute_clusters(accounts)
    print(f"\nClusters: {len(clusters)} rows")
    print(clusters.to_string(index=False))
    assert len(clusters) == 3, clusters
    assert "recovery_rate" in clusters.columns

    # Recovery rate sanity: nobody got paid, so all clusters should have
    # recovery_rate == 0.
    assert (clusters["recovery_rate"].fillna(0) == 0).all(), clusters

    print("\nselftest OK: features computed, clusters aggregated, contracts hold.")
    return 0


# --- CLI ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3 - per-account features + cluster aggregation."
    )
    p.add_argument("--events", type=Path, help="Events CSV path.")
    p.add_argument("--notes", type=Path, help="Notes (redacted) CSV path.")
    p.add_argument(
        "--synth",
        type=Path,
        help="Synthesis CSV from claude_processor.py. Optional - "
             "missing accounts will be marked UNSYNTHESIZED.",
    )
    p.add_argument(
        "--out-accounts",
        type=Path,
        help="Where to write the per-account features CSV.",
    )
    p.add_argument(
        "--out-clusters",
        type=Path,
        help="Where to write the cluster aggregation CSV.",
    )
    p.add_argument(
        "--labor-cost",
        type=float,
        default=DEFAULT_LABOR_COST_PER_TOUCH,
        help=f"Estimated labor cost per biller touch (default {DEFAULT_LABOR_COST_PER_TOUCH}).",
    )
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.selftest:
        return selftest()

    if not args.events or not args.notes:
        print("ERROR: --events and --notes are required (or use --selftest).", file=sys.stderr)
        return 2

    events = load_events(args.events)
    notes = load_notes(args.notes)
    synth = pd.read_csv(args.synth) if args.synth and args.synth.exists() else None

    accounts = compute_account_features(events, notes, synth, args.labor_cost)
    clusters = compute_clusters(accounts)

    out_accounts = args.out_accounts or (ROOT / "data" / "accounts.csv")
    out_clusters = args.out_clusters or (ROOT / "data" / "clusters.csv")
    out_accounts.parent.mkdir(parents=True, exist_ok=True)
    accounts.to_csv(out_accounts, index=False)
    clusters.to_csv(out_clusters, index=False)

    print(f"\nWrote {len(accounts)} account rows -> {out_accounts}")
    print(f"Wrote {len(clusters)} cluster rows -> {out_clusters}")

    print("\nTop 10 clusters by open balance:")
    show = clusters.head(10)[[
        "payor", "terminal_root_cause", "dos_quarter",
        "account_count", "total_open_balance", "recovery_rate",
        "avg_cycle_days", "pct_with_anomaly",
    ]]
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
