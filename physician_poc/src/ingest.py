"""
Phase 1 - data ingest and per-account timeline build.

Loads the production events extract and the redacted notes CSV, joins them
on visit/accountnumber, and produces a single chronologically-ordered
event stream per account ready for LLM synthesis.

Design notes:

  * The events file is event-sourced - one row per event (Charges, Payment,
    Denial, Adjustment). Multiple rows per account.
  * The notes file is also event-sourced - one row per biller touch, with
    free text in `notescurrentvalue`. Multiple rows per account.
  * System-generated note rows (eventcreatedby in the SYSTEM_PROCESSES set)
    do not contribute to biller-labor calculations and can be excluded
    from journey-classification input. We keep them in the timeline for
    audit but flag them.
  * Adjustment reversals (sign-flipping, cancelling pairs) are common in
    the events feed. We do NOT net them here - the LLM and downstream
    consumers see the raw ledger and decide how to interpret. Netting
    happens at the cluster-aggregation stage.

Usage:
    from ingest import load_and_join, build_account_timelines
    timelines = build_account_timelines("events.csv", "notes_redacted.csv")
    for visit, timeline in timelines.items():
        ...

CLI:
    python ingest.py --events events.csv --notes notes_redacted.csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

# System actors that emit notes but are not human biller touches. These
# rows still count as timeline events (useful for audit) but do not count
# as biller labor and should be excluded from the LLM journey classification.
SYSTEM_PROCESSES = frozenset(
    {
        "CLAIMAUTOUPLOAD",
        "SHSIndexXMLCreation_Interface_Hosp_PROD",
        "PM_ResponseInterface",
    }
)

EVENTS_REQUIRED = [
    "visit",
    "transaction_type",
    "transaction_date",
    "transaction_amount",
]
NOTES_REQUIRED = [
    "accountnumber",
    "touchstartdate",
    "notescurrentvalue",
    "eventcreatedby",
]


@dataclass
class TimelineEvent:
    """One entry in an account's chronological timeline.

    `source` is either 'event' (from events.csv) or 'note' (from notes CSV).
    Other fields are populated based on source; missing fields are None.
    """

    visit: str
    timestamp: pd.Timestamp
    source: str
    event_kind: str  # event.transaction_type OR note actor type ('biller' / 'system')
    amount: float | None = None
    cpt_code: str | None = None
    denial_category: str | None = None
    note_text: str | None = None
    actor: str | None = None
    is_system: bool = False


# --- Loaders ---------------------------------------------------------------


def load_events(path: Path) -> pd.DataFrame:
    """Read the events CSV and validate the required columns are present."""
    # Force visit to string dtype so Excel-mangled scientific-notation values
    # ("5.072E+11") are preserved verbatim and caught by the regex check
    # below. Without dtype=str, pandas silently coerces to float, which
    # makes the corruption invisible.
    df = pd.read_csv(path, dtype={"visit": str})
    missing = [c for c in EVENTS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Events file is missing required columns: {missing}")
    df["visit"] = df["visit"].astype(str)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    if df["transaction_date"].isna().any():
        bad = df["transaction_date"].isna().sum()
        raise ValueError(
            f"Events file has {bad} unparseable transaction_date value(s). "
            "Re-export with proper datetime formatting."
        )
    if df["visit"].str.contains(r"E\+", regex=True, na=False).any():
        raise ValueError(
            "Events file contains visit IDs in scientific notation (e.g. '5.072E+11'). "
            "Re-export from SQL directly without round-tripping through Excel, "
            "or format the visit column as Text before export."
        )
    return df


def load_notes(path: Path) -> pd.DataFrame:
    """Read the redacted notes CSV and validate the required columns."""
    # Same reasoning as load_events - force string dtype so we can detect
    # Excel scientific-notation corruption ("5.072E+11") on the join key.
    df = pd.read_csv(path, dtype={"accountnumber": str})
    missing = [c for c in NOTES_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Notes file is missing required columns: {missing}")
    df["accountnumber"] = df["accountnumber"].astype(str)
    df["touchstartdate"] = pd.to_datetime(df["touchstartdate"], errors="coerce")
    if df["touchstartdate"].isna().any():
        bad = df["touchstartdate"].isna().sum()
        raise ValueError(
            f"Notes file has {bad} unparseable touchstartdate value(s). "
            "Re-export with proper timestamp formatting (not the Excel mm:ss.s artifact)."
        )
    if df["accountnumber"].str.contains(r"E\+", regex=True, na=False).any():
        raise ValueError(
            "Notes file contains accountnumber values in scientific notation. "
            "Re-export from SQL with the column formatted as Text."
        )
    return df


# --- Joiner / timeline builder --------------------------------------------


def load_and_join(events_path: Path, notes_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both files and return them after light validation.

    Does NOT merge the rows (their schemas are too different). Returns
    (events_df, notes_df) ready for build_account_timelines() to walk
    in lockstep per account.
    """
    events = load_events(events_path)
    notes = load_notes(notes_path)

    event_visits = set(events["visit"].unique())
    note_visits = set(notes["accountnumber"].unique())
    join_overlap = event_visits & note_visits
    print(
        f"events: {len(events):>5} rows / {len(event_visits):>4} visits | "
        f"notes: {len(notes):>5} rows / {len(note_visits):>4} accounts | "
        f"overlap: {len(join_overlap):>4} accounts"
    )
    return events, notes


def _events_to_timeline(events_for_account: pd.DataFrame) -> Iterator[TimelineEvent]:
    visit = str(events_for_account["visit"].iloc[0])
    for _, row in events_for_account.iterrows():
        yield TimelineEvent(
            visit=visit,
            timestamp=row["transaction_date"],
            source="event",
            event_kind=str(row["transaction_type"]),
            amount=float(row["transaction_amount"]) if pd.notna(row["transaction_amount"]) else None,
            cpt_code=row["cpt_code"] if "cpt_code" in row and pd.notna(row.get("cpt_code")) else None,
            denial_category=row["denial_category1"] if "denial_category1" in row and pd.notna(row.get("denial_category1")) else None,
            actor=None,
            is_system=False,
        )


def _notes_to_timeline(notes_for_account: pd.DataFrame) -> Iterator[TimelineEvent]:
    visit = str(notes_for_account["accountnumber"].iloc[0])
    for _, row in notes_for_account.iterrows():
        actor = str(row["eventcreatedby"]) if pd.notna(row["eventcreatedby"]) else ""
        is_system = actor in SYSTEM_PROCESSES
        yield TimelineEvent(
            visit=visit,
            timestamp=row["touchstartdate"],
            source="note",
            event_kind="system" if is_system else "biller",
            amount=None,
            cpt_code=None,
            denial_category=None,
            note_text=str(row["notescurrentvalue"]) if pd.notna(row["notescurrentvalue"]) else None,
            actor=actor,
            is_system=is_system,
        )


def build_account_timelines(
    events_path: Path,
    notes_path: Path,
) -> dict[str, list[TimelineEvent]]:
    """Build a chronologically-sorted timeline per account.

    Returns a dict mapping visit (string) to a list of TimelineEvent objects,
    sorted ascending by timestamp. Accounts that appear only in events or
    only in notes are still included; their timeline just has events from
    one side.
    """
    events, notes = load_and_join(events_path, notes_path)

    all_visits = set(events["visit"].unique()) | set(notes["accountnumber"].unique())
    timelines: dict[str, list[TimelineEvent]] = {}

    events_by_visit = {visit: g for visit, g in events.groupby("visit")}
    notes_by_visit = {acct: g for acct, g in notes.groupby("accountnumber")}

    for visit in sorted(all_visits):
        ts_events: list[TimelineEvent] = []
        if visit in events_by_visit:
            ts_events.extend(_events_to_timeline(events_by_visit[visit]))
        if visit in notes_by_visit:
            ts_events.extend(_notes_to_timeline(notes_by_visit[visit]))
        ts_events.sort(key=lambda e: (e.timestamp, e.source))
        timelines[visit] = ts_events

    return timelines


def summarize_timelines(timelines: dict[str, list[TimelineEvent]]) -> str:
    """Print a one-line-per-account summary suitable for spot-checks."""
    out = []
    for visit, events in timelines.items():
        n_events = sum(1 for e in events if e.source == "event")
        n_human = sum(1 for e in events if e.source == "note" and not e.is_system)
        n_sys = sum(1 for e in events if e.source == "note" and e.is_system)
        if events:
            span = (events[-1].timestamp - events[0].timestamp).days
        else:
            span = 0
        out.append(
            f"  {visit}: {n_events:>3} events, {n_human:>3} biller notes, "
            f"{n_sys:>2} system notes, span={span:>4}d"
        )
    return "\n".join(out)


# --- CLI -------------------------------------------------------------------


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Load events + redacted notes, build per-account timelines."
    )
    parser.add_argument("--events", type=Path, required=True, help="Path to events CSV.")
    parser.add_argument("--notes", type=Path, required=True, help="Path to notes_redacted CSV.")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Print summary for the first N accounts (default: 10). 0 prints all.",
    )
    args = parser.parse_args()

    if not args.events.exists():
        print(f"ERROR: events file not found: {args.events}", file=sys.stderr)
        sys.exit(1)
    if not args.notes.exists():
        print(f"ERROR: notes file not found: {args.notes}", file=sys.stderr)
        sys.exit(1)

    timelines = build_account_timelines(args.events, args.notes)
    print()
    print(f"Built {len(timelines)} per-account timeline(s).")
    print()
    if args.limit == 0:
        print(summarize_timelines(timelines))
    else:
        head = dict(list(timelines.items())[: args.limit])
        print(summarize_timelines(head))
        if len(timelines) > args.limit:
            print(f"  ... and {len(timelines) - args.limit} more.")


if __name__ == "__main__":
    run_cli()
