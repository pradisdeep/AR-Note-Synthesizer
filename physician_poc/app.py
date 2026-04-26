"""
Physician AR Synthesizer - Streamlit dashboard.

Reads the per-account features (accounts.csv) and cluster aggregation
(clusters.csv) produced by transform.py and renders:

  * Sidebar filters (Payor, Terminal Root Cause, DOS Quarter, Anomaly Flag, Status)
  * KPIs: Total Open Balance, Cluster Recovery Rate, Anomaly Account Share,
    Avg Cycle Time
  * Cluster table sorted by open balance with recovery rate, top journey,
    top anomaly flag
  * Cycle-time vs Open Balance scatter coloured by terminal root cause
  * Anomaly Flag prevalence bar
  * Per-account audit table with narrative drill-in

Run from the repo root:
    streamlit run physician_poc/app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ACCOUNTS_CSV = ROOT / "data" / "accounts.csv"
CLUSTERS_CSV = ROOT / "data" / "clusters.csv"

# Anomaly flags worth highlighting individually in the prevalence chart.
KNOWN_FLAGS = [
    "INCORRECT_WRITEOFF",
    "WRITEOFF_LIKELY_PR",
    "MASS_REGELIG_DENIAL",
    "ADJUSTMENT_REVERSAL_NOISE",
    "EOB_DATA_INTEGRITY",
    "AGED_OPEN",
    "CASCADE_PATTERN",
    "NO_HUMAN_TOUCH",
]

st.set_page_config(
    page_title="Physician AR Synthesizer",
    page_icon=None,
    layout="wide",
)


# --- Data loading ----------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_accounts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"visit": str})
    # Normalise list-shaped columns from comma-joined strings back to lists
    # so we can filter on flags without substring tricks.
    df["anomaly_flags_list"] = (
        df["anomaly_flags"].fillna("").apply(
            lambda s: [t for t in str(s).split(",") if t]
        )
    )
    df["dos_quarter"] = df["dos_quarter"].fillna("Unknown").astype(str)
    df["terminal_root_cause"] = df["terminal_root_cause"].fillna("UNSYNTHESIZED")
    df["narrative"] = df["narrative"].fillna("")
    return df


@st.cache_data(show_spinner=False)
def load_clusters(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# --- Filters ---------------------------------------------------------------


def apply_filters(
    df: pd.DataFrame,
    payors: list[str],
    terminals: list[str],
    quarters: list[str],
    flags: list[str],
    statuses: list[str],
) -> pd.DataFrame:
    out = df
    if payors:
        out = out[out["payor"].isin(payors)]
    if terminals:
        out = out[out["terminal_root_cause"].isin(terminals)]
    if quarters:
        out = out[out["dos_quarter"].isin(quarters)]
    if statuses:
        out = out[out["latest_status_code"].isin(statuses)]
    if flags:
        # Match if any selected flag is present in the row's flag list.
        flag_set = set(flags)
        out = out[out["anomaly_flags_list"].apply(lambda xs: bool(flag_set & set(xs)))]
    return out


# --- Renderers -------------------------------------------------------------


def render_cluster_table(clusters: pd.DataFrame) -> None:
    cols = [
        "payor", "terminal_root_cause", "dos_quarter",
        "account_count", "total_open_balance", "recovery_rate",
        "avg_cycle_days", "avg_biller_touches",
        "top_journey", "top_anomaly_flag", "pct_with_anomaly",
    ]
    present = [c for c in cols if c in clusters.columns]
    st.dataframe(
        clusters[present],
        use_container_width=True,
        hide_index=True,
        column_config={
            "total_open_balance": st.column_config.NumberColumn(
                "Open Balance", format="$%.2f"
            ),
            "recovery_rate": st.column_config.NumberColumn(
                "Recovery Rate", format="%.1f%%",
                help="Cluster-level: sum payments / sum charges.",
            ),
            "avg_cycle_days": st.column_config.NumberColumn(
                "Avg Cycle (days)", format="%.0f"
            ),
            "avg_biller_touches": st.column_config.NumberColumn(
                "Avg Touches", format="%.1f"
            ),
            "pct_with_anomaly": st.column_config.NumberColumn(
                "% Anomaly", format="%.0f%%"
            ),
            "account_count": st.column_config.NumberColumn("Accts"),
            "top_journey": st.column_config.TextColumn("Top Journey", width="medium"),
            "top_anomaly_flag": st.column_config.TextColumn("Top Anomaly", width="medium"),
        },
    )


def render_cycle_balance_scatter(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    plot = df.copy()
    plot["touches_size"] = plot["num_biller_touches"].clip(lower=1) * 4 + 6
    fig = px.scatter(
        plot,
        x="cycle_time_days",
        y="current_balance",
        color="terminal_root_cause",
        size="touches_size",
        hover_data={
            "visit": True,
            "payor": True,
            "dos_quarter": True,
            "anomaly_flags": True,
            "num_biller_touches": True,
            "touches_size": False,
        },
    )
    fig.update_layout(
        height=460,
        xaxis_title="Cycle Time (days from placement to last activity)",
        yaxis_title="Current Balance ($)",
        legend=dict(title="Terminal Root Cause", orientation="v"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def render_flag_prevalence(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    counts = {flag: 0 for flag in KNOWN_FLAGS}
    for flags in df["anomaly_flags_list"]:
        for flag in flags:
            if flag in counts:
                counts[flag] += 1
    series = pd.Series(counts).sort_values(ascending=True)
    series = series[series > 0]
    if series.empty:
        return go.Figure()
    fig = go.Figure(
        go.Bar(
            x=series.values,
            y=series.index,
            orientation="h",
            text=series.values,
            textposition="outside",
            marker=dict(color="rgb(220, 53, 69)"),
        )
    )
    fig.update_layout(
        height=320,
        xaxis_title="Accounts in view with this flag",
        yaxis_title="",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


# --- Main ------------------------------------------------------------------


def main() -> None:
    st.title("Physician AR Synthesizer")
    st.caption(
        "Per-account journey synthesis on event-sourced production AR data. "
        "Each account's full ledger + biller-note timeline is fed to "
        "claude-haiku-4-5 to identify terminal cause, denial journey, "
        "and operational anomalies."
    )

    if not ACCOUNTS_CSV.exists() or not CLUSTERS_CSV.exists():
        st.error(
            f"Missing `{ACCOUNTS_CSV.name}` or `{CLUSTERS_CSV.name}` under "
            f"`{ACCOUNTS_CSV.parent.relative_to(ROOT.parent)}`. "
            "Run `python physician_poc/src/transform.py` first."
        )
        st.stop()

    accounts = load_accounts(ACCOUNTS_CSV)
    clusters = load_clusters(CLUSTERS_CSV)

    # Sidebar
    st.sidebar.header("Filters")
    payor_opts = sorted(accounts["payor"].dropna().unique())
    terminal_opts = sorted(accounts["terminal_root_cause"].dropna().unique())
    quarter_opts = sorted(accounts["dos_quarter"].dropna().unique())
    status_opts = sorted(accounts["latest_status_code"].dropna().unique())
    flag_opts = KNOWN_FLAGS

    sel_payors = st.sidebar.multiselect("Payor", payor_opts, default=payor_opts)
    sel_terminals = st.sidebar.multiselect(
        "Terminal Root Cause", terminal_opts, default=terminal_opts,
        help="Current blocker (or resolution state) on the account.",
    )
    sel_quarters = st.sidebar.multiselect("DOS Quarter", quarter_opts, default=quarter_opts)
    sel_flags = st.sidebar.multiselect(
        "Anomaly Flag (any-of)", flag_opts, default=[],
        help="Show only accounts with at least one of the selected flags.",
    )
    sel_statuses = st.sidebar.multiselect("Latest Status Code", status_opts, default=status_opts)

    filtered = apply_filters(
        accounts, sel_payors, sel_terminals, sel_quarters, sel_flags, sel_statuses
    )
    st.sidebar.markdown(f"**Accounts in view:** {len(filtered):,} of {len(accounts):,}")

    if filtered.empty:
        st.warning("No accounts match the current filters.")
        st.stop()

    # KPIs
    total_open = filtered["current_balance"].clip(lower=0).sum()
    total_charges = filtered["charges_total"].sum()
    total_payments = filtered["payments_total"].sum()
    cluster_recovery = (total_payments / total_charges) if total_charges > 0 else 0.0
    accts_with_flag = (
        filtered["anomaly_flags_list"].apply(lambda xs: len(xs) > 0).sum()
    )
    pct_flag = accts_with_flag / max(len(filtered), 1)
    avg_cycle = filtered["cycle_time_days"].dropna().mean() or 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Open Balance", f"${total_open:,.0f}")
    k2.metric("Recovery Rate", f"{cluster_recovery*100:.1f}%")
    k3.metric(
        "Accounts with Anomaly",
        f"{accts_with_flag} ({pct_flag*100:.0f}%)",
    )
    k4.metric("Avg Cycle Time", f"{avg_cycle:.0f} days")

    # Cluster section
    st.markdown("### Top Clusters by Open Balance")
    st.caption(
        "Each cluster is one (payor × terminal cause × DOS quarter) bucket. "
        "Recovery rate is computed on cluster totals (collected ÷ billed), "
        "not as an average of per-account ratios."
    )
    # Filter the cluster table to match the sidebar selection where keys overlap.
    # Clusters were aggregated on payor / terminal / DOS quarter, so we filter by those.
    cluster_view = clusters[
        clusters["payor"].isin(sel_payors)
        & clusters["terminal_root_cause"].isin(sel_terminals)
        & clusters["dos_quarter"].isin(sel_quarters)
    ].copy()
    if not cluster_view.empty:
        cluster_view["recovery_rate"] = (cluster_view["recovery_rate"] * 100)
        cluster_view["pct_with_anomaly"] = (cluster_view["pct_with_anomaly"] * 100)
    render_cluster_table(cluster_view)

    # Scatter
    st.markdown("### Cycle Time vs Open Balance")
    st.caption(
        "Each dot is one account. Size scales with biller touches. "
        "Top-right = old, expensive, still-open. Bottom-right = old but resolved (good)."
    )
    st.plotly_chart(render_cycle_balance_scatter(filtered), use_container_width=True)

    # Anomalies
    col_a, col_b = st.columns([2, 3])
    with col_a:
        st.markdown("### Anomaly Flag Prevalence")
        st.plotly_chart(render_flag_prevalence(filtered), use_container_width=True)
        st.caption(
            "Counts are per-account: an account with multiple flags counts "
            "in each of its flag rows."
        )
    with col_b:
        st.markdown("### High-Priority Accounts")
        st.caption(
            "Open balance > $0 AND at least one anomaly flag. Sorted by balance."
        )
        priority = (
            filtered[
                (filtered["current_balance"] > 0)
                & (filtered["anomaly_flags_list"].apply(lambda xs: len(xs) > 0))
            ]
            .sort_values("current_balance", ascending=False)
        )
        if priority.empty:
            st.info("No accounts in view have an anomaly flag with open balance.")
        else:
            st.dataframe(
                priority[[
                    "visit", "payor", "terminal_root_cause",
                    "current_balance", "num_biller_touches",
                    "anomaly_flags", "narrative",
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "current_balance": st.column_config.NumberColumn(
                        "Balance", format="$%.2f"
                    ),
                    "num_biller_touches": st.column_config.NumberColumn("Touches"),
                    "narrative": st.column_config.TextColumn("Narrative", width="large"),
                    "anomaly_flags": st.column_config.TextColumn("Flags", width="medium"),
                },
            )

    # Audit table
    st.markdown("### Audit the LLM synthesis")
    st.caption(
        "Read the full per-account picture: financials, journey, flags, narrative. "
        "Use this to spot mislabels and adjust the prompt or taxonomy."
    )
    audit_cols = [
        "visit", "payor", "dos_quarter", "latest_status_code",
        "charges_total", "payments_total", "adjustments_total",
        "current_balance", "num_biller_touches", "num_system_touches",
        "cycle_time_days", "terminal_root_cause", "denial_journey",
        "journey_length", "anomaly_flags", "narrative",
    ]
    present = [c for c in audit_cols if c in filtered.columns]
    st.dataframe(
        filtered[present].sort_values("current_balance", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "charges_total": st.column_config.NumberColumn("Charges", format="$%.2f"),
            "payments_total": st.column_config.NumberColumn("Payments", format="$%.2f"),
            "adjustments_total": st.column_config.NumberColumn("Adjustments", format="$%.2f"),
            "current_balance": st.column_config.NumberColumn("Balance", format="$%.2f"),
            "narrative": st.column_config.TextColumn("Narrative", width="large"),
            "denial_journey": st.column_config.TextColumn("Journey", width="medium"),
            "anomaly_flags": st.column_config.TextColumn("Flags", width="medium"),
        },
    )

    with st.expander("Methodology"):
        st.markdown(
            """
- **Source data**: events_redacted.csv (event-sourced AR ledger:
  Charges/Payment/Denial/Adjustment with dollar amounts and CPT codes)
  + notes_redacted.csv (post-PHI-scrub biller notes).
- **Synthesis**: `claude-haiku-4-5` reads the chronological merge of
  ledger + notes per account and returns
  `{terminal_root_cause, denial_journey, anomaly_flags, narrative}`
  as JSON. Temperature=0.0 for reproducibility.
- **Recovery rate**: collected ÷ billed at the cluster level (not avg
  of per-account ratios) so tiny accounts don't distort the headline.
- **Cycle time**: placement_date → max(last ledger event, last biller touch).
- **Anomaly flags** are independent of the journey: an account can be
  RESOLVED_PAID and still flagged AGED_OPEN for the prior cycle.
            """
        )


if __name__ == "__main__":
    main()
