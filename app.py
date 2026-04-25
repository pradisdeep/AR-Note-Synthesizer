"""
RCM Note Synthesizer & Touch-Count Dashboard - Streamlit frontend.

Loads the Phase 2 categorized CSV, runs the Phase 3 transform, and renders:
  * Sidebar filters (Payor Name, Primary DX, LLM Root Cause)
  * Three KPI metrics (At-Risk Balance, Avg Touches, Labor Waste)
  * Plotly scatter of Touches vs Balance with the Burnout Zone shaded
  * Horizontal bar of average touches by root cause
  * Audit table of the underlying rows
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from data_transformer import transform  # noqa: E402

CATEGORIZED_CSV = ROOT / "data" / "categorized_output.csv"

BURNOUT_TOUCHES_THRESHOLD = 5
BURNOUT_BALANCE_THRESHOLD = 300

st.set_page_config(
    page_title="RCM Note Synthesizer",
    page_icon=None,
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return transform(df)


def apply_filters(df: pd.DataFrame, payors, dxs, root_causes) -> pd.DataFrame:
    out = df
    if payors:
        out = out[out["Payor Name"].isin(payors)]
    if dxs:
        out = out[out["Primary DX"].isin(dxs)]
    if root_causes:
        out = out[out["LLM_Root_Cause"].isin(root_causes)]
    return out


def render_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=BURNOUT_TOUCHES_THRESHOLD,
        x1=13,
        y0=0,
        y1=BURNOUT_BALANCE_THRESHOLD,
        fillcolor="rgba(220, 53, 69, 0.18)",
        line=dict(color="rgba(220, 53, 69, 0.55)", width=1, dash="dash"),
        layer="below",
    )
    fig.add_annotation(
        x=12,
        y=BURNOUT_BALANCE_THRESHOLD - 15,
        xref="x",
        yref="y",
        text="<b>Burnout Zone</b><br>touches > 5 &amp; balance &lt; $300",
        showarrow=False,
        font=dict(color="rgb(120, 20, 30)", size=11),
        align="right",
        xanchor="right",
        yanchor="top",
    )

    fig.add_trace(
        go.Scatter(
            x=df["Total Touches"],
            y=df["Current Balance"],
            mode="markers",
            marker=dict(
                size=10,
                color=df["Labor Waste ($)"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(title="Labor Waste ($)"),
                line=dict(width=0.5, color="rgba(0,0,0,0.4)"),
            ),
            customdata=df[["Account Number", "LLM_Root_Cause", "Payor Name"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Payor: %{customdata[2]}<br>"
                "Root Cause: %{customdata[1]}<br>"
                "Touches: %{x}<br>"
                "Balance: $%{y:,.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=480,
        xaxis=dict(title="Total Touches", range=[0, 13], dtick=1),
        yaxis=dict(title="Current Balance ($)", range=[0, 2100]),
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
    )
    return fig


def render_root_cause_bar(df: pd.DataFrame) -> go.Figure:
    grouped = (
        df.groupby("LLM_Root_Cause")["Total Touches"]
        .mean()
        .sort_values(ascending=True)
        .round(2)
    )

    fig = go.Figure(
        go.Bar(
            x=grouped.values,
            y=grouped.index,
            orientation="h",
            text=[f"{v:.2f}" for v in grouped.values],
            textposition="outside",
            marker=dict(color="rgb(31, 119, 180)"),
        )
    )
    fig.update_layout(
        height=380,
        xaxis_title="Average Touches",
        yaxis_title="",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def main() -> None:
    st.title("RCM Note Synthesizer & Touch-Count Dashboard")
    st.caption(
        "Bridging unstructured biller notes and operational BI - find the claims "
        "that cost more in labor to work than their collectible balance."
    )

    if not CATEGORIZED_CSV.exists():
        st.error(
            f"`{CATEGORIZED_CSV.relative_to(ROOT)}` not found. "
            "Run `python src/generate_pms_data.py` then `python src/claude_processor.py` first."
        )
        st.stop()

    data = load_data(CATEGORIZED_CSV)

    st.sidebar.header("Filters")
    payor_options = sorted(data["Payor Name"].unique())
    dx_options = sorted(data["Primary DX"].unique())
    rc_options = sorted(data["LLM_Root_Cause"].dropna().unique())

    selected_payors = st.sidebar.multiselect("Payor Name", payor_options, default=payor_options)
    selected_dxs = st.sidebar.multiselect("Primary DX", dx_options, default=dx_options)
    selected_rcs = st.sidebar.multiselect("LLM Root Cause", rc_options, default=rc_options)

    filtered = apply_filters(data, selected_payors, selected_dxs, selected_rcs)
    st.sidebar.markdown(f"**Rows in view:** {len(filtered):,} of {len(data):,}")

    if filtered.empty:
        st.warning("No rows match the current filters. Widen the selection in the sidebar.")
        st.stop()

    total_balance = filtered["Current Balance"].sum()
    avg_touches = filtered["Total Touches"].mean()
    total_waste = filtered["Labor Waste ($)"].sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Total At-Risk Balance", f"${total_balance:,.0f}")
    k2.metric("Average Touches per Account", f"{avg_touches:.2f}")
    k3.metric("Total Est. Labor Waste ($)", f"${total_waste:,.0f}")

    st.markdown("### Touches vs. Balance - the Burnout Zone")
    st.plotly_chart(render_scatter(filtered), use_container_width=True)
    burnout = filtered[
        (filtered["Total Touches"] > BURNOUT_TOUCHES_THRESHOLD)
        & (filtered["Current Balance"] < BURNOUT_BALANCE_THRESHOLD)
    ]
    st.caption(
        f"{len(burnout)} of {len(filtered)} accounts in view sit inside the Burnout Zone "
        f"(more than {BURNOUT_TOUCHES_THRESHOLD} touches and less than ${BURNOUT_BALANCE_THRESHOLD} to recover)."
    )

    st.markdown("### Average Touches by LLM Root Cause")
    st.plotly_chart(render_root_cause_bar(filtered), use_container_width=True)

    st.markdown("### Audit the LLM classification")
    st.caption(
        "Read the Notes column and check how the model labeled each row. "
        "Use this view to spot misclassifications."
    )
    audit_cols = [
        "Account Number",
        "Payor Name",
        "Primary DX",
        "LLM_Root_Cause",
        "Total Touches",
        "Current Balance",
        "Labor Waste ($)",
        "Notes",
    ]
    st.dataframe(
        filtered[audit_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Current Balance": st.column_config.NumberColumn(format="$%.2f"),
            "Labor Waste ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Notes": st.column_config.TextColumn(width="large"),
        },
    )


if __name__ == "__main__":
    main()
