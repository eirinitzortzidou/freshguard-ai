"""Small presentation helpers shared by the dashboard pages."""

from __future__ import annotations

import math

import streamlit as st

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#9a9893"
CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
div[data-testid="stMetric"] {background: var(--secondary-background-color);
  border-radius: 10px; padding: 12px 16px;}
div[data-testid="stMetricLabel"] p {font-size: 0.85rem;}
.fg-note {font-size: 0.85rem; opacity: 0.8;}
.fg-badge {display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.75rem;
  background: rgba(235,104,52,0.15); color: inherit; margin-left: 6px;}
</style>
"""


def header(title: str, subtitle: str = "") -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def eur(x: float, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    sign = "-" if x < 0 else ""
    return f"{sign}€{abs(x):,.{digits}f}"


def pct(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}%"


def hypothetical_note() -> None:
    st.markdown(
        "<span class='fg-note'>Financial values are <b>simulated, hypothetical EUR</b> "
        "outcomes from an offline backtest with documented cost assumptions. They are not "
        "observed retailer savings.</span>", unsafe_allow_html=True)


def plot_layout(fig, height: int = 360, ytitle: str = "") -> None:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified",
                      yaxis_title=ytitle, template="plotly_white")
    fig.update_xaxes(showgrid=False)
