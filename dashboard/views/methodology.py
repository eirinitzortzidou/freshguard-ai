"""Methodology, assumptions and limitations (what the numbers do and do not mean)."""

from __future__ import annotations

import streamlit as st

from data import config, data_mode, demo_manifest
from ui import header

header("Methodology and limitations",
       "How the results were produced, which inputs are observed and which are assumed.")
cfg = config()
inv, sp = cfg["inventory"], cfg["splits"]

st.subheader("Data")
st.markdown(
    "**FreshRetailNet-50K** (Dingdong-Inc, CC BY 4.0): 50,000 store-SKU series of a fresh "
    "e-grocery retailer, 898 stores, 97 days (2024-03-28 to 2024-07-02), daily and hourly sales, "
    "**hourly stockout flags**, discounts, activity and holiday flags, weather. Category ids are "
    "anonymised and units are normalised by the publisher.")
c1, c2 = st.columns(2)
c1.markdown("**Observed in the data**\n\n- daily and hourly sales\n- hourly stock status\n"
            "- discounts, activity and holiday flags\n- weather\n- anonymised hierarchy ids")
c2.markdown("**Assumed (scenario parameters)**\n\n"
            f"- shelf life {inv['shelf_life_days']} days, lead time {inv['lead_time_days']} day\n"
            "- prices, margins, disposal, holding and goodwill costs (hypothetical EUR)\n"
            f"- capacity {inv['capacity_multiplier']} × max historical daily demand\n"
            "- no pack sizes, supplier minimums or substitution")

st.subheader("Demand reconstruction")
st.markdown(
    "When a shelf is empty, sales understate demand. For each censored day, demand is "
    "estimated from the share of the series' intraday demand profile that fell into in-stock "
    "hours (profiles fitted on training days only), with a fallback for near-full-day "
    "stockouts. This adds about 26% to total units. **Reconstructed demand is an estimate, "
    "not an observation**: forecast errors on censored days are measured against it, so "
    "accuracy is also reported against observed sales on fully in-stock days, and the "
    "simulation is repeated with observed sales and with 1.5 × the estimated lost demand.")

st.subheader("Evaluation")
st.markdown(
    f"- **Original split**: training targets ≤ {sp['train_end']}, validation "
    f"{sp['valid_start']} to {sp['valid_end']} (all tuning), test "
    f"{sp['test_folds'][0][0]} to {sp['test_folds'][-1][1]} in two expanding-window refits. "
    "The test window was inspected more than once during development (the baseline's buffer "
    "grid was widened and the equal-availability comparison was added after the first run), "
    "so it is **not** an untouched holdout.\n"
    "- **Nested rolling-origin evaluation**: four weekly outer folds; forecasts, calibration "
    "tables and policy parameters for each week use only earlier days. See the *Robustness* "
    "page.\n"
    "- **No untouched holdout exists** for this dataset; new data would be needed for one.\n"
    "- Uncertainty: store-cluster paired bootstraps (they do not capture shocks common to "
    "all stores in a week, so they understate uncertainty about a different period).")

st.subheader("Simulated, not observed")
st.markdown(
    "Inventory, deliveries, waste and profit are **simulated** with a FEFO lost-sales model. "
    "Euro values are hypothetical outcomes of documented scenario assumptions, not observed "
    "retailer savings. The system has not been deployed or piloted.")

st.subheader("Known limitations")
st.markdown(
    "- 97 days of history; one demand regime (late-June growth) dominates the test weeks.\n"
    "- Uniform shelf life, lead time and costs for all SKUs (categories are anonymised).\n"
    "- The headline profit-optimal policy trades availability for waste; at the selected "
    "point it has more stockout events than the tuned baseline.\n"
    "- Results reverse if no demand is lost at stockouts (observed-sales sensitivity).")

mode = data_mode()
if mode == "demo":
    m = demo_manifest()
    st.caption(f"Demo mode: per-SKU pages use a random sample of {m.get('n_series', '?')} of "
               f"{m.get('of_series', '?')} series ({m.get('selection', '')}). All aggregate "
               "numbers come from the full pipeline run.")
st.caption("Full documentation: docs/methodology.md, docs/EVALUATION_PROTOCOL.md, "
           "docs/CALIBRATION_STUDY.md, docs/REPRODUCIBILITY_REPORT.md.")
