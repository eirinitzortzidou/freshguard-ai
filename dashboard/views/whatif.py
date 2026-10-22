"""What-if simulator: re-run tuning + simulation live with user assumptions."""

from __future__ import annotations

import numpy as np
import polars as pl
import streamlit as st

from data import MissingArtifactError, arrays, config, forecasts
from freshguard.finance.economics import Economics, compare
from freshguard.forecasting.backtest import ForecastStore
from freshguard.inventory.experiment import (
    AI,
    BASELINE,
    SimContext,
    _inputs,
    candidate_policies,
    tune_and_test,
)
from freshguard.inventory.policies import PolicySpec, critical_fractile
from freshguard.inventory.simulator import SimSettings
from freshguard.splits import resolve_splits
from ui import eur, header, pct

header("What-if simulator", "Hypothetical scenario: every number below is recomputed by the "
       "real tuning + simulation code on a random sample of store-SKUs.")
st.info("Hypothetical scenario. Results depend on the assumptions you set and are not "
        "observed retailer outcomes.", icon="ℹ️")
try:
    pa, store = arrays(), forecasts()
except (MissingArtifactError, FileNotFoundError) as exc:
    st.warning(str(exc))
    st.stop()
cfg = config()
base_econ = cfg["finance"]["scenarios"]["base"]
with st.form("whatif"):
    c1, c2, c3 = st.columns(3)
    shelf = c1.slider("Shelf life (days)", 1, 7, int(cfg["inventory"]["shelf_life_days"]))
    lead = c1.select_slider("Supplier lead time (days)", options=sorted(store.fc)[:3],
                            value=int(cfg["inventory"]["lead_time_days"]))
    price = c2.number_input("Shelf price per unit (€)", 0.5, 20.0,
                            float(base_econ["unit_price"]), 0.25)
    margin = c2.slider("Gross margin", 0.10, 0.60, float(base_econ["gross_margin"]), 0.05)
    disposal = c2.number_input("Spoilage disposal cost per unit (€)", 0.0, 2.0,
                               float(base_econ["disposal_cost"]), 0.05)
    holding = c3.slider("Holding cost (% of unit cost per year)", 0, 100,
                        int(100 * base_econ["holding_rate_annual"]), 5)
    goodwill = c3.slider("Lost-sales penalty (share of margin, non-cash)", 0.0, 2.0,
                         float(base_econ["goodwill_share_of_margin"]), 0.1)
    sl = c3.slider("Fixed target service level (optional comparison)", 0.50, 0.99, 0.90, 0.01)
    sizes = [k for k in (500, 1000, 2000, 5000) if k < pa.n_series] + [pa.n_series]
    sizes = sizes if pa.n_series <= 5000 else [500, 1000, 2000, 5000]
    n = st.select_slider("Sample size (store-SKUs)", sizes, min(2000, sizes[-1]))
    go = st.form_submit_button("Run simulation", type="primary")

if go:
    rng = np.random.default_rng(int(cfg["seed"]))
    idx = np.sort(rng.choice(pa.n_series, size=min(n, pa.n_series), replace=False))
    sub = pa.subset(idx)
    sp = resolve_splits(cfg, pa)
    st_sub = ForecastStore(len(idx), pa.n_days, {h: v[idx] for h, v in store.fc.items()},
                           store.residuals)
    inp = _inputs(sub, st_sub)
    econ = Economics(gross_margin=margin, disposal_cost=disposal,
                     holding_rate_annual=holding / 100, goodwill_share_of_margin=goodwill,
                     order_line_cost=float(base_econ["order_line_cost"]), unit_price=price)
    cap = float(cfg["inventory"]["capacity_multiplier"]) * np.maximum(
        sub.m["demand_hist"][:, : sp.train_end + 1].max(axis=1), 1e-3)
    ctx = SimContext(sub.m["demand"].astype(float), np.full((len(idx), pa.n_days), price), cap,
                     SimSettings(lead_time=lead, shelf_life=shelf, check=False,
                                 stockout_tolerance=float(cfg["inventory"]["stockout_tolerance"])),
                     econ, int(cfg["inventory"]["warmup_days"]))
    fams = candidate_policies(econ, price)
    fams = {BASELINE: fams[BASELINE], AI: fams[AI],
            "fixed_sl": [PolicySpec("fixed_sl", "forecast", "quantile", sl)]}
    with st.spinner("Tuning on validation, simulating the test window ..."):
        r = tune_and_test(inp, ctx, (sp.valid.start + 4, sp.valid.end),
                          (sp.test_start, sp.test_end), fams, keep_ledgers=False)
    d = compare(r["test"][BASELINE], r["test"][AI])
    c = st.columns(4)
    c[0].metric("Waste reduction", pct(d["waste_reduction_pct"]))
    c[1].metric("Stockout reduction", pct(d["stockout_reduction_pct"]))
    c[2].metric("Net benefit (sample, test window)", eur(d["net_benefit_eur"]))
    c[3].metric("Chosen τ / baseline buffer",
                f"{r['chosen'][AI]} / {r['chosen'][BASELINE]:.0%}")
    rows = []
    for fam, s in r["test"].items():
        rows.append({"Policy": {BASELINE: "Traditional MA7 + tuned buffer",
                                AI: "FreshGuard cost-optimal", "fixed_sl": f"LightGBM τ={sl}"}[fam],
                     "Waste units": s["waste_units"], "Stockout events": s["stockout_events"],
                     "Fill rate %": 100 * s["fill_rate"], "Gross profit €": s["gross_profit"]})
    st.dataframe(pl.DataFrame(rows), hide_index=True, use_container_width=True)
    cf = critical_fractile(price, price * (1 - margin), disposal, goodwill * price * margin)
    st.caption(f"Sample of {len(idx):,} store-SKUs; demand = reconstructed demand. The "
               f"single-period critical fractile for these costs is {cf:.2f}; the "
               "cost-optimal τ can differ because unsold units carry over until expiry.")
else:
    st.caption("Set assumptions and press **Run simulation**.")
