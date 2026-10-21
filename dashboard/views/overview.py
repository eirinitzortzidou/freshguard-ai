"""Executive overview: headline simulated impact vs the traditional baseline."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from data import MissingArtifactError, forecast_metrics, forecast_overall, simulation
from ui import BLUE, ORANGE, eur, header, hypothetical_note, pct, plot_layout

header("Executive overview",
       "FreshGuard (LightGBM + cost-optimised order-up-to) vs. a traditional 7-day-average "
       "policy, both tuned on validation and evaluated on the untouched test window.")
try:
    sim = simulation()
    fm = forecast_metrics()
    ov = forecast_overall()
except MissingArtifactError as exc:
    st.warning(str(exc))
    st.stop()

base = sim["base"]
ai, bl = "lgbm_cost_optimal", "traditional_ma7_tuned"
d = base["vs_baseline"][ai]
ci = base["bootstrap_store_cluster"]
pol = base["policies"]

c = st.columns(4)
c[0].metric("Waste reduction", pct(d["waste_reduction_pct"]),
            help="(baseline expired units - FreshGuard expired units) / baseline")
c[1].metric("Stockout reduction", pct(d["stockout_reduction_pct"]),
            help="Stockout event = store-SKU-day with lost demand above tolerance")
c[2].metric("Net benefit (test window)", eur(d["net_benefit_eur"]),
            help="Difference in simulated gross profit, hypothetical EUR")
h0 = ov.filter((pl.col("window") == "test") & (pl.col("horizon") == 1))
w_ai = h0.filter(pl.col("method") == "lightgbm")["wape"][0]
w_bb = h0.filter(pl.col("method") == fm["best_baseline_by_valid"])["wape"][0]
c[3].metric("Forecast WAPE (next day)", pct(100 * w_ai),
            delta=f"{100 * (w_ai - w_bb):+.1f} pp vs best baseline",
            delta_color="inverse")
c = st.columns(4)
c[0].metric("Fill rate FreshGuard", pct(100 * pol[ai]["fill_rate"]),
            delta=f"{d['fill_rate_change_pp']:+.1f} pp")
c[1].metric("Availability FreshGuard", pct(100 * pol[ai]["availability"]),
            delta=f"{d['availability_change_pp']:+.1f} pp")
c[2].metric("Waste rate FreshGuard", pct(100 * pol[ai]["waste_rate"]),
            delta=f"{100 * (pol[ai]['waste_rate'] - pol[bl]['waste_rate']):+.1f} pp",
            delta_color="inverse")
c[3].metric("Avg inventory change", pct(d["avg_inventory_change_pct"]),
            delta_color="off")
wr, sr, nb = ci["waste_reduction_pct"], ci["stockout_reduction_pct"], ci["net_benefit_eur"]
st.caption(
    f"95% store-cluster bootstrap CIs: waste reduction {wr['ci_low']:.1f}"
    f" to {wr['ci_high']:.1f}%, stockout reduction {sr['ci_low']:.1f} to {sr['ci_high']:.1f}%,"
    f" net benefit {eur(nb['ci_low'])} to {eur(nb['ci_high'])}."
    f" Test window {sim['windows']['test'][0]} to {sim['windows']['test'][1]},"
    f" {sim['n_series']:,} store-SKUs.")
hypothetical_note()

iso = sim.get("iso_service_vs_baseline", {}).get("availability")
if d["stockout_reduction_pct"] < 0:
    st.warning(
        f"Trade-off: the validation-selected FreshGuard setting cuts waste but has "
        f"{-d['stockout_reduction_pct']:.1f}% more stockout events than the baseline on the "
        "test window (demand rose faster than forecast in late June). See the frontier below.",
        icon="⚖️")
if iso:
    st.markdown(
        f"**At equal availability** ({100 * iso['matched_availability']:.1f}%), the LightGBM "
        f"frontier wastes **{iso['waste_reduction_pct']:.1f}% fewer units** than the baseline "
        f"and earns **{eur(iso['profit_change_eur'])}** more (interpolated on the test grid; "
        "descriptive, not a selected policy).")

st.subheader("All policies on the test window")
rows = []
for fam, s in pol.items():
    rows.append({"Policy": base["labels"].get(fam, fam),
                 "Tuned parameter": base["chosen_params"][fam],
                 "Waste units": s["waste_units"], "Waste rate %": 100 * s["waste_rate"],
                 "Stockout events": s["stockout_events"], "Fill rate %": 100 * s["fill_rate"],
                 "Gross profit €": s["gross_profit"],
                 "Δ profit vs baseline €": base["vs_baseline"][fam]["net_benefit_eur"]})
df = pl.DataFrame(rows).sort("Gross profit €", descending=True)
st.dataframe(df, hide_index=True, use_container_width=True,
             column_config={k: st.column_config.NumberColumn(format="%.1f")
                            for k in ("Waste rate %", "Fill rate %")} | {
                 k: st.column_config.NumberColumn(format="%.0f")
                 for k in ("Waste units", "Stockout events", "Gross profit €",
                           "Δ profit vs baseline €")})

st.subheader("Waste vs availability trade-off")
par = pl.DataFrame(sim["pareto"])
fig = go.Figure()
for name, color in (("LightGBM quantile", BLUE), ("Traditional MA7 + buffer", ORANGE)):
    p = par.filter(pl.col("policy") == name)
    fig.add_trace(go.Scatter(x=100 * p["fill_rate"], y=p["waste_units"], mode="lines+markers",
                             name=name, line=dict(color=color, width=2), marker=dict(size=8),
                             text=[f"param={v}" for v in p["param"]],
                             hovertemplate="%{text}<br>fill %{x:.1f}%<br>waste %{y:,.0f}"))
for fam, color, sym in ((ai, BLUE, "star"), (bl, ORANGE, "diamond")):
    fig.add_trace(go.Scatter(x=[100 * pol[fam]["fill_rate"]], y=[pol[fam]["waste_units"]],
                             mode="markers", name=f"selected: {base['labels'][fam]}",
                             marker=dict(color=color, size=16, symbol=sym,
                                         line=dict(color="white", width=2))))
plot_layout(fig, 420, "Expired units (test window)")
fig.update_xaxes(title="Unit fill rate (%)")
fig.update_layout(hovermode="closest")
st.plotly_chart(fig, use_container_width=True)
st.caption("Each point is one buffer (traditional) or service-level quantile (LightGBM) "
           "evaluated on the test window; the large markers are the settings chosen on "
           "validation. Lower-right is better.")
