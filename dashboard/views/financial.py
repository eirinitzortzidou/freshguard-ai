"""Financial impact: decomposition, scenarios and distribution across stores."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from data import MissingArtifactError, csv, simulation
from ui import BLUE, ORANGE, eur, header, hypothetical_note, pct, plot_layout

header("Financial impact", "Hypothetical EUR economics applied to simulated unit flows.")
try:
    sim = simulation()
    by_store = csv("impact_by_store_id.csv")
except MissingArtifactError as exc:
    st.warning(str(exc))
    st.stop()
hypothetical_note()
base = sim["base"]
ai, bl = "lgbm_cost_optimal", "traditional_ma7_tuned"
d = base["vs_baseline"][ai]
p_ai, p_bl = base["policies"][ai], base["policies"][bl]

c = st.columns(4)
c[0].metric("Waste cost (at cost) baseline → AI",
            f"{eur(p_bl['waste_cost'])} → {eur(p_ai['waste_cost'])}")
c[1].metric("Lost sales value baseline → AI",
            f"{eur(p_bl['lost_sales_value'])} → {eur(p_ai['lost_sales_value'])}")
c[2].metric("Gross profit change", eur(d["net_benefit_eur"]))
c[3].metric("Incl. goodwill penalty", eur(d["economic_benefit_eur"]),
            help="Adds the non-cash lost-sales goodwill penalty; not part of the cash figure")

parts = [("Recovered sales margin", d["recovered_sales_margin_eur"]),
         ("Avoided waste (at cost)", d["avoided_waste_cost_eur"]),
         ("Avoided disposal", d["avoided_disposal_eur"]),
         ("Holding cost change", d["holding_cost_change_eur"]),
         ("Ordering cost change", d["ordering_cost_change_eur"])]
fig = go.Figure(go.Waterfall(
    x=[p[0] for p in parts] + ["Net benefit"], y=[p[1] for p in parts] + [0],
    measure=["relative"] * len(parts) + ["total"],
    increasing=dict(marker_color=BLUE), decreasing=dict(marker_color=ORANGE),
    totals=dict(marker_color="#52514e")))
plot_layout(fig, 380, "EUR over the test window")
fig.update_layout(title="Where the net benefit comes from (sums exactly, no double counting)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Economic scenarios (policies re-tuned per scenario)")
rows = []
for name, sc in sim["scenarios"].items():
    e, im, ci = sc["economics"], sc["impact"], sc["ci_store_cluster"]
    rows.append({"Scenario": name, "Price €": e["unit_price"], "Margin": e["gross_margin"],
                 "Disposal €/unit": e["disposal_cost"], "Goodwill share": e[
                     "goodwill_share_of_margin"],
                 "Waste red. %": im["waste_reduction_pct"],
                 "Stockout red. %": im["stockout_reduction_pct"],
                 "Net benefit €": im["net_benefit_eur"],
                 "95% CI €": f"{eur(ci['net_benefit_eur']['ci_low'])} to "
                             f"{eur(ci['net_benefit_eur']['ci_high'])}"})
st.dataframe(pl.DataFrame(rows), hide_index=True, use_container_width=True)

st.subheader("Net benefit by store")
fig = go.Figure(go.Histogram(x=by_store["net_benefit_eur"], marker_color=BLUE, nbinsx=60))
plot_layout(fig, 300, "Stores")
fig.update_xaxes(title="Net benefit per store over the test window (EUR)")
st.plotly_chart(fig, use_container_width=True)
s = base["by_store_id_summary"]
st.caption(f"{pct(100 * s['share_groups_net_benefit_positive'])} of {s['groups']} stores "
           f"gain; waste falls in {pct(100 * s['share_groups_waste_reduced'])} and stockouts "
           f"in {pct(100 * s['share_groups_stockouts_reduced'])} of stores.")
