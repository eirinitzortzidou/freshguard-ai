"""Inventory optimisation: simulated trajectories of baseline vs FreshGuard for one SKU."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots

from data import MissingArtifactError, arrays, forecasts, series_lookup, simulation, trajectories
from ui import BLUE, GREY, ORANGE, header, plot_layout

header("Inventory optimization", "Order-up-to decisions made at each close with only "
       "past information; lead time, shelf life and costs from the base scenario.")
try:
    sim, tr, pa, store = simulation(), trajectories(), arrays(), forecasts()
except (MissingArtifactError, FileNotFoundError) as exc:
    st.warning(str(exc))
    st.stop()
base = sim["base"]
ai, bl = "lgbm_cost_optimal", "traditional_ma7_tuned"
settings = base["settings"]
keys = series_lookup()
c1, c2 = st.columns(2)
s_id = c1.selectbox("Store", sorted(keys["store_id"].unique().to_list()))
p_id = c2.selectbox("Product", sorted(keys.filter(pl.col("store_id") == s_id)["product_id"]
                                      .to_list()))
sid = int(keys.filter((pl.col("store_id") == s_id) & (pl.col("product_id") == p_id))["sid"][0])

t0 = pa.day_index(sim["windows"]["test"][0])
n = tr[f"{ai}__sold"].shape[1]
dates = np.array(pa.dates[t0:t0 + n], dtype="datetime64[D]")
st.markdown(f"**Assumptions** · lead time {settings['lead_time']} day(s) · shelf life "
            f"{settings['shelf_life']} days · FreshGuard service quantile τ = "
            f"{base['chosen_params'][ai]} · baseline buffer = {base['chosen_params'][bl]:.0%}")

last = t0 + n - 1
fc_next = store.fc[settings["lead_time"]][sid, last]
rq = store.residuals[settings["lead_time"]]
q = float(rq.quantile(np.array([fc_next]), base["chosen_params"][ai])[0])
c = st.columns(4)
c[0].metric("Expected demand (next arrival day)", f"{fc_next:.2f}")
c[1].metric("Order-up-to level", f"{q:.2f}")
c[2].metric("Closing stock today", f"{tr[f'{ai}__closing'][sid, -1]:.2f}")
c[3].metric("Recommended order", f"{tr[f'{ai}__ordered'][sid, -1]:.2f}",
            help="Order placed at the last simulated close")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=("Opening stock vs demand", "Expired and lost units"))
dem = pa.m["demand"][sid, t0:t0 + n]
fig.add_trace(go.Bar(x=dates, y=dem, name="Demand", marker_color=GREY, opacity=0.5), 1, 1)
for fam, color, name in ((bl, ORANGE, "Baseline"), (ai, BLUE, "FreshGuard")):
    fig.add_trace(go.Scatter(x=dates, y=tr[f"{fam}__opening"][sid], name=f"{name} stock",
                             line=dict(color=color)), 1, 1)
    fig.add_trace(go.Bar(x=dates, y=tr[f"{fam}__expired"][sid], name=f"{name} expired",
                         marker_color=color), 2, 1)
    fig.add_trace(go.Scatter(x=dates, y=tr[f"{fam}__lost"][sid], name=f"{name} lost",
                             mode="markers", marker=dict(color=color, symbol="x", size=9)), 2, 1)
plot_layout(fig, 580)
fig.update_layout(legend=dict(orientation="h", y=-0.08, x=0))
st.plotly_chart(fig, use_container_width=True)

st.subheader("This SKU over the test window")
rows = []
for fam, name in ((bl, "Baseline"), (ai, "FreshGuard")):
    d = {k: float(tr[f"{fam}__{k}"][sid].sum()) for k in ("sold", "lost", "expired",
                                                         "received")}
    rows.append({"Policy": name, "Received": d["received"], "Sold": d["sold"],
                 "Lost demand": d["lost"], "Expired": d["expired"],
                 "Stockout days": int((tr[f"{fam}__lost"][sid] > 0.05).sum())})
st.dataframe(pl.DataFrame(rows), hide_index=True, use_container_width=True)
