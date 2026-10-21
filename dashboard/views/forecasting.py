"""Demand forecasting explorer for one store-SKU."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data import MissingArtifactError, arrays, config, forecasts, series_lookup
from ui import BLUE, GREY, ORANGE, header, plot_layout

header("Demand forecasting", "Next-day LightGBM forecasts with empirical prediction "
       "intervals, against observed sales and reconstructed (stockout-adjusted) demand.")
try:
    pa, store = arrays(), forecasts()
except (MissingArtifactError, FileNotFoundError) as exc:
    st.warning(f"Artifacts missing: {exc}")
    st.stop()

keys = series_lookup()
c1, c2, c3 = st.columns([1, 1, 1])
stores = sorted(keys["store_id"].unique().to_list())
s_id = c1.selectbox("Store", stores, index=0)
prods = sorted(keys.filter(keys["store_id"] == s_id)["product_id"].to_list())
p_id = c2.selectbox("Product", prods, index=0)
h = c3.selectbox("Horizon (days ahead)", sorted(store.fc), index=0)
row = keys.filter((keys["store_id"] == s_id) & (keys["product_id"] == p_id)).row(0, named=True)
sid = int(row["sid"])
st.caption(f"Hierarchy: city {row['city_id']} · group {row['management_group_id']} · "
           f"category ids {row['first_category_id']}/{row['second_category_id']}/"
           f"{row['third_category_id']} (anonymised in the source data)")

dates = np.array(pa.dates, dtype="datetime64[D]")
fc = store.by_target(h)[sid]
rq = store.residuals.get(h)
lo = rq.quantile(fc, 0.1) if rq else None
hi = rq.quantile(fc, 0.9) if rq else None
sales = pa.m["sale_amount"][sid]
dem = pa.m["demand"][sid]
cens = pa.m["censored_truth"][sid] > 0 if "censored_truth" in pa.m else np.zeros_like(sales,
                                                                                       bool)
fig = go.Figure()
ok = np.isfinite(fc)
if rq is not None:
    fig.add_trace(go.Scatter(x=np.concatenate([dates[ok], dates[ok][::-1]]),
                             y=np.concatenate([hi[ok], lo[ok][::-1]]), fill="toself",
                             fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                             name="80% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=dates, y=sales, name="Observed sales", line=dict(color=ORANGE)))
fig.add_trace(go.Scatter(x=dates, y=dem, name="Reconstructed demand",
                         line=dict(color=GREY, dash="dot")))
fig.add_trace(go.Scatter(x=dates[ok], y=fc[ok], name=f"LightGBM forecast (h={h})",
                         line=dict(color=BLUE)))
fig.add_trace(go.Scatter(x=dates[cens], y=sales[cens], mode="markers", name="Stockout day",
                         marker=dict(color=ORANGE, size=7, symbol="x")))
cfg = config()
for name in ("valid_start", "test_folds"):
    v = cfg["splits"][name]
    d0 = v if isinstance(v, str) else v[0][0]
    fig.add_vline(x=np.datetime64(d0).astype("datetime64[ms]").astype(object),
                  line=dict(color=GREY, width=1, dash="dash"))
plot_layout(fig, 430, "Units (dataset-normalised)")
st.plotly_chart(fig, use_container_width=True)
st.caption("Dashed lines: start of validation and of the test window. Forecasts exist only "
           "for out-of-sample origins.")

err = fc[ok] - dem[ok]
c = st.columns(4)
c[0].metric("MAE", f"{np.mean(np.abs(err)):.3f}")
c[1].metric("WAPE", f"{100 * np.abs(err).sum() / max(dem[ok].sum(), 1e-9):.1f}%")
c[2].metric("Bias", f"{100 * err.sum() / max(dem[ok].sum(), 1e-9):+.1f}%")
cov = np.mean((dem[ok] >= lo[ok]) & (dem[ok] <= hi[ok])) if rq else float("nan")
c[3].metric("80% interval coverage", f"{100 * cov:.0f}%")
