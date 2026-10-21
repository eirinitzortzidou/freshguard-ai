"""Model evaluation: baselines, horizons, segments and limitations."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from data import MissingArtifactError, csv, forecast_metrics, forecast_overall
from ui import BLUE, GREY, header, plot_layout

header("Model evaluation", "Out-of-sample accuracy on identical rows for every method. "
       "Target: reconstructed demand.")
try:
    fm, ov = forecast_metrics(), forecast_overall()
    unc = csv("forecast_uncensored_sales.csv")
except MissingArtifactError as exc:
    st.warning(str(exc))
    st.stop()

win = st.radio("Window", ["test", "valid"], horizontal=True)
h = st.selectbox("Horizon", sorted(ov["horizon"].unique().to_list()))
t = ov.filter((pl.col("window") == win) & (pl.col("horizon") == h)).sort("wape")
fig = go.Figure(go.Bar(x=t["wape"] * 100, y=t["method"], orientation="h",
                       marker_color=[BLUE if m == "lightgbm" else GREY for m in t["method"]],
                       hovertemplate="%{y}: %{x:.2f}%"))
plot_layout(fig, 360, "")
fig.update_xaxes(title="WAPE (%) – lower is better")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, use_container_width=True)
st.dataframe(t.select(["method", "mae", "rmse", "wape", "bias"]), hide_index=True,
             use_container_width=True)
st.caption("Uncensored check (FreshRetailNet protocol: observed sales on fully in-stock days)")
st.dataframe(unc.filter((pl.col("window") == win) & (pl.col("horizon") == h))
             .select(["method", "mae", "wape", "bias"]).sort("wape"), hide_index=True,
             use_container_width=True)

st.subheader("Accuracy by horizon (test)")
hz = ov.filter((pl.col("window") == "test") & pl.col("method").is_in(
    ["lightgbm", fm["best_baseline_by_valid"]])).sort("horizon")
fig = go.Figure()
for m, color in (("lightgbm", BLUE), (fm["best_baseline_by_valid"], GREY)):
    q = hz.filter(pl.col("method") == m)
    fig.add_trace(go.Scatter(x=q["horizon"], y=q["wape"] * 100, name=m, mode="lines+markers",
                             line=dict(color=color)))
plot_layout(fig, 300, "WAPE (%)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Segments (test, next day)")
c1, c2 = st.columns(2)
c1.dataframe(pl.DataFrame(fm["by_volume_class"]).select(
    ["volume_class", "method", "n", "wape", "bias"]), hide_index=True)
c2.dataframe(pl.DataFrame(fm["by_demand_class"]).select(
    ["demand_class", "method", "n", "wape", "bias"]), hide_index=True)
b = fm["bootstrap_wape_diff_vs_best_baseline"]
st.markdown(
    f"- LightGBM minus best baseline (**{fm['best_baseline_by_valid']}**) WAPE: "
    f"**{100 * b['diff']:+.2f} pp**, 95% store-cluster bootstrap CI "
    f"[{100 * b['ci_low']:+.2f}, {100 * b['ci_high']:+.2f}] pp.\n"
    f"- LightGBM better on {fm['days_lgbm_better']} of {fm['n_test_days']} test days, for "
    f"{100 * fm['share_skus_lgbm_better']:.0f}% of SKUs and "
    f"{100 * fm['share_stores_lgbm_better_than_best_baseline']:.0f}% of stores.\n"
    f"- Empirical quantile coverage on test: " + ", ".join(
        f"τ={k}: {100 * v:.1f}%" for k, v in fm["quantile_coverage_test"].items()))
with st.expander("Limitations"):
    st.markdown(
        "- The target is *reconstructed* demand; errors on censored days are measured "
        "against an estimate, so the uncensored-sales table above is shown as a check.\n"
        "- Category ids are anonymised; no product names (fruit, meat, bakery) are inferred.\n"
        "- 90 training days limit the learnable seasonality to weekly patterns.\n"
        "- Weather of the target day is deliberately not used (not known when ordering).")
