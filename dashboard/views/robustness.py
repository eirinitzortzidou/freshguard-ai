"""Robustness: nested rolling-origin evaluation and quantile calibration."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from data import MissingArtifactError, csv, nested
from ui import AQUA, BLUE, GREY, YELLOW, eur, header, hypothetical_note, pct, plot_layout

header("Robustness and calibration",
       "Four consecutive weeks, each forecast, calibrated and tuned with data from strictly "
       "earlier days only (nested rolling-origin evaluation).")
try:
    res = nested()
    cov = csv("nested/calibration_coverage.csv")
except MissingArtifactError as exc:
    st.warning(f"{exc} Run `make nested`.")
    st.stop()

AI = "lgbm_cost_optimal"
CAL = {"insample": ("In-sample table (published method)", BLUE),
       "static": ("Out-of-sample, frozen", YELLOW),
       "rolling": ("Out-of-sample, weekly refit", AQUA)}
plan = res["plan"]
st.info("No untouched holdout exists: FreshRetailNet-50K ends on 2024-07-02 and every week "
        "below was inspected during development (labels in the table). The nested design "
        "removes parameter selection on evaluation data, not design knowledge.", icon="ℹ️")

pooled = res["policies"]["pooled"]
fcp = res["forecast"]["pooled"]
c = st.columns(4)
c[0].metric("Walk-forward WAPE (28 days)", pct(100 * fcp["lightgbm_wape"]),
            delta=f"{100 * (fcp['lightgbm_wape'] - fcp['selected_baseline_wape']):+.1f} pp vs "
                  "selected baseline", delta_color="inverse")
prim = pooled[f"{AI}__{plan['primary']}"]["vs_baseline"]
c[1].metric("Waste vs tuned baseline", pct(prim["waste_reduction_pct"]),
            help="Published calibration method, pooled over the 4 outer weeks")
c[2].metric("Stockout events vs baseline", pct(prim["stockout_reduction_pct"]),
            help="Positive = fewer stockout events than the baseline")
c[3].metric("Net benefit (hypothetical)", eur(prim["net_benefit_eur"]))
hypothetical_note()

st.subheader("Outer weeks")
fold_rows = []
pf = pl.DataFrame(res["policies"]["by_fold"])
labels = {f["start"]: f["label"] for f in plan["folds"]}
for f in res["forecast"]["by_fold"]:
    sel = f["baseline_selected_on_tuning"]
    q = pf.filter((pl.col("fold_start") == f["fold_start"])
                  & (pl.col("policy") == f"{AI}__{plan['primary']}")).row(0, named=True)
    fold_rows.append({"Week from": f["fold_start"], "Status": labels[f["fold_start"]],
                      "LightGBM WAPE %": 100 * f["lightgbm_wape"],
                      "Selected baseline WAPE %": 100 * f[f"{sel}_wape"],
                      "Selected baseline": sel, "τ (tuned before week)": q["param"],
                      "Waste red. %": q["vs_baseline_waste_reduction_pct"],
                      "Stockout red. %": q["vs_baseline_stockout_reduction_pct"],
                      "Net benefit €": q["vs_baseline_net_benefit_eur"]})
st.dataframe(pl.DataFrame(fold_rows), hide_index=True, use_container_width=True,
             column_config={k: st.column_config.NumberColumn(format="%.1f") for k in (
                 "LightGBM WAPE %", "Selected baseline WAPE %", "Waste red. %",
                 "Stockout red. %")} | {"Net benefit €": st.column_config.NumberColumn(
                     format="%.0f")})

st.subheader("Does better calibration lead to better decisions?")
fig = go.Figure()
metric = st.radio("Metric", ["vs_baseline_net_benefit_eur", "vs_baseline_waste_reduction_pct",
                             "vs_baseline_stockout_reduction_pct"], horizontal=True,
                  format_func=lambda m: {"vs_baseline_net_benefit_eur": "Net benefit (€)",
                                         "vs_baseline_waste_reduction_pct": "Waste reduction (%)",
                                         "vs_baseline_stockout_reduction_pct":
                                             "Stockout reduction (%)"}[m])
for m, (lab, color) in CAL.items():
    q = pf.filter(pl.col("policy") == f"{AI}__{m}")
    fig.add_trace(go.Bar(x=q["fold_start"], y=q[metric], name=lab, marker_color=color))
plot_layout(fig, 340, "")
fig.update_layout(barmode="group", hovermode="closest")
st.plotly_chart(fig, use_container_width=True)
rows = []
for m, (lab, _) in CAL.items():
    p = pooled[f"{AI}__{m}"]
    ci = p["bootstrap_store_cluster"]["net_benefit_eur"]
    rows.append({"Calibration": lab, "Waste red. %": p["vs_baseline"]["waste_reduction_pct"],
                 "Stockout red. %": p["vs_baseline"]["stockout_reduction_pct"],
                 "Fill rate Δ pp": p["vs_baseline"]["fill_rate_change_pp"],
                 "Net benefit €": p["vs_baseline"]["net_benefit_eur"],
                 "95% CI (store bootstrap)": f"{eur(ci['ci_low'])} to {eur(ci['ci_high'])}"})
st.dataframe(pl.DataFrame(rows), hide_index=True, use_container_width=True,
             column_config={k: st.column_config.NumberColumn(format="%.1f") for k in (
                 "Waste red. %", "Stockout red. %", "Fill rate Δ pp")} | {
                 "Net benefit €": st.column_config.NumberColumn(format="%.0f")})
st.caption("Pooled over the four outer weeks against the traditional MA7 policy whose buffer "
           "is tuned on the same 14 days before each week. The bootstrap resamples stores "
           "with all their days; it does not capture week-to-week shocks shared by all stores.")

st.subheader("Calibration diagnostics")
gt = st.selectbox("Group by", ["all", "fold", "target_day_censored", "volume_class",
                               "demand_class", "observed_sales_uncensored"])
tab = cov.filter((pl.col("group_type") == gt) & (pl.col("metric") == "coverage"))
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                         line=dict(color=GREY, dash="dash")))
for m, (lab, color) in CAL.items():
    for g in tab["group"].unique().sort().to_list():
        q = tab.filter((pl.col("method") == m) & (pl.col("group") == g)).sort("tau")
        fig.add_trace(go.Scatter(x=q["tau"], y=q["value"], mode="lines+markers",
                                 name=f"{lab} · {g}" if gt != "all" else lab,
                                 line=dict(color=color), opacity=0.9))
plot_layout(fig, 420, "Share of outcomes ≤ quantile")
fig.update_xaxes(title="Nominal quantile level τ")
fig.update_layout(hovermode="closest")
st.plotly_chart(fig, use_container_width=True)
pin = res["calibration"]["mean_pinball_all"]
st.caption("Mean pinball loss over τ = 0.05..0.95 (lower is better): " + ", ".join(
    f"{CAL[m][0]} {v:.4f}" for m, v in pin.items()))

st.subheader("Bias and coverage by horizon (published forecasts, test window)")
hz = pl.DataFrame(res["calibration"]["horizon_published_forecasts"])
st.dataframe(hz.filter(pl.col("tau") == 0.9).select(
    ["horizon", "method", "coverage", "bias", "wape"]), hide_index=True,
    use_container_width=True)
st.caption("Coverage of the 0.9 quantile from the static validation table (as published) "
           "and after weekly rolling refits; bias < 0 means under-forecasting.")
