"""Result figures and the generated results summary (``reports/results_<config>.md``).

Reads only the JSON/CSV metrics written by the evaluate and simulate stages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from freshguard.config import Config
from freshguard.logging_utils import get_logger, kv
from freshguard.visualization import style

log = get_logger(__name__)
AI, BL = "lgbm_cost_optimal", "traditional_ma7_tuned"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(cfg: Config) -> None:
    style.apply()
    md, fd = cfg.metrics_dir, cfg.figures_dir
    sim: dict[str, Any] = json.loads((md / "simulation_results.json").read_text())
    fm: dict[str, Any] = json.loads((md / "forecast_metrics.json").read_text())
    ov = pl.read_csv(md / "forecast_overall.csv")

    # Forecast accuracy, all methods, test, h=1.
    t = ov.filter((pl.col("window") == "test") & (pl.col("horizon") == 1)).sort("wape",
                                                                               descending=True)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = [style.SERIES[0] if m == "lightgbm" else "#b9b7b0" for m in t["method"]]
    ax.barh(t["method"], t["wape"] * 100, color=colors, height=0.6)
    for y, v in enumerate(t["wape"] * 100):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", color=style.TEXT_2, fontsize=9)
    ax.set_xlabel("WAPE on reconstructed demand, test window (%)")
    ax.set_title("Next-day forecast error: LightGBM vs baselines")
    ax.grid(axis="y", visible=False)
    _save(fig, fd / "10_forecast_accuracy.png")

    # Horizon decay.
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for m, c in (("lightgbm", style.SERIES[0]), (fm["best_baseline_by_valid"], style.SERIES[1]),
                 ("ma7", style.SERIES[2])):
        q = ov.filter((pl.col("window") == "test") & (pl.col("method") == m)).sort("horizon")
        ax.plot(q["horizon"], q["wape"] * 100, marker="o", color=c, label=m)
    ax.set_xlabel("Horizon (days ahead)")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Accuracy by forecast horizon (test)")
    ax.legend()
    _save(fig, fd / "11_horizon_accuracy.png")

    # Pareto frontier.
    par = pl.DataFrame(sim["pareto"])
    pol = sim["base"]["policies"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, c in (("LightGBM quantile", style.SERIES[0]),
                    ("Traditional MA7 + buffer", style.SERIES[1])):
        p = par.filter(pl.col("policy") == name).filter(pl.col("availability") > 0.6)
        ax.plot(p["availability"] * 100, p["waste_units"], marker="o", color=c, label=name)
    for fam, c, mk, lab in ((AI, style.SERIES[0], "*", "FreshGuard (selected on validation)"),
                            (BL, style.SERIES[1], "D", "Baseline (selected on validation)")):
        ax.scatter(pol[fam]["availability"] * 100, pol[fam]["waste_units"], s=220, marker=mk,
                   color=c, edgecolor="white", linewidth=1.5, zorder=5, label=lab)
    ax.set_xlabel("Availability: store-SKU-days without a stockout (%)")
    ax.set_ylabel("Expired units, test window")
    ax.set_title("Waste vs availability frontier (lower-right is better)")
    ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, fd / "12_pareto_frontier.png")

    # Profit decomposition.
    d = sim["base"]["vs_baseline"][AI]
    parts = [("Recovered\nsales margin", d["recovered_sales_margin_eur"]),
             ("Avoided waste\n(at cost)", d["avoided_waste_cost_eur"]),
             ("Avoided\ndisposal", d["avoided_disposal_eur"]),
             ("Holding\ncost", d["holding_cost_change_eur"]),
             ("Ordering\ncost", d["ordering_cost_change_eur"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    cum = 0.0
    for i, (_lab, v) in enumerate(parts):
        ax.bar(i, abs(v), bottom=min(cum, cum + v), color=style.SERIES[0] if v >= 0
               else style.SERIES[1], width=0.6)
        ax.text(i, max(cum, cum + v) + 150, f"€{v:,.0f}", ha="center", fontsize=9,
                color=style.TEXT_2)
        cum += v
    ax.bar(len(parts), cum, color="#52514e", width=0.6)
    ax.text(len(parts), cum + 150, f"€{cum:,.0f}", ha="center", fontsize=9, color=style.TEXT)
    ax.set_xticks(range(len(parts) + 1), [p[0] for p in parts] + ["Net\nbenefit"])
    ax.axhline(0, color=style.TEXT_2, lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("EUR over the 14-day test window")
    ax.set_title("Net benefit decomposition vs baseline (hypothetical EUR, base case)")
    _save(fig, fd / "13_profit_waterfall.png")

    # Sensitivity: shelf life and lead time.
    sens = pl.DataFrame(sim["sensitivity"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, var, xlab in ((axes[0], "shelf_life", "Shelf life (days)"),
                          (axes[1], "lead_time", "Lead time (days)")):
        q = sens.filter(pl.col("varied") == var).sort(var)
        x = q[var].to_numpy()
        ax.plot(x, q["waste_reduction_pct"], marker="o", color=style.SERIES[0],
                label="Waste reduction %")
        ax.plot(x, q["stockout_reduction_pct"], marker="s", color=style.SERIES[1],
                label="Stockout reduction %")
        ax.axhline(0, color=style.TEXT_2, lw=0.8)
        ax.set_xlabel(xlab)
        ax.set_title(f"Impact vs {xlab.split(' (')[0].lower()}")
    axes[0].legend(fontsize=8)
    q = sens.filter(pl.col("varied").is_in(["demand_mode", "noise_sigma"]))
    labels = [f"{m}" if v == "demand_mode" else f"noise σ={s}" for v, m, s in
              zip(q["varied"], q["demand_mode"], q["noise_sigma"], strict=True)]
    axes[2].barh(labels, q["net_benefit_eur"], color=[style.SERIES[0] if v >= 0 else
                                                       style.SERIES[1]
                                                       for v in q["net_benefit_eur"]])
    axes[2].axvline(0, color=style.TEXT_2, lw=0.8)
    axes[2].set_title("Net benefit (€) by demand assumption")
    _save(fig, fd / "14_sensitivity.png")

    # Store distribution.
    st_ = pl.read_csv(md / "impact_by_store_id.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(st_["net_benefit_eur"], bins=60, color=style.SERIES[0])
    axes[0].axvline(0, color=style.TEXT_2, lw=0.8)
    axes[0].set_title("Net benefit per store (14 days, €)")
    axes[0].set_ylabel("Stores")
    axes[1].hist(np.clip(st_["waste_reduction_pct"].to_numpy(), -200, 100), bins=60,
                 color=style.SERIES[2])
    axes[1].axvline(0, color=style.TEXT_2, lw=0.8)
    axes[1].set_title("Waste reduction per store (%, clipped at -200)")
    _save(fig, fd / "15_store_distribution.png")
    log.info(kv("report.figures", dir=fd))
