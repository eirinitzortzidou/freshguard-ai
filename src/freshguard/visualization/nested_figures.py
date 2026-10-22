"""Figures of the nested rolling-origin evaluation and the calibration study."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from freshguard.config import Config
from freshguard.visualization import style

AI = "lgbm_cost_optimal"
CAL_LABEL = {"insample": "in-sample table (original method)",
             "static": "out-of-sample, frozen", "rolling": "out-of-sample, weekly refit"}


def nested_figures(cfg: Config, res: dict[str, Any], cov: pl.DataFrame) -> None:
    style.apply()
    fd = cfg.figures_dir
    fd.mkdir(parents=True, exist_ok=True)
    colors = {"insample": style.SERIES[0], "static": style.SERIES[3],
              "rolling": style.SERIES[2]}

    # 16. Calibration error (empirical - nominal coverage) on the outer weeks.
    cv = cov.filter(pl.col("metric") == "coverage")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for m, lab in CAL_LABEL.items():
        q = cv.filter((pl.col("group_type") == "all") & (pl.col("method") == m)).sort("tau")
        axes[0].plot(q["tau"], 100 * (q["value"] - q["tau"]), marker="o", color=colors[m],
                     label=lab)
    axes[0].set_title("All outer-week store-SKU-days", fontsize=10)
    for g, c in (("fully in stock", style.SERIES[2]),
                 ("censored (stockout hours)", style.SERIES[1])):
        q = cv.filter((pl.col("group_type") == "target_day_censored")
                      & (pl.col("method") == "insample") & (pl.col("group") == g)).sort("tau")
        axes[1].plot(q["tau"], 100 * (q["value"] - q["tau"]), marker="o", color=c,
                     label=f"target day {g}")
    axes[1].set_title("By stock status of the target day (published method)", fontsize=10)
    for ax, loc in zip(axes, ("lower left", "center left"), strict=True):
        ax.axhline(0, color=style.TEXT_2, lw=0.8)
        ax.set_xlabel("Nominal quantile level τ")
        ax.legend(fontsize=8, loc=loc)
    axes[0].set_ylabel("Empirical − nominal coverage (pp)")
    fig.suptitle("Quantile calibration error, nested outer weeks (0 = perfectly calibrated)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(fd / "16_calibration_reliability.png", dpi=130)
    plt.close(fig)

    # 17. Per-fold policy deltas vs the tuned traditional baseline.
    pf = pl.DataFrame(res["policies"]["by_fold"])
    folds = pf["fold_start"].unique(maintain_order=True).to_list()
    metrics = (("vs_baseline_waste_reduction_pct", "Waste reduction vs baseline (%)"),
               ("vs_baseline_stockout_reduction_pct", "Stockout reduction vs baseline (%)"),
               ("vs_baseline_net_benefit_eur", "Net benefit vs baseline (hypothetical €)"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(folds))
    width = 0.26
    for ax, (col, title) in zip(axes, metrics, strict=True):
        for i, m in enumerate(CAL_LABEL):
            q = pf.filter(pl.col("policy") == f"{AI}__{m}")
            ax.bar(x + (i - 1) * width, q[col].to_numpy(), width, color=colors[m],
                   label=CAL_LABEL[m])
        ax.axhline(0, color=style.TEXT_2, lw=0.8)
        ax.set_xticks(x, [f[5:] for f in folds])
        ax.set_xlabel("Outer week starting")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Nested rolling-origin evaluation: FreshGuard policy by calibration method",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(fd / "17_nested_policy_folds.png", dpi=130)
    plt.close(fig)

    # 18. Bias and 90% coverage by horizon for the published backtest forecasts.
    hz = pl.DataFrame(res["calibration"]["horizon_published_forecasts"])
    if hz.height:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        b = hz.filter((pl.col("method") == "static_validation_table") & (pl.col("tau") == 0.5))
        axes[0].bar(b["horizon"].cast(pl.Utf8), b["bias"] * 100, color=style.SERIES[0])
        axes[0].axhline(0, color=style.TEXT_2, lw=0.8)
        axes[0].set_xlabel("Horizon (days)")
        axes[0].set_ylabel("Bias (%) (<0 = under-forecast)")
        axes[0].set_title("Forecast bias on the test window", fontsize=10)
        for m, c, lab in (("static_validation_table", style.SERIES[0], "published static table"),
                          ("rolling_14d", style.SERIES[2], "weekly rolling refit")):
            q = hz.filter((pl.col("method") == m) & (pl.col("tau") == 0.9)).sort("horizon")
            axes[1].plot(q["horizon"], q["coverage"] * 100, marker="o", color=c, label=lab)
        axes[1].axhline(90, color=style.GRID, ls="--", lw=1.5)
        axes[1].set_xlabel("Horizon (days)")
        axes[1].set_ylabel("Coverage of the 0.9 quantile (%)")
        axes[1].set_title("Coverage of the 0.9 quantile, test window", fontsize=10)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fd / "18_horizon_bias_coverage.png", dpi=130)
        plt.close(fig)
