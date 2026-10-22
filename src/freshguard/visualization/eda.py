"""Exploratory data analysis: figures in ``reports/figures`` and ``reports/eda_report.md``.

All statistics are computed from the processed panel; the narrative in the report is
generated from these numbers so it cannot drift from the data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from freshguard.artifacts import write_json
from freshguard.config import Config
from freshguard.features.panel import to_arrays
from freshguard.logging_utils import get_logger, kv
from freshguard.visualization import style

log = get_logger(__name__)
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def run_eda(cfg: Config, panel: pl.DataFrame) -> dict[str, Any]:
    style.apply()
    fig_dir = cfg.figures_dir
    pa = to_arrays(panel)
    sales, demand = pa.m["sale_amount"].astype(float), pa.m["demand"].astype(float)
    cens = pa.m.get("censored_truth", pa.m["censored"]).astype(float)
    dates = np.array(pa.dates, dtype="datetime64[D]")
    st: dict[str, Any] = {"n_series": pa.n_series, "n_days": pa.n_days,
                          "date_range": [str(pa.dates[0]), str(pa.dates[-1])]}
    figs: list[str] = []

    # 1. Trend: observed sales vs reconstructed demand.
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(dates, sales.sum(0), label="Observed sales", color=style.SERIES[1])
    ax.plot(dates, demand.sum(0), label="Reconstructed demand", color=style.SERIES[0])
    hol = pa.m["holiday_flag"].max(0) > 0
    for d in dates[hol]:
        ax.axvspan(d, d + np.timedelta64(1, "D"), color=style.GRID, alpha=0.6, lw=0)
    ax.set_title("Daily units across all store-SKUs (grey = holiday flag)")
    ax.set_ylabel("Units (dataset-normalised)")
    ax.legend(loc="upper left")
    figs.append(_save(fig, fig_dir / "01_daily_trend.png"))
    st["total_sales"] = float(sales.sum())
    st["total_demand"] = float(demand.sum())
    st["lost_share_of_demand"] = float(1 - sales.sum() / demand.sum())
    first_wk, last_wk = sales[:, :7].sum(), sales[:, -7:].sum()
    st["sales_growth_first_to_last_week_pct"] = float(100 * (last_wk / first_wk - 1))

    # 2. Weekday seasonality (non-holiday days) and holiday effect.
    dow = np.array([d.weekday() for d in pa.dates])
    tot = sales.sum(0)
    non_h = ~hol
    # FreshRetailNet's holiday_flag also marks weekends, so the index uses all days.
    dow_mean = np.array([tot[dow == k].mean() for k in range(7)])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(DOW, dow_mean / np.nanmean(dow_mean), color=style.SERIES[0], width=0.6)
    ax.axhline(1, color=style.TEXT_2, lw=1)
    ax.set_title("Day-of-week index of total sales")
    ax.set_ylabel("Index (mean = 1)")
    figs.append(_save(fig, fig_dir / "02_weekday_index.png"))
    st["weekday_index"] = dict(zip(DOW, (dow_mean / np.nanmean(dow_mean)).round(3).tolist(),
                                   strict=True))
    st["holiday_flag_weekend_share"] = float(np.mean(dow[hol] >= 5)) if hol.any() else None
    st["holiday_vs_nonholiday_daily_sales_ratio"] = float(tot[hol].mean() / tot[non_h].mean()) \
        if hol.any() else None

    # 3. SKU-level demand distribution and concentration.
    mean_s = sales.mean(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(np.log10(mean_s + 1e-3), bins=60, color=style.SERIES[0])
    axes[0].set_title("Mean daily sales per store-SKU (log10)")
    axes[0].set_xlabel("log10(units / day)")
    axes[0].set_ylabel("Store-SKUs")
    srt = np.sort(sales.sum(1))[::-1]
    cum = np.cumsum(srt) / srt.sum()
    x = np.arange(1, srt.size + 1) / srt.size
    axes[1].plot(x * 100, cum * 100, color=style.SERIES[0])
    axes[1].plot([0, 100], [0, 100], color=style.GRID)
    axes[1].set_title("Concentration of sales across store-SKUs")
    axes[1].set_xlabel("% of store-SKUs (largest first)")
    axes[1].set_ylabel("% of units")
    figs.append(_save(fig, fig_dir / "03_sku_distribution_concentration.png"))
    st["share_units_top20pct_series"] = float(cum[int(0.2 * srt.size) - 1])
    st["mean_daily_sales_quantiles"] = {str(q): float(np.quantile(mean_s, q))
                                        for q in (0.1, 0.5, 0.9, 0.99)}

    # 4. Store-level differences.
    keys = pa.keys
    store_tot = pl.DataFrame({"store_id": keys["store_id"], "u": sales.sum(1),
                              "c": cens.mean(1)}).group_by("store_id").agg(
        pl.col("u").sum(), pl.col("c").mean())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(store_tot["u"].to_numpy(), bins=50, color=style.SERIES[0])
    axes[0].set_title("Total units per store")
    axes[0].set_xlabel("Units")
    axes[0].set_ylabel("Stores")
    axes[1].hist(store_tot["c"].to_numpy() * 100, bins=50, color=style.SERIES[1])
    axes[1].set_title("Share of store-SKU-days with stockout, per store")
    axes[1].set_xlabel("% of days censored")
    axes[1].set_ylabel("Stores")
    figs.append(_save(fig, fig_dir / "04_store_differences.png"))
    st["n_stores"] = store_tot.height
    st["store_units_p90_p10_ratio"] = float(np.quantile(store_tot["u"], 0.9)
                                            / max(np.quantile(store_tot["u"], 0.1), 1e-9))
    st["store_censored_share_range"] = [float(store_tot["c"].min()),
                                        float(store_tot["c"].max())]

    # 5. Category ids (anonymised: shown as ids, never as product names).
    cat = pl.DataFrame({"cat": keys["first_category_id"], "u": sales.sum(1),
                        "d": demand.sum(1), "c": cens.mean(1)}).group_by("cat").agg(
        pl.col("u").sum(), pl.col("d").sum(), pl.col("c").mean(), pl.len().alias("n")
    ).sort("u", descending=True)
    top = cat.head(15)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    xs = np.arange(top.height)
    ax.bar(xs, top["u"], color=style.SERIES[0], width=0.6, label="Observed sales")
    ax.bar(xs, top["d"] - top["u"], bottom=top["u"], color=style.SERIES[1], width=0.6,
           label="Estimated lost demand")
    ax.set_xticks(xs, [f"cat {c}" for c in top["cat"]], rotation=45, ha="right")
    ax.set_title("Top 15 first-level category ids: sales and estimated lost demand")
    ax.set_ylabel("Units")
    ax.legend()
    figs.append(_save(fig, fig_dir / "05_category_volume.png"))
    cat.write_csv(cfg.metrics_dir / "eda_category_summary.csv")
    st["n_first_categories"] = cat.height
    st["top3_category_share_units"] = float(cat["u"].head(3).sum() / cat["u"].sum())

    # 6. Promotion effect: sales relative to the series mean by discount bucket.
    disc = pa.m["discount"].astype(float)
    rel = sales / (mean_s[:, None] + 1e-6)
    edges = [0, 0.7, 0.8, 0.9, 0.95, 0.999, 1.2]
    labels = ["<0.70", "0.70-0.80", "0.80-0.90", "0.90-0.95", "0.95-1.00", "no discount"]
    b = np.digitize(disc, edges[1:-1])
    act = pa.m["activity_flag"].astype(bool)
    lift = [float(np.mean(rel[(b == i) & (mean_s[:, None] > 0.2)])) for i in range(len(labels))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar(labels, lift, color=style.SERIES[0], width=0.6)
    axes[0].axhline(1, color=style.TEXT_2, lw=1)
    axes[0].set_title("Sales vs series mean by price ratio (discount)")
    axes[0].set_ylabel("Relative sales")
    axes[0].tick_params(axis="x", rotation=30)
    act_lift = [float(np.mean(rel[~act])), float(np.mean(rel[act]))]
    axes[1].bar(["no activity", "activity"], act_lift, color=style.SERIES[2], width=0.5)
    axes[1].axhline(1, color=style.TEXT_2, lw=1)
    axes[1].set_title("Sales vs series mean by activity flag")
    figs.append(_save(fig, fig_dir / "06_promotion_effects.png"))
    st["relative_sales_by_discount_bucket"] = dict(zip(labels, np.round(lift, 3).tolist(),
                                                       strict=True))
    st["relative_sales_activity_vs_not"] = [round(v, 3) for v in act_lift]
    st["corr_note"] = ("descriptive association only: discounts may be targeted at slow "
                       "or fast items, so this is not a causal promotion effect")

    # 7. Censoring: stockout prevalence and its effect on sales.
    so_h = pa.m["stock_hour6_22_cnt"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(cens.mean(1) * 100, bins=50, color=style.SERIES[1])
    axes[0].set_title("Share of days with a stockout, per store-SKU")
    axes[0].set_xlabel("% of days")
    axes[0].set_ylabel("Store-SKUs")
    if so_h.max() > 0:
        bk = [0, 1, 4, 8, 12, 16, 17]
        names = ["0", "1-3", "4-7", "8-11", "12-15", "16 (all day)"]
        bi = np.digitize(so_h, bk[1:-1])
        rs = [float(np.mean(rel[bi == i])) for i in range(len(names))]
        rd = [float(np.mean((demand / (mean_s[:, None] + 1e-6))[bi == i]))
              for i in range(len(names))]
        xs = np.arange(len(names))
        axes[1].bar(xs - 0.18, rs, width=0.36, color=style.SERIES[1], label="Observed sales")
        axes[1].bar(xs + 0.18, rd, width=0.36, color=style.SERIES[0],
                    label="Reconstructed demand")
        axes[1].set_xticks(xs, names)
        axes[1].set_xlabel("Stockout hours between 06:00 and 22:00")
        axes[1].set_ylabel("Relative to series mean sales")
        axes[1].set_title("Sales fall with stockout hours: zero sales != zero demand")
        axes[1].legend()
        st["relative_sales_by_stockout_hours"] = dict(zip(names, np.round(rs, 3).tolist(),
                                                          strict=True))
        st["relative_demand_by_stockout_hours"] = dict(zip(names, np.round(rd, 3).tolist(),
                                                           strict=True))
    figs.append(_save(fig, fig_dir / "07_censoring.png"))
    st["share_days_censored"] = float(cens.mean())
    st["share_series_censored_over_half_days"] = float((cens.mean(1) > 0.5).mean())
    st["share_zero_sales_days"] = float((sales == 0).mean())
    st["share_zero_sales_days_that_are_censored"] = float(
        cens[sales == 0].mean()) if (sales == 0).any() else None

    # 8. Volatility and intermittency.
    cv = sales.std(1) / (mean_s + 1e-6)
    pos = sales > 0
    adi = sales.shape[1] / np.maximum(pos.sum(1), 1)
    with np.errstate(invalid="ignore"):
        nz = np.where(pos, sales, np.nan)
        cv2 = np.nan_to_num((np.nanstd(nz, 1) / np.nanmean(nz, 1)) ** 2)
    klass = np.where(adi < 1.32, np.where(cv2 < 0.49, "smooth", "erratic"),
                     np.where(cv2 < 0.49, "intermittent", "lumpy"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(np.clip(cv, 0, 4), bins=60, color=style.SERIES[0])
    axes[0].set_title("Coefficient of variation of daily sales")
    axes[0].set_xlabel("CV (clipped at 4)")
    axes[0].set_ylabel("Store-SKUs")
    kinds, counts = np.unique(klass, return_counts=True)
    axes[1].bar(kinds, counts, color=style.SERIES[0], width=0.5)
    axes[1].set_title("Syntetos-Boylan demand classes (ADI 1.32, CV2 0.49)")
    axes[1].set_ylabel("Store-SKUs")
    figs.append(_save(fig, fig_dir / "08_volatility_intermittency.png"))
    st["cv_median"] = float(np.median(cv))
    st["demand_classes"] = {k: int(c) for k, c in zip(kinds, counts, strict=True)}

    # 9. Example series.
    rng = np.random.default_rng(int(cfg["seed"]))
    busy = np.nonzero((cens.mean(1) > 0.3) & (mean_s > np.quantile(mean_s, 0.6)))[0]
    pick = rng.choice(busy if busy.size >= 3 else np.arange(pa.n_series), 3, replace=False)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, i in zip(axes, pick, strict=True):
        ax.plot(dates, sales[i], color=style.SERIES[1], label="Observed sales")
        ax.plot(dates, demand[i], color=style.SERIES[0], label="Reconstructed demand")
        cd = cens[i] > 0
        ax.scatter(dates[cd], sales[i][cd], s=10, color=style.TEXT_2, zorder=3,
                   label="Stockout day")
        r = keys.row(int(i), named=True)
        ax.set_title(f"store {r['store_id']} / product {r['product_id']}", fontsize=10)
    axes[0].legend(ncols=3, loc="upper left")
    figs.append(_save(fig, fig_dir / "09_example_series.png"))

    write_json(cfg.metrics_dir / "eda_stats.json", st)
    _write_report(cfg, st, figs)
    log.info(kv("eda.done", figures=len(figs)))
    return st


def _write_report(cfg: Config, st: dict[str, Any], figs: list[str]) -> None:
    rel = Path("figures") / Path(cfg.figures_dir).name if cfg["config_name"] != "base" else \
        Path("figures")
    lines = [f"# Exploratory Data Analysis ({cfg['config_name']})", "",
             "All numbers below are computed by `freshguard eda` "
             "(`src/freshguard/visualization/eda.py`); raw values are in "
             f"`{os.path.relpath(cfg.metrics_dir, cfg.root)}/eda_stats.json`.", ""]
    for f in figs:
        lines += [f"![{f}]({rel / f})", ""]
    lines += ["## Key statistics", "", "```json"]
    import json

    lines += [json.dumps(st, indent=2, default=str), "```", ""]
    out = cfg.path("reports") / f"eda_report_{cfg['config_name']}_stats.md"
    out.write_text("\n".join(lines), encoding="utf-8")
