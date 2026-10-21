"""Data-quality checks. Records are flagged and reported, never silently dropped."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from freshguard.data.schema import SERIES_KEYS
from freshguard.logging_utils import get_logger, kv

log = get_logger(__name__)


def quality_report(panel: pl.DataFrame, hours_sale: np.ndarray | None,
                   hours_stock: np.ndarray | None, open_start: int = 6,
                   open_end: int = 22) -> dict[str, Any]:
    """Compute data-quality diagnostics for the store-SKU-day panel."""
    n = panel.height
    rep: dict[str, Any] = {"rows": n}
    rep["series"] = panel.select(SERIES_KEYS).n_unique()
    rep["stores"] = panel["store_id"].n_unique()
    rep["products"] = panel["product_id"].n_unique()
    rep["cities"] = panel["city_id"].n_unique()
    for c in ("management_group_id", "first_category_id", "second_category_id",
              "third_category_id"):
        rep[f"n_{c}"] = panel[c].n_unique()
    rep["date_min"] = str(panel["dt"].min())
    rep["date_max"] = str(panel["dt"].max())
    n_dates = panel["dt"].n_unique()
    rep["n_dates"] = n_dates

    # Duplicates on the natural key.
    rep["duplicate_keys"] = int(n - panel.select([*SERIES_KEYS, "dt"]).n_unique())

    # Missing values.
    nulls = panel.null_count().row(0, named=True)
    rep["null_counts"] = {k: int(v) for k, v in nulls.items() if v}
    nan_cols = {c: int(panel[c].is_nan().sum()) for c, t in panel.schema.items()
                if t.is_float() and panel[c].is_nan().any()}
    rep["nan_counts"] = nan_cols

    # Temporal continuity: every series observed on every calendar day in range.
    per_series = panel.group_by(SERIES_KEYS).agg(
        pl.len().alias("n"), pl.col("dt").min().alias("first"), pl.col("dt").max().alias("last")
    )
    span = (per_series["last"] - per_series["first"]).dt.total_days() + 1
    rep["series_with_gaps"] = int((per_series["n"] != span).sum())
    rep["series_not_full_length"] = int((per_series["n"] != n_dates).sum())

    # Invalid values.
    sale = panel["sale_amount"].to_numpy()
    rep["negative_sales_rows"] = int((sale < 0).sum())
    if hours_sale is not None and hours_stock is not None:
        rep["negative_hourly_sales_cells"] = int((hours_sale < 0).sum())
        hourly_sum = hours_sale.sum(axis=1)
        diff = np.abs(hourly_sum - sale)
        rep["hourly_daily_mismatch_rows"] = int((diff > 1e-3 + 1e-3 * np.abs(sale)).sum())
        rep["hourly_daily_max_abs_diff"] = float(diff.max()) if n else 0.0
        stock_open = hours_stock[:, open_start:open_end].sum(axis=1)
        rep["stock_count_mismatch_rows"] = int((stock_open != panel["stock_hour6_22_cnt"]
                                                .to_numpy()).sum())
        rep["stock_status_values"] = sorted(int(v) for v in np.unique(hours_stock))
        rep["sales_during_stockout_hour_cells"] = int(
            ((hours_stock == 1) & (hours_sale > 0)).sum())
    disc = panel["discount"].to_numpy()
    rep["discount_min"] = float(np.nanmin(disc))
    rep["discount_max"] = float(np.nanmax(disc))
    rep["discount_out_of_range_rows"] = int(((disc <= 0) | (disc > 1.5)).sum())

    # Stockout prevalence (censoring), only when the source annotates stock status.
    if hours_stock is not None:
        so = panel["stock_hour6_22_cnt"].to_numpy()
        hours_open = open_end - open_start
        rep["share_days_any_stockout"] = float((so > 0).mean())
        rep["share_days_full_stockout"] = float((so >= hours_open).mean())
        rep["share_open_hours_stockout"] = float(so.sum() / (hours_open * n))
    if "price" in panel.columns:
        price = panel["price"].to_numpy()
        rep["price_missing_rows"] = int(np.isnan(price).sum())
        rep["price_nonpositive_rows"] = int((price <= 0).sum())
        rep["price_quantiles_usd"] = {q: float(np.nanquantile(price, q))
                                      for q in (0.0, 0.5, 0.99, 1.0)}

    # Zero sales and outliers (robust z-score within series, flagged only).
    rep["share_zero_sales_days"] = float((sale == 0).mean())
    med = panel.group_by(SERIES_KEYS, maintain_order=True).agg(
        pl.col("sale_amount").median().alias("med"),
        (pl.col("sale_amount") - pl.col("sale_amount").median()).abs().median().alias("mad"),
    )
    joined = panel.select([*SERIES_KEYS, "sale_amount"]).join(med, on=list(SERIES_KEYS),
                                                               how="left")
    z = (joined["sale_amount"] - joined["med"]) / (1.4826 * joined["mad"] + 1e-6)
    rep["outlier_rows_robust_z_gt_10"] = int((z > 10).sum())
    rep["sale_amount_quantiles"] = {
        q: float(np.quantile(sale, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    }
    rep["sale_amount_granularity_sample"] = sorted({round(float(v), 4) for v in sale[:2000]})[:15]
    # Promotions.
    rep["share_discounted_days"] = float((disc < 0.999).mean())
    rep["share_activity_days"] = float(panel["activity_flag"].to_numpy().mean())
    rep["share_holiday_days"] = float(panel["holiday_flag"].to_numpy().mean())
    log.info(kv("quality_report", rows=n, series=rep["series"],
                duplicates=rep["duplicate_keys"], gaps=rep["series_with_gaps"]))
    return rep


def write_quality_report(rep: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    lines = ["# Data Quality Report", "",
             "Generated by `freshguard.data.validate` from the processed panel. Nothing is "
             "dropped by these checks; anomalies are flagged and quantified.", "",
             "| Check | Value |", "|---|---|"]
    for key, value in rep.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        elif isinstance(value, dict | list):
            value = "`" + json.dumps(value, default=str) + "`"
        lines.append(f"| {key} | {value} |")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
