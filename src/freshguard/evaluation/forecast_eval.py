"""Out-of-sample forecast evaluation: LightGBM vs baselines on identical target rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from freshguard.artifacts import load_store, write_json
from freshguard.config import Config
from freshguard.evaluation.metrics import paired_bootstrap_wape_diff, summary
from freshguard.features.panel import PanelArrays
from freshguard.forecasting.backtest import BASELINES, baseline_forecasts, tune_ses_alpha
from freshguard.logging_utils import get_logger, kv
from freshguard.splits import Splits, resolve_splits

log = get_logger(__name__)


def demand_segments(pa: PanelArrays, sp: Splits, target: str = "demand") -> pl.DataFrame:
    """Per-series volume and intermittency classes from the *training* period only."""
    y = pa.m[target][:, : sp.train_end + 1].astype(np.float64)
    sales = pa.m["sale_amount"][:, : sp.train_end + 1].astype(np.float64)
    mean = y.mean(axis=1)
    q30, q80 = np.quantile(mean, [0.3, 0.8])
    volume = np.where(mean >= q80, "high (top 20%)",
                      np.where(mean < q30, "low (bottom 30%)", "medium"))
    # Syntetos-Boylan classes on observed sales occurrence (ADI) and size variability (CV^2).
    pos = sales > 0
    n_pos = pos.sum(axis=1)
    adi = np.where(n_pos > 0, sales.shape[1] / np.maximum(n_pos, 1), np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = np.where(pos, sales, np.nan)
        cv2 = (np.nanstd(nz, axis=1) / np.nanmean(nz, axis=1)) ** 2
    cv2 = np.nan_to_num(cv2, nan=0.0)
    klass = np.where(adi < 1.32, np.where(cv2 < 0.49, "smooth", "erratic"),
                     np.where(cv2 < 0.49, "intermittent", "lumpy"))
    zero_share = 1 - pos.mean(axis=1)
    return pa.keys.with_row_index("sid").with_columns(
        pl.col("sid").cast(pl.Int32),
        pl.Series("mean_demand_train", mean),
        pl.Series("volume_class", volume),
        pl.Series("demand_class", klass),
        pl.Series("zero_sales_share_train", zero_share),
        pl.Series("censored_share_train",
                  pa.m["censored_truth"][:, : sp.train_end + 1].mean(axis=1)
                  if "censored_truth" in pa.m else np.zeros(pa.n_series)),
    )


def _long(pa: PanelArrays, days: np.ndarray, preds: dict[str, np.ndarray], target: str
          ) -> pl.DataFrame:
    s = pa.n_series
    cols: dict[str, Any] = {
        "sid": np.repeat(np.arange(s, dtype=np.int32), days.size),
        "day": np.tile(days.astype(np.int16), s),
        "y": pa.m[target][:, days].ravel(),
        "sales": pa.m["sale_amount"][:, days].ravel(),
        "censored": pa.m.get("censored_truth", pa.m["censored"])[:, days].ravel(),
    }
    for name, mat in preds.items():
        cols[name] = mat[:, days].ravel()
    return pl.DataFrame(cols)


def evaluate_forecasts(cfg: Config, pa: PanelArrays) -> dict[str, Any]:
    sp = resolve_splits(cfg, pa)
    store = load_store(cfg)
    target = cfg["forecasting"]["target"]
    horizons = sorted(store.fc)
    h0 = int(cfg["forecasting"]["primary_horizon"])
    alpha = tune_ses_alpha(pa, sp, target, h0)
    base = baseline_forecasts(pa, horizons, target, ses_alpha=alpha)
    seg = demand_segments(pa, sp, target)
    windows = {"valid": np.arange(sp.valid.start, sp.valid.end + 1),
               "test": np.arange(sp.test_start, sp.test_end + 1)}
    out: dict[str, Any] = {"ses_alpha": alpha, "horizons": horizons, "overall": [],
                           "uncensored_sales": []}
    methods = ["lightgbm", *BASELINES]
    for h in horizons:
        preds = {"lightgbm": store.by_target(h), **{m: base[m][h] for m in BASELINES}}
        for wname, days in windows.items():
            df = _long(pa, days, preds, target)
            y = df["y"].to_numpy()
            unc = df["censored"].to_numpy() == 0
            for m in methods:
                p = df[m].to_numpy()
                ok = np.isfinite(p)
                out["overall"].append({"window": wname, "horizon": h, "method": m,
                                       **summary(y[ok], p[ok])})
                # Standard FreshRetailNet protocol: observed sales on fully in-stock days.
                out["uncensored_sales"].append(
                    {"window": wname, "horizon": h, "method": m,
                     **summary(df["sales"].to_numpy()[ok & unc], p[ok & unc])})
            if h == h0 and wname == "test":
                test_df = df
    overall = pl.DataFrame(out["overall"])
    # Model selection among baselines on the validation window (h0) - never on test.
    vb = overall.filter((pl.col("window") == "valid") & (pl.col("horizon") == h0)
                        & (pl.col("method") != "lightgbm")).sort("wape")
    best_base = vb["method"][0]
    out["best_baseline_by_valid"] = best_base

    # Segment-level analysis on the test window, primary horizon.
    td = test_df.join(seg, on="sid", how="left")
    seg_tables = {}
    for col in ("volume_class", "demand_class", "first_category_id", "store_id", "city_id"):
        rows = []
        for m in ("lightgbm", best_base, "ma7"):
            g = td.group_by(col).agg(
                pl.len().alias("n"), pl.col("y").sum().alias("actual"),
                (pl.col(m) - pl.col("y")).abs().sum().alias("abs_err"),
                (pl.col(m) - pl.col("y")).sum().alias("err"))
            g = g.with_columns(
                (pl.col("abs_err") / pl.col("actual")).alias("wape"),
                (pl.col("err") / pl.col("actual")).alias("bias"), pl.lit(m).alias("method"))
            rows.append(g)
        seg_tables[col] = pl.concat(rows, how="vertical_relaxed").sort([col, "method"])
    for col, tab in seg_tables.items():
        tab.write_csv(cfg.metrics_dir / f"forecast_by_{col}.csv")
    out["by_volume_class"] = seg_tables["volume_class"].to_dicts()
    out["by_demand_class"] = seg_tables["demand_class"].to_dicts()
    # Store and SKU distributions of WAPE.
    st = seg_tables["store_id"].filter(pl.col("wape").is_finite())
    out["store_wape_quantiles"] = {
        m: st.filter(pl.col("method") == m)["wape"].quantile(q)
        for m in ("lightgbm", best_base) for q in [0.5]}
    store_pivot = st.pivot(on="method", index="store_id", values="wape")
    out["share_stores_lgbm_better_than_best_baseline"] = float(
        (store_pivot["lightgbm"] < store_pivot[best_base]).mean())
    sku = td.group_by("sid").agg(
        pl.col("y").sum().alias("actual"),
        (pl.col("lightgbm") - pl.col("y")).abs().sum().alias("ae_l"),
        (pl.col(best_base) - pl.col("y")).abs().sum().alias("ae_b"))
    sku = sku.filter(pl.col("actual") > 0).with_columns(
        (pl.col("ae_l") / pl.col("actual")).alias("wape_l"),
        (pl.col("ae_b") / pl.col("actual")).alias("wape_b"))
    out["sku_wape_quantiles_lightgbm"] = {str(q): float(sku["wape_l"].quantile(q))
                                          for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    out["sku_wape_quantiles_best_baseline"] = {str(q): float(sku["wape_b"].quantile(q))
                                               for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    out["share_skus_lgbm_better"] = float((sku["ae_l"] < sku["ae_b"]).mean())
    sku.write_parquet(cfg.metrics_dir / "sku_wape_test.parquet")

    # Paired uncertainty: store-cluster bootstrap of the WAPE difference on the test rows.
    sid = td["sid"].to_numpy()
    stores = pa.keys["store_id"].to_numpy()[sid]
    ok = np.isfinite(td[best_base].to_numpy())
    out["bootstrap_wape_diff_vs_best_baseline"] = paired_bootstrap_wape_diff(
        td["y"].to_numpy()[ok], td["lightgbm"].to_numpy()[ok], td[best_base].to_numpy()[ok],
        stores[ok], n_boot=int(cfg["finance"]["bootstrap_samples"]), seed=int(cfg["seed"]))
    daily = td.group_by("day").agg(
        ((pl.col("lightgbm") - pl.col("y")).abs().sum() / pl.col("y").sum()).alias("lightgbm"),
        ((pl.col(best_base) - pl.col("y")).abs().sum() / pl.col("y").sum()).alias(best_base),
    ).sort("day")
    out["daily_test_wape"] = daily.to_dicts()
    out["days_lgbm_better"] = int((daily["lightgbm"] < daily[best_base]).sum())
    out["n_test_days"] = daily.height

    # Calibration of the empirical quantiles on the test window.
    rq = store.residuals[h0]
    p = td["lightgbm"].to_numpy().astype(np.float64)
    yv = td["y"].to_numpy().astype(np.float64)
    out["quantile_coverage_test"] = {str(t): rq.coverage(p, yv, t)
                                     for t in (0.5, 0.7, 0.8, 0.9, 0.95)}
    overall.write_csv(cfg.metrics_dir / "forecast_overall.csv")
    pl.DataFrame(out["uncensored_sales"]).write_csv(cfg.metrics_dir /
                                                    "forecast_uncensored_sales.csv")
    seg.write_parquet(cfg.artifacts_dir / "series_segments.parquet")
    write_json(cfg.metrics_dir / "forecast_metrics.json", out)
    log.info(kv("evaluate.done", best_baseline=best_base,
                lgbm_test_wape=overall.filter((pl.col("window") == "test")
                                              & (pl.col("horizon") == h0)
                                              & (pl.col("method") == "lightgbm"))["wape"][0]))
    return out
