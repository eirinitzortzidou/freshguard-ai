"""Origin-based feature construction for direct multi-horizon forecasting.

A training/prediction row is a (series, origin day ``t``, horizon ``h``) triple whose target is
demand on day ``t + h``. Every *history* feature only reads matrix columns ``<= t``; the only
target-day inputs are calendar fields and retailer-controlled plans (discount, activity flag)
that are known when the order is placed. ``tests/test_features.py::test_no_future_leakage``
enforces this by perturbing all data after the origin and asserting that features do not change.
"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import polars as pl

from freshguard.features.panel import PanelArrays

CATEGORICAL = ["store_id", "product_id", "city_id", "management_group_id",
               "first_category_id", "second_category_id", "third_category_id"]


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """Value k days earlier (NaN where unavailable). Works on (S, T) matrices."""
    out = np.full_like(x, np.nan)
    if k == 0:
        return x.copy()
    out[:, k:] = x[:, :-k]
    return out


def _rolling(x: np.ndarray, w: int, fn: str) -> np.ndarray:
    """Trailing window statistic ending at (and including) each day; NaN if incomplete."""
    s, t = x.shape
    out = np.full((s, t), np.nan, dtype=np.float32)
    if t < w:
        return out
    win = np.lib.stride_tricks.sliding_window_view(x, w, axis=1)  # (S, T-w+1, w)
    if fn == "mean":
        val = win.mean(axis=2)
    elif fn == "std":
        val = win.std(axis=2)
    elif fn == "median":
        val = np.median(win, axis=2)
    elif fn == "max":
        val = win.max(axis=2)
    elif fn == "min":
        val = win.min(axis=2)
    elif fn == "sum":
        val = win.sum(axis=2)
    else:
        raise ValueError(fn)
    out[:, w - 1:] = val
    return out


def _days_since_positive(x: np.ndarray, cap: int = 60) -> np.ndarray:
    s, t = x.shape
    out = np.zeros((s, t), dtype=np.float32)
    run = np.full(s, cap, dtype=np.float32)
    for j in range(t):
        run = np.where(x[:, j] > 0, 0.0, np.minimum(run + 1.0, cap))
        out[:, j] = run
    return out


def history_column(pa: PanelArrays, target_col: str) -> str:
    """Causal history signal for a target: ``demand_hist`` when the target is ``demand``."""
    if target_col == "demand" and "demand_hist" in pa.m:
        return "demand_hist"
    return target_col


def history_features(pa: PanelArrays, lags: list[int], windows: list[int],
                     target_col: str = "demand") -> dict[str, np.ndarray]:
    """Features indexed by origin day (column j uses data from days <= j only)."""
    y = pa.m[history_column(pa, target_col)].astype(np.float32)
    sales = pa.m["sale_amount"].astype(np.float32)
    f: dict[str, np.ndarray] = {}
    for lag in lags:
        f[f"lag_{lag}"] = _shift(y, lag - 1)  # lag_1 == value on the origin day itself
    for w in windows:
        f[f"rmean_{w}"] = _rolling(y, w, "mean")
        f[f"rstd_{w}"] = _rolling(y, w, "std")
    for w in (7, 28):
        if w in windows:
            f[f"rmedian_{w}"] = _rolling(y, w, "median")
            f[f"rmax_{w}"] = _rolling(y, w, "max")
    f["sales_rmean_7"] = _rolling(sales, 7, "mean")
    f["trend_7_28"] = f["rmean_7"] / (f["rmean_28"] + 1e-3) if 28 in windows else f["rmean_7"]
    f["cv_28"] = f["rstd_28"] / (f["rmean_28"] + 1e-3) if 28 in windows else f["rstd_7"]
    zero = (sales <= 0).astype(np.float32)
    f["zero_share_28"] = _rolling(zero, 28, "mean")
    f["zero_share_7"] = _rolling(zero, 7, "mean")
    f["days_since_sale"] = _days_since_positive(sales)
    so = pa.m["stock_hour6_22_cnt"].astype(np.float32)
    if np.any(so > 0):  # only sources with intraday stock annotation (FreshRetailNet)
        f["stockout_hours_0"] = so
        f["stockout_hours_7"] = _rolling(so, 7, "sum")
    f["censored_share_7"] = _rolling(pa.m["censored"].astype(np.float32), 7, "mean")
    f["censored_share_28"] = _rolling(pa.m["censored"].astype(np.float32), 28, "mean")
    f["instock_share_0"] = pa.m["instock_share"].astype(np.float32)
    f["lost_share_7"] = 1.0 - _rolling(sales, 7, "sum") / (_rolling(y, 7, "sum") + 1e-6)
    f["discount_0"] = pa.m["discount"].astype(np.float32)
    f["discount_rmean_7"] = _rolling(pa.m["discount"].astype(np.float32), 7, "mean")
    f["activity_rmean_7"] = _rolling(pa.m["activity_flag"].astype(np.float32), 7, "mean")
    if "price" in pa.m:
        f["price_0"] = pa.m["price"].astype(np.float32)
    if "precpt" in pa.m:
        f["precpt_0"] = pa.m["precpt"].astype(np.float32)
    if "avg_temperature" in pa.m:
        f["temperature_0"] = pa.m["avg_temperature"].astype(np.float32)
    return f


def build_rows(pa: PanelArrays, origins: list[int], horizon: int, lags: list[int],
               windows: list[int], known_future: list[str], target_col: str = "demand",
               series_idx: np.ndarray | None = None,
               hist: dict[str, np.ndarray] | None = None) -> pl.DataFrame:
    """Assemble the long feature table for the given origin day indices and horizon.

    Rows whose target day lies outside the panel are kept with a null target so that the
    same function produces inference rows.
    """
    if hist is None:
        hist = history_features(pa, lags, windows, target_col)
    s_idx = np.arange(pa.n_series) if series_idx is None else series_idx
    y = pa.m[target_col]
    y_hist = pa.m[history_column(pa, target_col)]
    sales = pa.m["sale_amount"]
    dates = pa.dates
    frames = []
    for t in origins:
        if t < 0 or t >= pa.n_days:
            raise IndexError(f"origin {t} outside panel")
        tgt = t + horizon
        cols: dict[str, np.ndarray | list[object]] = {
            name: arr[s_idx, t] for name, arr in hist.items()
        }
        # Same-weekday-as-target history: T-7k, only when T-7k <= t.
        seas = []
        for k in range(1, 5):
            j = tgt - 7 * k
            col = (y_hist[s_idx, j] if 0 <= j <= t
                   else np.full(len(s_idx), np.nan, np.float32))
            cols[f"seasonal_lag_{k}w"] = col
            seas.append(col)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cols["seasonal_mean_4w"] = np.nanmean(np.vstack(seas), axis=0)
        target_date = dates[t] + dt.timedelta(days=horizon)
        cols["target_dow"] = np.full(len(s_idx), target_date.weekday(), np.int8)
        cols["target_is_weekend"] = np.full(len(s_idx), int(target_date.weekday() >= 5),
                                            np.int8)
        cols["target_dom"] = np.full(len(s_idx), target_date.day, np.int8)
        in_panel = tgt < pa.n_days
        for name in known_future:
            src = {"discount": "discount", "holiday_flag": "holiday_flag",
                   "activity_flag": "activity_flag"}[name]
            cols[f"target_{name}"] = (pa.m[src][s_idx, tgt] if in_panel
                                      else np.full(len(s_idx), np.nan, np.float32))
        if "discount" in known_future:
            cols["target_discount_rel"] = cols["target_discount"] / (
                hist["discount_rmean_7"][s_idx, t] + 1e-3)
        cols["sid"] = s_idx.astype(np.int32)
        cols["origin_idx"] = np.full(len(s_idx), t, np.int16)
        cols["target_idx"] = np.full(len(s_idx), tgt, np.int16)
        cols["horizon"] = np.full(len(s_idx), horizon, np.int8)
        cols["y"] = (y[s_idx, tgt] if in_panel else np.full(len(s_idx), np.nan, np.float32))
        cols["y_sales"] = (sales[s_idx, tgt] if in_panel
                           else np.full(len(s_idx), np.nan, np.float32))
        cols["y_censored"] = (pa.m.get("censored_truth", pa.m["censored"])[s_idx, tgt] if in_panel
                              else np.full(len(s_idx), np.nan, np.float32))
        frames.append(pl.DataFrame(cols))
    out = pl.concat(frames, how="vertical")
    keys = pa.keys.with_row_index("sid").with_columns(pl.col("sid").cast(pl.Int32))
    out = out.join(keys.select(["sid", *CATEGORICAL]), on="sid", how="left")
    return out


def feature_columns(df: pl.DataFrame) -> list[str]:
    exclude = {"sid", "origin_idx", "target_idx", "horizon", "y", "y_sales", "y_censored"}
    return [c for c in df.columns if c not in exclude]
