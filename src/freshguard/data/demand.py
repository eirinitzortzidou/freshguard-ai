"""Latent demand reconstruction from stockout-censored sales.

FreshRetailNet-50K annotates, for every store-SKU-day, which of the 16 opening hours
(06:00-22:00) were out of stock. Observed sales in those hours are censored: the demand that
would have materialised is unobserved. We estimate it with an intraday-profile ratio
estimator:

1. For every series, estimate the share of daily demand that falls into each hour, using only
   fully in-stock training days, shrunk towards the third-level category profile.
2. For a censored day, ``instock_share`` = expected share of daily demand that falls into the
   hours that were in stock. If it is large enough, ``demand = sales / instock_share``
   (capped at ``max_uplift_factor * sales``).
3. If almost the whole day was stocked out, the ratio is unstable; we fall back to
   ``sales + (1 - instock_share) * typical in-stock daily sales`` of the series.

Profiles and levels use training-period days only, so validation and test targets are
reconstructed with information from the past. Reconstructed demand is an *estimate*;
downstream simulations are also run on observed sales (a censored lower bound) to show how
much results depend on it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from freshguard.data.schema import HOURS, SERIES_KEYS
from freshguard.logging_utils import get_logger, kv

log = get_logger(__name__)


@dataclass(frozen=True)
class ReconstructionParams:
    open_start: int = 6
    open_end: int = 22
    min_instock_share: float = 0.25
    profile_shrinkage: float = 20.0
    max_uplift_factor: float = 4.0
    partial_hour_weight: float = 0.5


def series_index(panel: pl.DataFrame) -> np.ndarray:
    """Dense 0..n_series-1 id for a panel sorted by (store_id, product_id, dt)."""
    return (
        panel.select(pl.struct(list(SERIES_KEYS)).rank("dense").alias("sid"))["sid"]
        .to_numpy()
        .astype(np.int64)
        - 1
    )


def _group_sum(idx: np.ndarray, values: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.zeros((n_groups, *values.shape[1:]), dtype=np.float64)
    np.add.at(out, idx, values)
    return out


def hourly_profiles(panel: pl.DataFrame, hours_sale: np.ndarray, hours_stock: np.ndarray,
                    fit_mask: np.ndarray, params: ReconstructionParams
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Shrunk per-series intraday demand profiles (n_series, 24) and per-series level.

    ``fit_mask`` selects the rows (training period) the profiles may be estimated from.
    Returns ``(profiles, clean_daily_mean)``.
    """
    sid = series_index(panel)
    n_series = int(np.max(sid)) + 1
    clean = fit_mask & (hours_stock[:, params.open_start:params.open_end].sum(axis=1) == 0)
    sums = _group_sum(sid[clean], hours_sale[clean].astype(np.float64), n_series)
    n_clean = np.bincount(sid[clean], minlength=n_series).astype(np.float64)
    tot = sums.sum(axis=1)

    cat = panel["third_category_id"].to_numpy()
    _, cat_idx = np.unique(cat, return_inverse=True)
    cat_of_series = np.zeros(n_series, dtype=np.int64)
    cat_of_series[sid] = cat_idx
    cat_sums = _group_sum(cat_of_series, sums, int(np.max(cat_idx)) + 1)
    global_prof = sums.sum(axis=0)
    global_prof = global_prof / max(global_prof.sum(), 1e-12)
    cat_tot = cat_sums.sum(axis=1, keepdims=True)
    cat_prof = np.where(cat_tot > 0, cat_sums / np.maximum(cat_tot, 1e-12), global_prof)

    mean_daily = np.where(n_clean > 0, tot / np.maximum(n_clean, 1), 0.0)
    k = params.profile_shrinkage
    prior = cat_prof[cat_of_series]
    num = sums + k * mean_daily[:, None] * prior
    den = num.sum(axis=1, keepdims=True)
    profiles = np.where(den > 0, num / np.maximum(den, 1e-12), prior)

    # Typical in-stock daily level, falling back to all fit-period days when no clean day.
    fit_sales = panel["sale_amount"].to_numpy().astype(np.float64)
    n_fit = np.bincount(sid[fit_mask], minlength=n_series).astype(np.float64)
    fit_mean = np.bincount(sid[fit_mask], weights=fit_sales[fit_mask],
                           minlength=n_series) / np.maximum(n_fit, 1)
    level = np.where(n_clean >= 3, mean_daily, fit_mean)
    return profiles.astype(np.float64), level


def reconstruct_demand(panel: pl.DataFrame, hours_sale: np.ndarray, hours_stock: np.ndarray,
                       fit_end: str | dt.date, params: ReconstructionParams | None = None
                       ) -> pl.DataFrame:
    """Add ``instock_share``, ``demand``, ``lost_demand`` and ``censored`` to the panel."""
    params = params or ReconstructionParams()
    if hours_sale.shape != (panel.height, HOURS):
        raise ValueError("hourly arrays must align with the panel")
    end = dt.date.fromisoformat(fit_end) if isinstance(fit_end, str) else fit_end
    fit_mask = (panel["dt"] <= end).to_numpy()
    profiles, level = hourly_profiles(panel, hours_sale, hours_stock, fit_mask, params)
    sid = series_index(panel)

    # An hour flagged out-of-stock that still recorded sales was only partly unavailable
    # (the flag is hourly); it is counted as half lost. Zero-sale stockout hours count fully.
    o0, o1 = params.open_start, params.open_end
    so_mask = np.zeros(hours_stock.shape, dtype=bool)
    so_mask[:, o0:o1] = hours_stock[:, o0:o1] == 1
    so_weight = np.where(so_mask, np.where(hours_sale > 0, params.partial_hour_weight, 1.0),
                         0.0)
    stockout_share = (profiles[sid] * so_weight).sum(axis=1)
    instock_share = np.clip(1.0 - stockout_share, 0.0, 1.0)
    sales = panel["sale_amount"].to_numpy().astype(np.float64)
    censored = so_mask.any(axis=1)

    ratio_est = sales / np.maximum(instock_share, 1e-9)
    ratio_est = np.minimum(ratio_est, params.max_uplift_factor * sales)
    level_est = sales + (1.0 - instock_share) * level[sid]
    est = np.where(instock_share >= params.min_instock_share, ratio_est, level_est)
    demand = np.where(censored, np.maximum(est, sales), sales)
    out = panel.with_columns(
        pl.Series("instock_share", instock_share.astype(np.float32)),
        pl.Series("demand", demand.astype(np.float32)),
        pl.Series("lost_demand", (demand - sales).astype(np.float32)),
        pl.Series("censored", censored.astype(np.int8)),
    )
    log.info(kv("reconstruct_demand", rows=out.height,
                censored_share=float(censored.mean()),
                uplift_share=float((demand.sum() - sales.sum()) / max(sales.sum(), 1e-9)),
                fallback_rows=int((censored & (instock_share < params.min_instock_share)).sum())))
    return out


@dataclass(frozen=True)
class ZeroRunParams:
    """Stockout inference for datasets without stock data (M5)."""

    window: int = 28              # days used for the pre-run sales rate
    min_rate: float = 0.5         # only series selling >= this per day can be flagged
    p_threshold: float = 0.01     # flag when P(zero run this long | Poisson rate) < threshold
    max_run_days: int = 28        # longer runs = item not ranged / delisted, not a stockout


def zero_run_reconstruction(sales: np.ndarray, params: ZeroRunParams | None = None
                            ) -> dict[str, np.ndarray]:
    """Infer probable stockouts from implausibly long zero-sales runs.

    For a zero-sales run that starts on day ``s0``, the pre-run rate ``lam`` is the mean sales
    over the ``window`` days before ``s0``. Under a Poisson demand model the probability of
    ``n`` consecutive zero days is ``exp(-lam * n)``; when this is below ``p_threshold`` the
    zero days are treated as stockouts and demand is imputed as ``lam``. Runs longer than
    ``max_run_days`` are treated as the item not being ranged (delisting, seasonal range
    changes) rather than as lost demand, and are left at zero.

    Two versions are returned:

    * ``*_truth`` uses the *full* run length (whether the run continues later). It defines the
      evaluation target and simulation demand; it may use information after a given day, which
      is acceptable for a target but not for features.
    * ``*_hist`` uses only the run length *so far*, so it is safe to use as a feature input.
    """
    from freshguard.data.m5 import zero_run_lengths

    p = params or ZeroRunParams()
    s, t = sales.shape
    x = sales.astype(np.float64)
    c = np.cumsum(x, axis=1)
    pre = np.full((s, t), np.nan)  # pre[:, j] = mean of days j-window .. j-1
    w = p.window
    pre[:, w:] = (c[:, w - 1:-1] - np.concatenate([np.zeros((s, 1)), c[:, :-w - 1]], axis=1)
                  ) / w
    so_far, full, start = zero_run_lengths(x)
    rows = np.arange(s)[:, None].repeat(t, axis=1)
    lam = np.where(start >= 0, pre[rows, np.maximum(start, 0)], np.nan)
    lam = np.nan_to_num(lam, nan=0.0)
    eligible = (start >= 0) & (lam >= p.min_rate)
    flag_truth = eligible & (np.exp(-lam * full) < p.p_threshold) & (full <= p.max_run_days)
    flag_hist = eligible & (np.exp(-lam * so_far) < p.p_threshold) & (
        so_far <= p.max_run_days)
    demand_truth = np.where(flag_truth, lam, x)
    demand_hist = np.where(flag_hist, lam, x)
    log.info(kv("zero_run_reconstruction", flagged_share=float(flag_truth.mean()),
                uplift_share=float((demand_truth.sum() - x.sum()) / max(x.sum(), 1e-9))))
    return {
        "demand": demand_truth.astype(np.float32),
        "demand_hist": demand_hist.astype(np.float32),
        "censored_truth": flag_truth.astype(np.int8),
        "censored": flag_hist.astype(np.int8),
    }
