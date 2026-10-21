"""Forecast error metrics with explicit zero-denominator handling."""

from __future__ import annotations

import numpy as np
import polars as pl


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p))) if y.size else float("nan")


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2))) if y.size else float("nan")


def wape(y: np.ndarray, p: np.ndarray) -> float:
    """Sum |error| / sum |actual|; NaN (not inf) when total actual demand is zero."""
    den = float(np.sum(np.abs(y)))
    return float(np.sum(np.abs(y - p)) / den) if den > 0 else float("nan")


def bias(y: np.ndarray, p: np.ndarray) -> float:
    """Weighted percentage bias: sum(pred - actual) / sum(actual). >0 = overforecast."""
    den = float(np.sum(np.abs(y)))
    return float(np.sum(p - y) / den) if den > 0 else float("nan")


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"mae": mae(y, p), "rmse": rmse(y, p), "wape": wape(y, p), "bias": bias(y, p),
            "n": float(y.size)}


def grouped_metrics(df: pl.DataFrame, by: list[str], y: str = "y",
                    p: str = "pred") -> pl.DataFrame:
    """MAE / WAPE / bias per group (polars expressions, zero-safe WAPE)."""
    err = (pl.col(p) - pl.col(y))
    return (
        df.group_by(by)
        .agg(
            pl.len().alias("n"),
            pl.col(y).sum().alias("actual"),
            err.abs().sum().alias("abs_err"),
            err.sum().alias("err"),
            err.abs().mean().alias("mae"),
        )
        .with_columns(
            pl.when(pl.col("actual") > 0).then(pl.col("abs_err") / pl.col("actual"))
            .otherwise(None).alias("wape"),
            pl.when(pl.col("actual") > 0).then(pl.col("err") / pl.col("actual"))
            .otherwise(None).alias("bias"),
        )
        .sort(by)
    )


def paired_bootstrap_wape_diff(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                               cluster: np.ndarray, n_boot: int = 500,
                               seed: int = 0) -> dict[str, float]:
    """Cluster (e.g. store) bootstrap CI of WAPE(a) - WAPE(b) on identical rows."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(cluster, return_inverse=True)
    k = uniq.size
    ae_a = np.bincount(inv, np.abs(y - p_a), k)
    ae_b = np.bincount(inv, np.abs(y - p_b), k)
    act = np.bincount(inv, np.abs(y), k)
    point = ae_a.sum() / act.sum() - ae_b.sum() / act.sum()
    draws = rng.integers(0, k, size=(n_boot, k))
    cnt = np.zeros((n_boot, k))
    for i in range(n_boot):
        cnt[i] = np.bincount(draws[i], minlength=k)
    a = (cnt @ ae_a) / (cnt @ act)
    b = (cnt @ ae_b) / (cnt @ act)
    diff = a - b
    return {"diff": float(point), "ci_low": float(np.quantile(diff, 0.025)),
            "ci_high": float(np.quantile(diff, 0.975)),
            "share_boot_a_better": float(np.mean(diff < 0))}
