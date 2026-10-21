"""Vectorised baseline and intermittent-demand forecasters.

All functions take a (n_series, n_days) history matrix ``y`` and return a matrix ``level``
where ``level[:, t]`` is the forecast made at the close of day ``t`` (using days ``<= t``
only). Flat forecasters predict the same value for every horizon; seasonal ones are resolved
per target day by :func:`forecast_matrix`.
"""

from __future__ import annotations

import warnings

import numpy as np

FLAT_METHODS = ("naive", "ma7", "ma28", "ses", "croston_sba", "tsb")
SEASONAL_METHODS = ("seasonal_naive", "dow_mean_4w", "ma28_dow_index")


def _trailing_mean(y: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.nan_to_num(y, nan=0.0), axis=1, dtype=np.float64)
    out = np.full(y.shape, np.nan)
    out[:, w - 1:] = c[:, w - 1:]
    out[:, w:] -= c[:, :-w]
    out[:, w - 1:] /= w
    return out


def ses_level(y: np.ndarray, alpha: float) -> np.ndarray:
    """Simple exponential smoothing level, initialised with the first observation."""
    lvl = np.empty(y.shape, dtype=np.float64)
    cur = y[:, 0].astype(np.float64)
    for j in range(y.shape[1]):
        cur = alpha * y[:, j] + (1.0 - alpha) * cur
        lvl[:, j] = cur
    return lvl


def croston_sba(y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Croston's method with the Syntetos-Boylan bias correction (1 - alpha/2)."""
    s, t = y.shape
    z = np.full(s, np.nan)
    p = np.full(s, np.nan)
    q = np.ones(s)
    out = np.full((s, t), np.nan)
    for j in range(t):
        pos = y[:, j] > 0
        first = pos & np.isnan(z)
        z = np.where(first, y[:, j], z)
        p = np.where(first, q, p)
        upd = pos & ~first
        z = np.where(upd, alpha * y[:, j] + (1 - alpha) * z, z)
        p = np.where(upd, alpha * q + (1 - alpha) * p, p)
        q = np.where(pos, 1.0, q + 1.0)
        out[:, j] = np.where(np.isnan(z), 0.0, (1 - alpha / 2) * z / p)
    return out


def tsb(y: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> np.ndarray:
    """Teunter-Syntetos-Babai: smooths demand probability every period (handles obsolescence)."""
    s, t = y.shape
    pos0 = (y[:, 0] > 0).astype(float)
    prob = pos0.copy()
    z = np.where(pos0 > 0, y[:, 0], 0.0)
    out = np.empty((s, t))
    for j in range(t):
        pos = y[:, j] > 0
        prob = beta * pos + (1 - beta) * prob
        z = np.where(pos, alpha * y[:, j] + (1 - alpha) * z, z)
        out[:, j] = prob * z
    return out


def flat_level(y: np.ndarray, method: str, alpha: float = 0.2) -> np.ndarray:
    if method == "naive":
        return y.astype(np.float64)
    if method == "ma7":
        return _trailing_mean(y, 7)
    if method == "ma28":
        return _trailing_mean(y, 28)
    if method == "ses":
        return ses_level(y, alpha)
    if method == "croston_sba":
        return croston_sba(y, alpha)
    if method == "tsb":
        return tsb(y, alpha, alpha)
    raise ValueError(f"unknown flat method {method}")


def forecast_matrix(y: np.ndarray, method: str, horizon: int,
                    alpha: float = 0.2) -> np.ndarray:
    """Forecast for target day ``T`` made at origin ``T - horizon``; shape (S, T).

    Column ``T`` holds the forecast of day ``T``; columns without enough history are NaN.
    """
    s, t = y.shape
    out = np.full((s, t), np.nan)
    if method in FLAT_METHODS:
        lvl = flat_level(y, method, alpha)
        out[:, horizon:] = lvl[:, : t - horizon]
        return out
    # Seasonal: use only same-weekday observations at or before the origin.
    k0 = int(np.ceil(horizon / 7))  # first usable weekly lag
    if method == "seasonal_naive":
        lag = 7 * k0
        out[:, lag:] = y[:, : t - lag]
        return out
    if method == "dow_mean_4w":
        stack = []
        for k in range(k0, k0 + 4):
            m = np.full((s, t), np.nan)
            m[:, 7 * k:] = y[:, : t - 7 * k]
            stack.append(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.stack(stack), axis=0)
    if method == "ma28_dow_index":
        # Level (28-day mean at origin) x day-of-week index estimated on the same window.
        ma = _trailing_mean(y, 28)
        dowm = forecast_matrix(y, "dow_mean_4w", horizon)
        lvl = np.full((s, t), np.nan)
        lvl[:, horizon:] = ma[:, : t - horizon]
        idx = dowm / np.where(lvl > 0, lvl, np.nan)
        idx = np.clip(np.nan_to_num(idx, nan=1.0), 0.3, 3.0)
        return lvl * idx
    raise ValueError(f"unknown method {method}")
