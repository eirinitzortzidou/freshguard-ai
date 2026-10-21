"""Replenishment policies: turn information available at the close of day ``d`` into an
order-up-to level for the arrival day ``d + L`` (plus point forecasts for the days in
between, used to project usable stock). Order quantities themselves are computed by the
simulator from these targets, so every policy faces identical inventory mechanics.

Traditional policies use **observed sales**, which is what a store without
stockout-aware analytics sees. Forecast-driven policies use the LightGBM forecasts of
reconstructed demand and an empirical demand distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from freshguard.forecasting.baselines import _trailing_mean
from freshguard.forecasting.intervals import ResidualQuantiles


@dataclass(frozen=True)
class PolicySpec:
    name: str
    family: str          # traditional | forecast
    kind: str            # ma7_sales | naive_sales | seasonal_sales | ma7_demand |
    #                      point | quantile | normal
    param: float = 0.0   # buffer (traditional) or service level / tau (forecast)
    label: str = ""

    @property
    def display(self) -> str:
        return self.label or self.name


@dataclass
class PolicyInputs:
    """Everything a policy may look at, indexed by panel day (columns) and series (rows).

    ``forecasts[h][:, t]`` is the LightGBM point forecast of day ``t + h`` made at close of
    day ``t``. ``residuals[h]`` turns it into quantiles.
    """

    sales: np.ndarray
    demand_hist: np.ndarray
    forecasts: dict[int, np.ndarray]
    residuals: dict[int, ResidualQuantiles]


def critical_fractile(unit_price: float, unit_cost: float, disposal_cost: float,
                      goodwill: float = 0.0, salvage: float = 0.0) -> float:
    """Single-period newsvendor critical ratio Cu / (Cu + Co).

    Cu (underage) = lost margin + goodwill; Co (overage) = unit cost + disposal - salvage.
    This ignores carry-over to later days, so for shelf lives above one day it is a
    conservative (low) service level; the cost-optimised policy tunes tau by simulation.
    """
    cu = unit_price - unit_cost + goodwill
    co = unit_cost + disposal_cost - salvage
    if cu <= 0 or co <= 0:
        raise ValueError("underage and overage costs must be positive")
    return cu / (cu + co)


def _sales_signal(inp: PolicyInputs, kind: str, L: int) -> np.ndarray:
    """Per-origin traditional estimate of next-day demand (S, T)."""
    if kind == "ma7_sales":
        return _trailing_mean(inp.sales, 7)
    if kind == "ma7_demand":
        return _trailing_mean(inp.demand_hist, 7)
    if kind == "naive_sales":
        return inp.sales.astype(np.float64)
    if kind == "seasonal_sales":
        # Same weekday as the arrival day, last week: day (d + L - 7) <= d.
        lag = 7 - L if L < 7 else 0
        out = np.full(inp.sales.shape, np.nan)
        out[:, lag:] = inp.sales[:, : inp.sales.shape[1] - lag] if lag else inp.sales
        return out
    raise ValueError(kind)


def build_targets(spec: PolicySpec, inp: PolicyInputs, days: np.ndarray, L: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``targets (S, D)`` and ``mid (S, D, L-1)`` for decision days ``days``."""
    s = inp.sales.shape[0]
    mid = np.zeros((s, days.size, max(L - 1, 0)))
    if spec.family == "traditional":
        sig = _sales_signal(inp, spec.kind, L)[:, days]
        sig = np.nan_to_num(sig, nan=0.0)
        targets = (1.0 + spec.param) * sig
        flat = _trailing_mean(inp.sales if spec.kind != "ma7_demand" else inp.demand_hist,
                              7)[:, days]
        for k in range(L - 1):
            mid[:, :, k] = np.nan_to_num(flat, nan=0.0)
        return targets, mid
    fc = inp.forecasts[L][:, days]
    rq = inp.residuals[L]
    if spec.kind == "point":
        targets = fc
    elif spec.kind == "quantile":
        targets = rq.quantile(fc, spec.param)
    elif spec.kind == "normal":
        targets = rq.normal_quantile(fc, spec.param)
    else:
        raise ValueError(spec.kind)
    for k in range(L - 1):
        mid[:, :, k] = np.nan_to_num(inp.forecasts[k + 1][:, days], nan=0.0)
    return np.nan_to_num(targets, nan=0.0), mid
