"""Empirical (Mondrian split-conformal style) predictive quantiles for point forecasts.

Residuals ``actual - forecast`` from the validation window are grouped into bins of the
forecast value (low forecasts have small, right-skewed errors; high forecasts larger ones).
The tau-quantile of demand is ``forecast + residual_quantile(tau | bin)``, clipped at zero.
In the main backtest the bins are fitted on validation data only and applied unchanged to
the test window. :class:`RollingQuantiles` chains several tables in time, so a table refitted
on recent out-of-sample residuals can be applied from a given target day onwards (used by the
nested evaluation and the calibration study).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import norm

TAU_GRID = np.round(np.concatenate([[0.01, 0.025], np.arange(0.05, 0.951, 0.025),
                                    [0.975, 0.99]]), 4)


@dataclass
class ResidualQuantiles:
    n_bins: int = 12
    edges: np.ndarray = field(default_factory=lambda: np.array([]))
    taus: np.ndarray = field(default_factory=lambda: TAU_GRID.copy())
    table: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (bins, taus)
    sigma: np.ndarray = field(default_factory=lambda: np.array([]))      # per-bin std

    def fit(self, pred: np.ndarray, actual: np.ndarray) -> ResidualQuantiles:
        ok = np.isfinite(pred) & np.isfinite(actual)
        pred, actual = pred[ok], actual[ok]
        qs = np.quantile(pred, np.linspace(0, 1, self.n_bins + 1)[1:-1])
        self.edges = np.unique(qs)
        b = np.searchsorted(self.edges, pred, side="right")
        nb = self.edges.size + 1
        res = actual - pred
        self.table = np.zeros((nb, self.taus.size))
        self.sigma = np.zeros(nb)
        for i in range(nb):
            r = res[b == i]
            if r.size == 0:
                r = res
            self.table[i] = np.quantile(r, self.taus)
            self.sigma[i] = r.std()
        # Enforce monotone quantiles per bin (guards against tiny bins).
        self.table = np.maximum.accumulate(self.table, axis=1)
        return self

    def _bin(self, pred: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, np.nan_to_num(pred, nan=0.0), side="right")

    def quantile(self, pred: np.ndarray, tau: float) -> np.ndarray:
        """Demand quantile at level ``tau`` for each point forecast (any array shape)."""
        tau = float(np.clip(tau, self.taus[0], self.taus[-1]))
        col = np.array([np.interp(tau, self.taus, row) for row in self.table])
        out = pred + col[self._bin(pred)]
        return np.where(np.isfinite(pred), np.maximum(out, 0.0), np.nan)

    def normal_quantile(self, pred: np.ndarray, service_level: float) -> np.ndarray:
        """Classic safety-stock alternative: forecast + z * sigma(bin)."""
        z = norm.ppf(service_level)
        out = pred + z * self.sigma[self._bin(pred)]
        return np.where(np.isfinite(pred), np.maximum(out, 0.0), np.nan)

    def coverage(self, pred: np.ndarray, actual: np.ndarray, tau: float) -> float:
        ok = np.isfinite(pred) & np.isfinite(actual)
        return float(np.mean(actual[ok] <= self.quantile(pred[ok], tau)))

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps({"n_bins": self.n_bins, "edges": self.edges.tolist(),
                                    "taus": self.taus.tolist(), "table": self.table.tolist(),
                                    "sigma": self.sigma.tolist()}), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> ResidualQuantiles:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(n_bins=d["n_bins"], edges=np.array(d["edges"]), taus=np.array(d["taus"]),
                   table=np.array(d["table"]), sigma=np.array(d["sigma"]))


@dataclass
class RollingQuantiles:
    """Time-indexed sequence of residual tables: ``tables[i]`` applies to target days
    ``starts[i] .. starts[i + 1] - 1``. Each table must be fitted only on residuals of targets
    that are known before ``starts[i]`` is forecast; that is the caller's responsibility and is
    enforced in :mod:`freshguard.evaluation.nested`. Days before the first start use the first
    table."""

    starts: list[int] = field(default_factory=list)
    tables: list[ResidualQuantiles] = field(default_factory=list)

    def add(self, start: int, table: ResidualQuantiles) -> None:
        if self.starts and start <= self.starts[-1]:
            raise ValueError("segments must be added in increasing start order")
        self.starts.append(start)
        self.tables.append(table)

    def segment(self, target_days: np.ndarray) -> np.ndarray:
        if not self.starts:
            raise ValueError("no calibration table fitted")
        idx = np.searchsorted(np.asarray(self.starts), target_days, side="right") - 1
        return np.maximum(idx, 0)

    def _apply(self, pred: np.ndarray, target_days: np.ndarray, fn: str, level: float
               ) -> np.ndarray:
        """``pred`` is (S, D) with column j forecasting target day ``target_days[j]``."""
        out = np.full(pred.shape, np.nan)
        seg = self.segment(np.asarray(target_days))
        for i in np.unique(seg):
            cols = seg == i
            out[:, cols] = getattr(self.tables[i], fn)(pred[:, cols], level)
        return out

    def quantile_at(self, pred: np.ndarray, target_days: np.ndarray, tau: float) -> np.ndarray:
        return self._apply(pred, target_days, "quantile", tau)

    def normal_quantile_at(self, pred: np.ndarray, target_days: np.ndarray,
                           service_level: float) -> np.ndarray:
        return self._apply(pred, target_days, "normal_quantile", service_level)
