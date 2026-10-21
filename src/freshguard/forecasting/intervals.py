"""Empirical (Mondrian split-conformal style) predictive quantiles for point forecasts.

Residuals ``actual - forecast`` from the validation window are grouped into bins of the
forecast value (low forecasts have small, right-skewed errors; high forecasts larger ones).
The tau-quantile of demand is ``forecast + residual_quantile(tau | bin)``, clipped at zero.
Bins are fitted on validation data only and applied unchanged to the test window.
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
