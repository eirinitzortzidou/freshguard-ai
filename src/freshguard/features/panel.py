"""Rectangular (series x day) matrix view of the store-SKU-day panel.

FreshRetailNet-50K is a complete rectangle (every series observed on every day), which lets
features be computed with vectorised NumPy operations along the time axis instead of grouped
DataFrame operations. ``to_arrays`` verifies the rectangle and fails loudly otherwise.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from freshguard.data.schema import HIERARCHY, SERIES_KEYS

MATRIX_COLUMNS: tuple[str, ...] = (
    "sale_amount", "demand", "demand_hist", "stock_hour6_22_cnt", "censored",
    "censored_truth", "instock_share", "discount", "holiday_flag", "activity_flag", "price",
    "precpt", "avg_temperature",
)


@dataclass
class PanelArrays:
    """Series metadata plus one (n_series, n_days) float32 matrix per variable."""

    keys: pl.DataFrame                      # one row per series: hierarchy columns
    dates: list[dt.date]
    m: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_series(self) -> int:
        return self.keys.height

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def day_index(self, day: str | dt.date) -> int:
        d = dt.date.fromisoformat(day) if isinstance(day, str) else day
        try:
            return self.dates.index(d)
        except ValueError as exc:
            raise KeyError(f"{d} not in panel date range") from exc

    def subset(self, series_idx: np.ndarray) -> PanelArrays:
        return PanelArrays(self.keys[series_idx], list(self.dates),
                           {k: v[series_idx] for k, v in self.m.items()})


def to_arrays(panel: pl.DataFrame, columns: tuple[str, ...] = MATRIX_COLUMNS) -> PanelArrays:
    """Pivot a sorted long panel into matrices; raises if the panel is not rectangular."""
    panel = panel.sort([*SERIES_KEYS, "dt"])
    dates = panel["dt"].unique().sort().to_list()
    keys = panel.select(list(HIERARCHY)).unique(subset=list(SERIES_KEYS),
                                                 maintain_order=True)
    n_s, n_d = keys.height, len(dates)
    if panel.height != n_s * n_d:
        raise ValueError(
            f"panel is not rectangular: {panel.height} rows vs {n_s} series x {n_d} days"
        )
    mats = {
        c: panel[c].to_numpy().astype(np.float32).reshape(n_s, n_d)
        for c in columns if c in panel.columns
    }
    return PanelArrays(keys, dates, mats)
