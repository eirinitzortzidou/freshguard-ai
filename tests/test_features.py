"""Feature correctness and temporal-leakage prevention."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from freshguard.data.demand import reconstruct_demand
from freshguard.data.ingest import build_panel
from freshguard.features.build import build_rows, history_features
from freshguard.features.panel import to_arrays

LAGS = [1, 2, 7, 14]
WINDOWS = [3, 7, 28]
KNOWN = ["discount", "holiday_flag", "activity_flag"]


@pytest.fixture(scope="module")
def pa(synthetic_raw):
    panel, hs, hst = build_panel(synthetic_raw.drop("true_demand"))
    panel = reconstruct_demand(panel, hs, hst, "2024-05-20").with_columns(
        pl.col("demand").alias("demand_hist"), pl.col("censored").alias("censored_truth"))
    return to_arrays(panel)


def test_rectangular_check(pa, synthetic_raw):
    panel, _, _ = build_panel(synthetic_raw.drop("true_demand"))
    with pytest.raises(ValueError, match="rectangular"):
        to_arrays(panel.head(panel.height - 1))


def test_lags_and_rolling_values(pa):
    f = history_features(pa, LAGS, WINDOWS)
    y = pa.m["demand_hist"]
    t = 40
    assert f["lag_1"][0, t] == pytest.approx(y[0, t])
    assert f["lag_7"][0, t] == pytest.approx(y[0, t - 6])
    assert f["rmean_7"][0, t] == pytest.approx(y[0, t - 6: t + 1].mean(), rel=1e-5)
    assert f["rstd_3"][0, t] == pytest.approx(y[0, t - 2: t + 1].std(), rel=1e-4)
    assert np.isnan(f["rmean_28"][0, 26]) and np.isfinite(f["rmean_28"][0, 27])


def test_seasonal_lag_and_target_alignment(pa):
    df = build_rows(pa, [40], 3, LAGS, WINDOWS, KNOWN)
    y = pa.m["demand"]
    r = df.filter(pl.col("sid") == 1).row(0, named=True)
    assert r["target_idx"] == 43 and r["y"] == pytest.approx(y[1, 43])
    assert r["seasonal_lag_1w"] == pytest.approx(y[1, 36])   # 43 - 7 <= origin 40
    assert r["target_dow"] == pa.dates[43].weekday()
    assert r["target_discount"] == pytest.approx(pa.m["discount"][1, 43])


@pytest.mark.parametrize("h", [1, 2, 7])
def test_no_future_leakage(pa, h):
    """Perturbing every value after the origin must not change any history feature."""
    origin = 45
    base = build_rows(pa, [origin], h, LAGS, WINDOWS, KNOWN)
    m2 = {k: v.copy() for k, v in pa.m.items()}
    rng = np.random.default_rng(0)
    for _k, v in m2.items():
        v[:, origin + 1:] = rng.uniform(0, 50, v[:, origin + 1:].shape)
    from freshguard.features.panel import PanelArrays

    pert = PanelArrays(pa.keys, pa.dates, m2)
    other = build_rows(pert, [origin], h, LAGS, WINDOWS, KNOWN)
    allowed_to_change = {"y", "y_sales", "y_censored", *[f"target_{k}" for k in KNOWN],
                         "target_discount_rel"}
    for col in base.columns:
        if col in allowed_to_change:
            continue
        a, b = base[col].to_numpy(), other[col].to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"feature {col} leaks future data"
