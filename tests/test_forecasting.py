"""Baseline forecasters, metrics, predictive quantiles and the GBM wrapper."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from freshguard.evaluation.metrics import bias, grouped_metrics, mae, pinball, rmse, wape
from freshguard.forecasting.baselines import croston_sba, forecast_matrix, ses_level, tsb
from freshguard.forecasting.gbm import GBMForecaster
from freshguard.forecasting.intervals import ResidualQuantiles


def test_metrics_and_zero_denominator():
    y = np.array([1.0, 3.0, 0.0])
    p = np.array([2.0, 1.0, 0.0])
    assert mae(y, p) == pytest.approx(1.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert wape(y, p) == pytest.approx(3 / 4)
    assert bias(y, p) == pytest.approx(-1 / 4)
    assert np.isnan(wape(np.zeros(3), np.ones(3)))
    assert pinball(np.array([2.0]), np.array([1.0]), 0.9) == pytest.approx(0.9)


def test_grouped_metrics_zero_safe():
    df = pl.DataFrame({"g": [1, 1, 2], "y": [1.0, 1.0, 0.0], "pred": [2.0, 1.0, 1.0]})
    out = grouped_metrics(df, ["g"])
    assert out.filter(pl.col("g") == 1)["wape"][0] == pytest.approx(0.5)
    assert out.filter(pl.col("g") == 2)["wape"][0] is None


def test_forecast_matrix_alignment_uses_only_past():
    y = np.arange(30, dtype=float)[None, :]
    naive = forecast_matrix(y, "naive", 2)
    assert naive[0, 10] == 8.0                    # forecast of day 10 made at day 8
    sn = forecast_matrix(y, "seasonal_naive", 3)
    assert sn[0, 20] == 13.0                      # same weekday, one week earlier
    sn8 = forecast_matrix(y, "seasonal_naive", 8)
    assert sn8[0, 20] == 6.0                      # needs two weeks back when h > 7
    ma = forecast_matrix(y, "ma7", 1)
    assert ma[0, 10] == pytest.approx(np.mean(np.arange(3, 10)))
    for meth in ("ma28", "ses", "croston_sba", "tsb", "dow_mean_4w", "ma28_dow_index"):
        f = forecast_matrix(y, meth, 1)
        y2 = y.copy()
        y2[0, 15:] = 999.0
        g = forecast_matrix(y2, meth, 1)
        np.testing.assert_allclose(f[0, :16], g[0, :16], equal_nan=True)


def test_intermittent_methods():
    y = np.array([[0, 0, 4, 0, 0, 4, 0, 0, 4.0]])
    c = croston_sba(y, alpha=0.1)
    assert c[0, -1] == pytest.approx((1 - 0.05) * 4 / 3, rel=1e-6)
    t = tsb(y, 0.1, 0.1)
    assert np.all(t >= 0)
    assert ses_level(np.array([[2.0, 2.0, 2.0]]), 0.3)[0, -1] == pytest.approx(2.0)


def test_residual_quantiles_coverage_and_monotonicity(rng):
    pred = rng.uniform(0, 10, 20000)
    actual = np.maximum(pred + rng.normal(0, 1 + 0.2 * pred), 0)
    rq = ResidualQuantiles().fit(pred, actual)
    for tau in (0.5, 0.8, 0.95):
        assert abs(rq.coverage(pred, actual, tau) - tau) < 0.02
    q = [rq.quantile(np.array([5.0]), t)[0] for t in (0.3, 0.6, 0.9)]
    assert q[0] <= q[1] <= q[2]
    assert np.all(rq.quantile(np.zeros(3), 0.05) >= 0)


def test_gbm_nonnegative_deterministic_and_shape(rng):
    n = 3000
    x = rng.uniform(0, 5, n)
    df = pl.DataFrame({"f1": x, "f2": rng.normal(size=n), "store_id": rng.integers(0, 3, n),
                       "y": np.maximum(2 * x + rng.normal(size=n), 0)})
    params = {"objective": "tweedie", "tweedie_variance_power": 1.2, "learning_rate": 0.1,
              "num_leaves": 15, "min_data_in_leaf": 20, "verbose": -1, "num_boost_round": 50,
              "num_threads": 1}
    p1 = GBMForecaster(params, seed=1).fit(df).predict(df)
    p2 = GBMForecaster(params, seed=1).fit(df).predict(df)
    assert p1.shape == (n,)
    assert np.all(p1 >= 0)
    np.testing.assert_array_equal(p1, p2)
