"""Leakage and correctness checks of the nested evaluation and rolling recalibration."""

from __future__ import annotations

import numpy as np
import pytest

from freshguard.evaluation.nested import by_target, calibrators, make_plan, rolling_tables
from freshguard.forecasting.intervals import ResidualQuantiles, RollingQuantiles
from freshguard.inventory.policies import PolicyInputs, PolicySpec, build_targets


def _tables(rng, n_series=40, n_days=60):
    pred = rng.uniform(0.5, 5, (n_series, n_days))
    y = np.maximum(pred + rng.normal(0, 1, pred.shape), 0)
    return pred, y


def test_rolling_quantiles_selects_table_by_target_day():
    lo = ResidualQuantiles().fit(np.linspace(0, 1, 500), np.linspace(0, 1, 500) - 1.0)
    hi = ResidualQuantiles().fit(np.linspace(0, 1, 500), np.linspace(0, 1, 500) + 1.0)
    rq = RollingQuantiles()
    rq.add(10, lo)
    rq.add(20, hi)
    pred = np.full((2, 4), 0.5)
    q = rq.quantile_at(pred, np.array([5, 15, 20, 25]), 0.5)
    np.testing.assert_allclose(q[:, 0], q[:, 1])          # before first start -> first table
    assert np.all(q[:, 2] > q[:, 1])
    with pytest.raises(ValueError):
        rq.add(15, lo)


def test_rolling_tables_use_only_known_outcomes(rng):
    """Changing outcomes on or after the forecast day must not change that day's table."""
    pred, y = _tables(rng)
    starts, h = [20, 27, 34, 41], 1
    a = rolling_tables(pred, y, starts, h, calib_days=14, min_days=7, first_target=10)
    for k, s in enumerate(starts):
        y2 = y.copy()
        y2[:, s - h + 1:] += 50.0                           # perturb unknown outcomes
        b = rolling_tables(pred, y2, starts, h, calib_days=14, min_days=7, first_target=10)
        np.testing.assert_array_equal(a.tables[k].table, b.tables[k].table)
        assert a.starts[k] == s


def test_rolling_tables_respect_horizon_lag(rng):
    pred, y = _tables(rng)
    h = 3
    a = rolling_tables(pred, y, [30], h, 14, 7, 0)
    y2 = y.copy()
    y2[:, 30 - h + 1:] += 50.0
    b = rolling_tables(pred, y2, [30], h, 14, 7, 0)
    np.testing.assert_array_equal(a.tables[0].table, b.tables[0].table)
    y3 = y.copy()
    y3[:, 30 - h] += 50.0                                   # last known outcome does matter
    c = rolling_tables(pred, y3, [30], h, 14, 7, 0)
    assert not np.array_equal(a.tables[0].table, c.tables[0].table)


def test_static_calibrator_is_frozen_after_first_full_window(rng):
    from freshguard.evaluation.nested import NestedPlan

    pred, y = _tables(rng, n_days=70)
    plan = NestedPlan(horizon=1, weeks=[10, 17, 24, 31, 38], folds=[31, 38], fold_days=7,
                      tune_days=14, calib_days=14, min_calib_days=7, es_days=7, warmup=7,
                      primary="insample")
    ins = {w: (pred[:, w - 7:w].ravel(), y[:, w - 7:w].ravel()) for w in plan.weeks}
    cal = calibrators(pred, y, plan, ins)
    st, ro = cal["static"], cal["rolling"]
    assert st.starts == ro.starts == [17, 24, 31, 38]
    assert st.tables[1] is st.tables[2] is st.tables[3]    # frozen from the first full table
    assert ro.tables[2] is not ro.tables[1]
    assert cal["insample"].starts == plan.weeks


def test_by_target_shift():
    fc = np.arange(12, dtype=float).reshape(2, 6)
    t = by_target(fc, 2)
    assert np.isnan(t[:, :2]).all()
    np.testing.assert_array_equal(t[:, 2:], fc[:, :4])


def test_policy_targets_with_rolling_quantiles(rng):
    s, t = 5, 30
    fc = rng.uniform(1, 3, (s, t))
    tab = ResidualQuantiles().fit(rng.uniform(0, 4, 2000), rng.uniform(0, 4, 2000))
    rq = RollingQuantiles([0], [tab])
    inp = PolicyInputs(sales=fc, demand_hist=fc, forecasts={1: fc}, residuals={1: rq})
    days = np.arange(10, 20)
    tgt, _ = build_targets(PolicySpec("x", "forecast", "quantile", 0.8), inp, days, 1)
    np.testing.assert_allclose(tgt, tab.quantile(fc[:, days], 0.8))


def test_make_plan_validates_dates():
    from freshguard.config import PROJECT_ROOT, load_config
    from freshguard.data.synthetic import make_synthetic_frn
    from freshguard.features.panel import to_arrays
    from freshguard.splits import resolve_splits

    cfg = load_config(PROJECT_ROOT / "configs" / "synthetic.yaml")
    raw = make_synthetic_frn(n_stores=1, n_products=2, n_days=97, seed=1)
    from freshguard.data.demand import reconstruct_demand
    from freshguard.data.ingest import build_panel

    panel, hs, hk = build_panel(raw)
    panel = reconstruct_demand(panel, hs, hk, cfg["splits"]["train_end"])
    import polars as pl

    panel = panel.with_columns(pl.col("demand").alias("demand_hist"),
                               pl.col("censored").alias("censored_truth"))
    pa = to_arrays(panel)
    sp = resolve_splits(cfg, pa)
    plan = make_plan(cfg, pa, sp)
    assert plan.folds[-1] + plan.fold_days - 1 < pa.n_days
    assert all(f in plan.weeks for f in plan.folds)
    bad = cfg.with_overrides({"nested": {"folds": ["2024-06-20"]}})
    with pytest.raises(ValueError):
        make_plan(bad, pa, sp)
    early = cfg.with_overrides({"nested": {"folds": ["2024-05-29"]}})
    with pytest.raises(ValueError):
        make_plan(early, pa, sp)
