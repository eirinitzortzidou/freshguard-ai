"""Policies, newsvendor ratio, financial accounting and currency conversion."""

from __future__ import annotations

import numpy as np
import pytest

from freshguard.finance.economics import (
    Economics,
    bootstrap_compare,
    compare,
    convert_currency,
    pct_reduction,
    series_ledger,
    summarize,
)
from freshguard.forecasting.intervals import ResidualQuantiles
from freshguard.inventory.policies import (
    PolicyInputs,
    PolicySpec,
    build_targets,
    critical_fractile,
)
from freshguard.inventory.simulator import SimSettings, simulate

ECON = Economics(gross_margin=0.3, disposal_cost=0.1, holding_rate_annual=0.365,
                 goodwill_share_of_margin=0.5, order_line_cost=0.02)


def test_critical_fractile():
    # Cu = 2.5 - 1.75 = 0.75, Co = 1.75 + 0.1 -> 0.75 / 2.6
    assert critical_fractile(2.5, 1.75, 0.1) == pytest.approx(0.75 / 2.6)
    with pytest.raises(ValueError):
        critical_fractile(1.0, 2.0, 0.0)


def _inputs(rng, s=5, t=40, L=1):
    sales = rng.poisson(4, (s, t)).astype(float)
    fc = {h: np.full((s, t), 4.0) for h in (1, 2)}
    rq = ResidualQuantiles().fit(rng.uniform(0, 8, 5000), rng.uniform(0, 8, 5000))
    return PolicyInputs(sales, sales, fc, {1: rq, 2: rq})


def test_traditional_targets_use_past_sales_only(rng):
    inp = _inputs(rng)
    days = np.arange(10, 20)
    tgt, _ = build_targets(PolicySpec("b", "traditional", "ma7_sales", 0.2), inp, days, 1)
    expected = 1.2 * np.array([inp.sales[:, d - 6: d + 1].mean(1) for d in days]).T
    np.testing.assert_allclose(tgt, expected)


def test_forecast_targets_increase_with_service_level(rng):
    inp = _inputs(rng)
    days = np.arange(10, 20)
    lo, _ = build_targets(PolicySpec("q", "forecast", "quantile", 0.5), inp, days, 1)
    hi, mid = build_targets(PolicySpec("q", "forecast", "quantile", 0.9), inp, days, 2)
    assert np.all(hi >= lo) and mid.shape == (5, 10, 1)


def test_ledger_by_hand():
    demand = np.array([[3.0, 1.0]])
    res = simulate(demand, np.array([[0.0, 0.0]]), SimSettings(shelf_life=1),
                   initial_on_hand=np.array([2.0]))
    led = series_ledger(res, np.array([[2.0, 2.0]]), ECON)
    # Day 0: sell 2, lose 1; day 1: nothing on hand, lose 1.
    assert led["sold_units"][0] == 2 and led["lost_units"][0] == 2
    assert led["revenue"][0] == pytest.approx(4.0)
    assert led["cogs"][0] == pytest.approx(2 * 1.4)
    assert led["gross_profit"][0] == pytest.approx(4.0 - 2.8)
    assert led["goodwill"][0] == pytest.approx(0.5 * 0.6 * 2)
    assert led["stockout_events"][0] == 2


def test_compare_decomposition_sums_to_net_benefit(rng):
    s, t = 30, 20
    demand = rng.gamma(2, 2, (s, t))
    price = rng.uniform(1, 4, (s, t))
    r1 = simulate(demand, rng.uniform(0, 10, (s, t)), SimSettings(shelf_life=2))
    r2 = simulate(demand, rng.uniform(0, 10, (s, t)), SimSettings(shelf_life=2))
    a, b = summarize(series_ledger(r1, price, ECON)), summarize(series_ledger(r2, price, ECON))
    d = compare(a, b)
    parts = (d["recovered_sales_margin_eur"] + d["avoided_waste_cost_eur"]
             + d["avoided_disposal_eur"] + d["holding_cost_change_eur"]
             + d["ordering_cost_change_eur"])
    assert parts == pytest.approx(d["net_benefit_eur"], rel=1e-9, abs=1e-9)
    assert d["waste_reduction_pct"] == pytest.approx(
        (a["waste_units"] - b["waste_units"]) / a["waste_units"] * 100)


def test_bootstrap_identical_policies_zero_effect(rng):
    s, t = 40, 10
    demand = rng.gamma(2, 2, (s, t))
    r = simulate(demand, rng.uniform(0, 10, (s, t)), SimSettings(shelf_life=2))
    led = series_ledger(r, np.full((s, t), 2.0), ECON)
    ci = bootstrap_compare(led, led, n_boot=50, seed=1)
    assert ci["net_benefit_eur"]["ci_low"] == pytest.approx(0.0)
    assert ci["net_benefit_eur"]["ci_high"] == pytest.approx(0.0)


def test_pct_reduction_and_currency():
    assert pct_reduction(200, 150) == pytest.approx(25.0)
    assert np.isnan(pct_reduction(0, 5))
    assert convert_currency(100.0, 0.9034) == pytest.approx(90.34)
    assert convert_currency(5.0, None) == 5.0
    with pytest.raises(ValueError):
        convert_currency(1.0, -1.0)
    with pytest.raises(ValueError):
        Economics(gross_margin=1.2, disposal_cost=0, holding_rate_annual=0,
                  goodwill_share_of_margin=0, order_line_cost=0)
