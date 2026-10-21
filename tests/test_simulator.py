"""Inventory simulator: conservation, expiry, lead time, FEFO and constraints."""

from __future__ import annotations

import numpy as np
import pytest

from freshguard.inventory.simulator import SimSettings, simulate


def _run(demand, targets, **kw):
    fields = SimSettings.__dataclass_fields__
    settings = SimSettings(**{k: kw.pop(k) for k in list(kw) if k in fields})
    return simulate(np.asarray(demand, float), np.asarray(targets, float), settings, **kw)


def test_inventory_balance_and_nonnegativity(rng):
    s, d = 50, 40
    demand = rng.gamma(1.5, 2.0, (s, d))
    targets = rng.uniform(0, 8, (s, d))
    res = _run(demand, targets, lead_time=2, shelf_life=3,
               mid_forecasts=rng.uniform(0, 4, (s, d, 1)), initial_on_hand=np.full(s, 3.0),
               initial_pipeline=np.full((s, 1), 2.0))
    for arr in (res.opening, res.sold, res.lost, res.expired, res.closing, res.ordered):
        assert np.all(arr >= -1e-12)
    # Day-level balance: closing = opening - sold - expired; opening = prev closing + received.
    np.testing.assert_allclose(res.closing, res.opening - res.sold - res.expired, atol=1e-9)
    np.testing.assert_allclose(res.opening[:, 1:], res.closing[:, :-1] + res.received[:, 1:],
                               atol=1e-9)
    # Sales never exceed availability nor demand; lost = demand - sold.
    assert np.all(res.sold <= res.opening + 1e-9)
    np.testing.assert_allclose(res.sold + res.lost, demand, atol=1e-9)
    # Global conservation: initial + all receipts = sold + expired + final closing.
    total_in = 3.0 + res.received.sum(1)
    np.testing.assert_allclose(total_in, res.sold.sum(1) + res.expired.sum(1) + res.closing[:, -1],
                               atol=1e-6)


def test_shelf_life_one_day_everything_unsold_expires():
    demand = np.array([[2.0, 2.0, 2.0, 2.0]])
    targets = np.full((1, 4), 5.0)
    res = _run(demand, targets, lead_time=1, shelf_life=1, initial_on_hand=np.array([5.0]))
    np.testing.assert_allclose(res.sold[0], [2, 2, 2, 2])
    np.testing.assert_allclose(res.expired[0], [3, 3, 3, 3])
    np.testing.assert_allclose(res.closing[0], 0.0)


def test_expiry_after_exact_shelf_life_without_demand():
    # 10 units arrive on day 0 with shelf life 3: sellable on days 0, 1, 2 and expire at close
    # of day 2 when nothing sells.
    demand = np.zeros((1, 6))
    targets = np.zeros((1, 6))
    res = _run(demand, targets, lead_time=1, shelf_life=3, initial_on_hand=np.array([10.0]))
    np.testing.assert_allclose(res.expired[0], [0, 0, 10, 0, 0, 0])
    np.testing.assert_allclose(res.closing[0], [10, 10, 0, 0, 0, 0])


def test_fefo_sells_oldest_first():
    # Day 0: 4 old units. Order 4 more arriving day 1. Demand 3 on day 1 must come from the
    # old batch, so 1 old unit expires at close of day 1 when shelf life is 2.
    demand = np.array([[0.0, 3.0, 0.0, 0.0]])
    targets = np.array([[8.0, 0.0, 0.0, 0.0]])
    res = _run(demand, targets, lead_time=1, shelf_life=2, initial_on_hand=np.array([4.0]))
    assert res.received[0, 1] == pytest.approx(4.0)
    assert res.expired[0, 1] == pytest.approx(1.0)
    assert res.expired[0, 2] == pytest.approx(4.0)


@pytest.mark.parametrize("lead", [1, 2, 3])
def test_orders_arrive_exactly_after_lead_time(lead):
    d = 10
    demand = np.zeros((1, d))
    targets = np.zeros((1, d))
    targets[0, 2] = 7.0  # order placed at close of day 2
    mid = np.zeros((1, d, lead - 1)) if lead > 1 else None
    res = _run(demand, targets, lead_time=lead, shelf_life=5, mid_forecasts=mid)
    assert res.ordered[0, 2] == pytest.approx(7.0)
    expected = np.zeros(d)
    expected[2 + lead] = 7.0
    np.testing.assert_allclose(res.received[0], expected)


def test_order_up_to_accounts_for_pipeline_and_projected_expiry():
    # L=2: at close of day 0 we hold 5 units with 1 day of life left (expire close of day 1)
    # and forecast 2 units of demand on day 1. Projected usable stock at the arrival day (2)
    # is 0, so the order equals the full target of 6.
    demand = np.zeros((1, 3))
    targets = np.array([[6.0, 0.0, 0.0]])
    res = simulate(demand, targets, SimSettings(lead_time=2, shelf_life=2),
                   mid_forecasts=np.full((1, 3, 1), 2.0), initial_on_hand=np.array([5.0]))
    assert res.ordered[0, 0] == pytest.approx(6.0)


def test_pack_size_and_capacity_constraints():
    demand = np.zeros((2, 3))
    targets = np.array([[7.3, 0, 0], [50.0, 0, 0]])
    res = _run(demand, targets, lead_time=1, shelf_life=3, pack_size=2.0,
               capacity=np.array([100.0, 10.0]))
    assert res.ordered[0, 0] == pytest.approx(8.0)          # rounded up to whole packs
    assert res.ordered[1, 0] == pytest.approx(10.0)         # capped
    assert np.allclose(np.mod(res.ordered, 2.0), 0.0)


def test_simulation_is_deterministic(rng):
    demand = rng.poisson(3, (20, 15)).astype(float)
    targets = rng.uniform(0, 6, (20, 15))
    a = _run(demand, targets, lead_time=1, shelf_life=2)
    b = _run(demand, targets, lead_time=1, shelf_life=2)
    for k in a.__dict__:
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))


def test_stockout_event_definition():
    demand = np.array([[5.0, 0.0, 1.0]])
    targets = np.zeros((1, 3))
    res = _run(demand, targets, lead_time=1, shelf_life=2, initial_on_hand=np.array([3.0]),
               stockout_tolerance=0.0)
    np.testing.assert_array_equal(res.stockout[0], [True, False, True])


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        SimSettings(lead_time=0)
    with pytest.raises(ValueError):
        SimSettings(shelf_life=0)
    with pytest.raises(ValueError):
        simulate(np.array([[-1.0]]), np.array([[1.0]]), SimSettings())
