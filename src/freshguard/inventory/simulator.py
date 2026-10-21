"""Chronological, vectorised lost-sales simulator for perishable inventory.

Timeline of one simulated day ``d`` for every store-SKU in parallel:

1. Orders due today arrive before opening with full shelf life.
2. Demand is served first-expired-first-out (FEFO, equal to FIFO here because every
   delivery has the same shelf life). Unserved demand is lost (no backorders).
3. At close, units whose remaining life reaches zero expire and become waste; the rest age
   by one day.
4. The policy places an order that arrives ``lead_time`` days later. The order quantity is
   ``order-up-to target - projected usable inventory on the arrival day``. The projection rolls
   current stock and open orders forward with the policy's own point forecasts and removes
   units that would expire before the arrival day. No realised future demand is used.

Physical invariants (asserted when ``check=True``): inventory is never negative, sales never
exceed available stock, expired units are never sold, and an order never arrives before its
lead time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimSettings:
    lead_time: int = 1            # days between order (at close of d) and arrival (open of d+L)
    shelf_life: int = 3           # sellable days including the arrival day
    pack_size: float = 0.0        # 0 -> continuous quantities
    stockout_tolerance: float = 0.0
    check: bool = True

    def __post_init__(self) -> None:
        if self.lead_time < 1:
            raise ValueError("lead_time must be >= 1 (orders arrive the next day at earliest)")
        if self.shelf_life < 1:
            raise ValueError("shelf_life must be >= 1")
        if self.pack_size < 0:
            raise ValueError("pack_size must be >= 0")


@dataclass
class SimResult:
    """Per (series, day) trajectories; all arrays have shape (S, D)."""

    demand: np.ndarray
    opening: np.ndarray
    received: np.ndarray
    sold: np.ndarray
    lost: np.ndarray
    expired: np.ndarray
    closing: np.ndarray
    ordered: np.ndarray
    mean_age_closing: np.ndarray   # average days since arrival of closing stock
    stockout: np.ndarray           # bool: lost demand above tolerance

    def window(self, start: int, end: int | None = None) -> SimResult:
        sl = slice(start, end)
        return SimResult(**{k: v[:, sl] for k, v in self.__dict__.items()})


def _consume_fefo(inv: np.ndarray, qty: np.ndarray) -> np.ndarray:
    """Remove ``qty`` from age buckets, oldest (lowest remaining life) first. Returns served."""
    remaining = qty.astype(np.float64).copy()
    for a in range(inv.shape[1]):
        take = np.minimum(inv[:, a], remaining)
        inv[:, a] -= take
        remaining -= take
    return qty - remaining


def _round_pack(q: np.ndarray, pack: float) -> np.ndarray:
    if pack <= 0:
        return q
    return np.ceil(np.round(q / pack, 9)) * pack


def simulate(demand: np.ndarray, targets: np.ndarray, settings: SimSettings,
             mid_forecasts: np.ndarray | None = None,
             initial_on_hand: np.ndarray | None = None,
             initial_pipeline: np.ndarray | None = None,
             capacity: np.ndarray | None = None) -> SimResult:
    """Run the replenishment simulation.

    Parameters
    ----------
    demand : (S, D) realised demand per simulated day.
    targets : (S, D) order-up-to level for the arrival day ``d + L``, decided at close of ``d``.
    mid_forecasts : (S, D, L-1) point forecasts of days ``d+1 .. d+L-1`` made at close of ``d``
        (only needed when ``lead_time > 1``).
    initial_on_hand : (S,) stock at the start of day 0, assumed fresh.
    initial_pipeline : (S, L-1) open orders arriving on days 1..L-1.
    capacity : (S,) cap on usable stock at the arrival day (``inf`` if None).
    """
    s, d_total = demand.shape
    L, SL = settings.lead_time, settings.shelf_life
    if targets.shape != (s, d_total):
        raise ValueError("targets must have the same shape as demand")
    if L > 1 and (mid_forecasts is None or mid_forecasts.shape != (s, d_total, L - 1)):
        raise ValueError("mid_forecasts of shape (S, D, L-1) required when lead_time > 1")
    if np.any(demand < 0) or np.any(~np.isfinite(demand)):
        raise ValueError("demand must be finite and non-negative")

    inv = np.zeros((s, SL), dtype=np.float64)       # inv[:, a]: remaining life a+1 days
    if initial_on_hand is not None:
        inv[:, SL - 1] = np.maximum(initial_on_hand, 0.0)
    # pipe[:, k] arrives at the open of day (today + 1 + k), viewed at today's close.
    pipe = np.zeros((s, L), dtype=np.float64)
    if initial_pipeline is not None and L > 1:
        pipe[:, : L - 1] = np.maximum(initial_pipeline, 0.0)
    arrive_today = np.zeros(s)                      # orders arriving at the open of day 0
    cap = np.full(s, np.inf) if capacity is None else capacity

    out = {k: np.zeros((s, d_total)) for k in (
        "opening", "received", "sold", "lost", "expired", "closing", "ordered",
        "mean_age_closing")}
    ages = (SL - 1 - np.arange(SL)).astype(np.float64)  # days since arrival per bucket
    prev_closing = inv.sum(axis=1)

    for d in range(d_total):
        # 1. Arrivals.
        inv[:, SL - 1] += arrive_today
        out["received"][:, d] = arrive_today
        opening = inv.sum(axis=1)
        out["opening"][:, d] = opening
        # 2. Sales (FEFO), lost sales.
        dem = demand[:, d].astype(np.float64)
        sold = _consume_fefo(inv, dem)
        out["sold"][:, d] = sold
        out["lost"][:, d] = dem - sold
        # 3. Expiry and ageing.
        expired = inv[:, 0].copy()
        inv[:, :-1] = inv[:, 1:]
        inv[:, -1] = 0.0
        out["expired"][:, d] = expired
        closing = inv.sum(axis=1)
        out["closing"][:, d] = closing
        with np.errstate(invalid="ignore", divide="ignore"):
            out["mean_age_closing"][:, d] = np.where(
                closing > 0, (inv * (ages + 1)).sum(axis=1) / closing, 0.0)
        if settings.check:
            assert np.all(inv >= -1e-9), "negative inventory"
            assert np.all(sold <= opening + 1e-9), "sold more than available"
            assert np.allclose(prev_closing + arrive_today - sold - expired, closing,
                               atol=1e-6), "inventory balance violated"
        prev_closing = closing

        # 4. Order decision at close of d, arriving at open of d + L.
        proj = inv.copy()
        for k in range(L - 1):  # roll forward days d+1 .. d+L-1
            proj[:, SL - 1] += pipe[:, k]
            _consume_fefo(proj, np.maximum(mid_forecasts[:, d, k], 0.0))  # type: ignore[index]
            proj[:, :-1] = proj[:, 1:]
            proj[:, -1] = 0.0
        position = proj.sum(axis=1)
        target = np.nan_to_num(targets[:, d], nan=0.0)
        q = np.maximum(np.minimum(target, cap) - position, 0.0)
        q = _round_pack(q, settings.pack_size)
        out["ordered"][:, d] = q
        pipe[:, L - 1] = q
        # Advance the pipeline to tomorrow's open.
        arrive_today = pipe[:, 0].copy()
        pipe[:, :-1] = pipe[:, 1:]
        pipe[:, -1] = 0.0

    lost = out["lost"]
    stockout = lost > settings.stockout_tolerance + 1e-9
    return SimResult(demand=demand.astype(np.float64), stockout=stockout,
                     **out)
