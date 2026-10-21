"""KPIs and financial accounting for simulation results.

Accounting identity (per store-SKU, summed over the KPI window)::

    gross_profit = revenue - cost of goods consumed - disposal - holding - ordering
    revenue                 = price * units sold
    cost of goods consumed  = unit_cost * (units sold + units expired)

Valuing consumed rather than purchased units neutralises differences in opening/closing
stock between policies. The difference in gross profit between two policies decomposes
exactly into recovered-sales margin, avoided waste (at cost), avoided disposal, holding and
ordering effects, so no benefit is counted twice. Goodwill (a non-cash penalty for lost
sales) is reported separately and never included in the cash ``net_benefit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from freshguard.inventory.simulator import SimResult


@dataclass(frozen=True)
class Economics:
    gross_margin: float
    disposal_cost: float
    holding_rate_annual: float
    goodwill_share_of_margin: float
    order_line_cost: float
    unit_price: float = 2.5     # used only when no observed price matrix is supplied

    @classmethod
    def from_config(cls, d: dict[str, Any]) -> Economics:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def __post_init__(self) -> None:
        if not 0 < self.gross_margin < 1:
            raise ValueError("gross_margin must be in (0, 1)")
        if min(self.disposal_cost, self.holding_rate_annual, self.goodwill_share_of_margin,
               self.order_line_cost) < 0:
            raise ValueError("cost parameters must be non-negative")


def convert_currency(amount: np.ndarray | float, rate: float | None) -> np.ndarray | float:
    """Convert to EUR with an explicit rate (EUR per unit of source currency)."""
    if rate is None:
        return amount
    if rate <= 0:
        raise ValueError("exchange rate must be positive")
    return amount * rate


def series_ledger(res: SimResult, price: np.ndarray, econ: Economics) -> dict[str, np.ndarray]:
    """Per-series (S,) totals of units and EUR over the result window."""
    price = np.broadcast_to(price, res.sold.shape).astype(np.float64)
    cost = price * (1.0 - econ.gross_margin)
    sold, lost, exp_ = res.sold, res.lost, res.expired
    led = {
        "demand_units": res.demand.sum(1),
        "sold_units": sold.sum(1),
        "lost_units": lost.sum(1),
        "waste_units": exp_.sum(1),
        "received_units": res.received.sum(1),
        "ordered_units": res.ordered.sum(1),
        "stockout_events": res.stockout.sum(1).astype(np.float64),
        "demand_days": (res.demand > 0).sum(1).astype(np.float64),
        "days": np.full(sold.shape[0], sold.shape[1], dtype=np.float64),
        "closing_units_sum": res.closing.sum(1),
        "order_lines": (res.ordered > 0).sum(1).astype(np.float64),
        "revenue": (price * sold).sum(1),
        "cogs": (cost * (sold + exp_)).sum(1),
        "waste_cost": (cost * exp_).sum(1),
        "disposal": econ.disposal_cost * exp_.sum(1),
        "holding": (econ.holding_rate_annual / 365.0 * cost * res.closing).sum(1),
        "lost_sales_value": (price * lost).sum(1),
        "lost_margin": ((price - cost) * lost).sum(1),
        "goodwill": (econ.goodwill_share_of_margin * (price - cost) * lost).sum(1),
        "inventory_value_sum": (cost * res.closing).sum(1),
    }
    led["ordering"] = econ.order_line_cost * led["order_lines"]
    led["gross_profit"] = (led["revenue"] - led["cogs"] - led["disposal"] - led["holding"]
                           - led["ordering"])
    led["economic_profit"] = led["gross_profit"] - led["goodwill"]
    return led


def summarize(led: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, float]:
    """Aggregate a ledger (optionally a subset of series) into headline KPIs."""
    m = np.ones_like(led["sold_units"], dtype=bool) if mask is None else mask
    t = {k: float(v[m].sum()) for k, v in led.items()}
    days = t["days"]
    out = dict(t)
    out["fill_rate"] = t["sold_units"] / t["demand_units"] if t["demand_units"] else np.nan
    out["waste_rate"] = t["waste_units"] / t["received_units"] if t["received_units"] else 0.0
    out["availability"] = 1.0 - t["stockout_events"] / days if days else np.nan
    out["stockout_rate_demand_days"] = (t["stockout_events"] / t["demand_days"]
                                        if t["demand_days"] else np.nan)
    out["avg_inventory_units"] = t["closing_units_sum"] / days if days else np.nan
    out["avg_inventory_value"] = t["inventory_value_sum"] / days if days else np.nan
    n_days_window = float(np.max(led["days"][m])) if m.any() else 0.0
    avg_inv_val_total = t["inventory_value_sum"] / max(n_days_window, 1.0)
    cogs_sold = t["cogs"] - t["waste_cost"]
    out["inventory_turnover_period"] = cogs_sold / avg_inv_val_total if avg_inv_val_total \
        else np.nan
    out["days_of_supply"] = (t["closing_units_sum"] / t["sold_units"]
                             if t["sold_units"] else np.nan)
    out["gross_margin_pct"] = t["gross_profit"] / t["revenue"] if t["revenue"] else np.nan
    return out


def pct_reduction(baseline: float, candidate: float) -> float:
    """(baseline - candidate) / baseline * 100, NaN when the baseline is zero."""
    return float((baseline - candidate) / baseline * 100.0) if baseline else float("nan")


def compare(base: dict[str, float], cand: dict[str, float]) -> dict[str, float]:
    """Business-impact deltas of ``cand`` vs ``base`` with an exact profit decomposition."""
    d = {
        "waste_reduction_pct": pct_reduction(base["waste_units"], cand["waste_units"]),
        "stockout_reduction_pct": pct_reduction(base["stockout_events"],
                                                cand["stockout_events"]),
        "lost_sales_reduction_pct": pct_reduction(base["lost_units"], cand["lost_units"]),
        "fill_rate_change_pp": 100 * (cand["fill_rate"] - base["fill_rate"]),
        "availability_change_pp": 100 * (cand["availability"] - base["availability"]),
        "avg_inventory_change_pct": -pct_reduction(base["avg_inventory_units"],
                                                   cand["avg_inventory_units"]),
        "net_benefit_eur": cand["gross_profit"] - base["gross_profit"],
        "net_benefit_pct_of_baseline_profit": 100 * (cand["gross_profit"] - base["gross_profit"])
        / abs(base["gross_profit"]) if base["gross_profit"] else float("nan"),
        "lost_sales_value_change_eur": base["lost_sales_value"] - cand["lost_sales_value"],
        "economic_benefit_eur": cand["economic_profit"] - base["economic_profit"],
        # Decomposition (sums exactly to net_benefit_eur).
        "recovered_sales_margin_eur": (cand["revenue"] - base["revenue"])
        - ((cand["cogs"] - cand["waste_cost"]) - (base["cogs"] - base["waste_cost"])),
        "avoided_waste_cost_eur": base["waste_cost"] - cand["waste_cost"],
        "avoided_disposal_eur": base["disposal"] - cand["disposal"],
        "holding_cost_change_eur": base["holding"] - cand["holding"],
        "ordering_cost_change_eur": base["ordering"] - cand["ordering"],
        "goodwill_change_eur": base["goodwill"] - cand["goodwill"],
    }
    return d


def bootstrap_compare(led_base: dict[str, np.ndarray], led_cand: dict[str, np.ndarray],
                      n_boot: int = 500, seed: int = 0,
                      cluster: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """Paired bootstrap (series or cluster resampling) CIs of the key deltas."""
    rng = np.random.default_rng(seed)
    n = led_base["sold_units"].size
    groups = np.arange(n) if cluster is None else np.unique(cluster, return_inverse=True)[1]
    k = int(groups.max()) + 1
    keys = ["waste_units", "stockout_events", "gross_profit", "economic_profit",
            "sold_units", "demand_units", "lost_units"]
    gb = {key: np.bincount(groups, led_base[key], k) for key in keys}
    gc = {key: np.bincount(groups, led_cand[key], k) for key in keys}
    stats: dict[str, list[float]] = {"waste_reduction_pct": [], "stockout_reduction_pct": [],
                                     "net_benefit_eur": [], "fill_rate_change_pp": []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(np.float64)
        b = {key: float(w @ v) for key, v in gb.items()}
        c = {key: float(w @ v) for key, v in gc.items()}
        stats["waste_reduction_pct"].append(pct_reduction(b["waste_units"], c["waste_units"]))
        stats["stockout_reduction_pct"].append(pct_reduction(b["stockout_events"],
                                                             c["stockout_events"]))
        stats["net_benefit_eur"].append(c["gross_profit"] - b["gross_profit"])
        stats["fill_rate_change_pp"].append(
            100 * (c["sold_units"] / c["demand_units"] - b["sold_units"] / b["demand_units"]))
    return {key: {"ci_low": float(np.nanquantile(v, 0.025)),
                  "ci_high": float(np.nanquantile(v, 0.975))} for key, v in stats.items()}
