"""Synthetic FreshRetailNet-shaped fixture for tests, CI and smoke runs.

The generator draws a *known* latent hourly demand, simulates intraday stockouts and censors
sales accordingly. It is never used for headline results: it exists so that schema checks,
demand reconstruction, leakage tests and the end-to-end pipeline can run without the real
dataset.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from freshguard.data.schema import HOURS


def make_synthetic_frn(n_stores: int = 6, n_products: int = 8, n_days: int = 97,
                       start: str = "2024-03-28", seed: int = 7,
                       return_truth: bool = False) -> pl.DataFrame:
    """Generate a FreshRetailNet-50K-like raw table.

    With ``return_truth=True`` an extra ``true_demand`` column holds the uncensored daily
    demand, which tests use to verify the reconstruction logic.
    """
    rng = np.random.default_rng(seed)
    start_date = dt.date.fromisoformat(start)
    dates = [start_date + dt.timedelta(days=i) for i in range(n_days)]
    dow = np.array([d.weekday() for d in dates])
    holiday = np.zeros(n_days, dtype=np.int64)
    holiday[[i for i in (30, 31, 32, 60) if i < n_days]] = 1

    # Typical grocery intraday profile: morning and evening peaks, little at night.
    hours = np.arange(HOURS)
    profile = (np.exp(-0.5 * ((hours - 9.5) / 2.0) ** 2)
               + 1.4 * np.exp(-0.5 * ((hours - 18.5) / 2.2) ** 2) + 0.02)
    profile[(hours < 6) | (hours >= 22)] *= 0.15
    profile /= profile.sum()

    rows: list[dict[str, object]] = []
    for s in range(n_stores):
        city = s % 3
        store_mult = rng.uniform(0.6, 1.6)
        for p in range(n_products):
            cat1 = p % 4
            base = rng.lognormal(mean=0.0, sigma=0.9) * store_mult
            weekly = 1.0 + 0.25 * np.isin(dow, [5, 6]) + 0.05 * rng.standard_normal(7)[dow]
            trend = 1.0 + rng.normal(0, 0.002) * np.arange(n_days)
            discount = np.clip(1.0 - rng.binomial(1, 0.15, n_days) * rng.uniform(0.1, 0.4,
                                                                             n_days), 0.5, 1.0)
            activity = (discount < 0.95).astype(np.int64) * rng.binomial(1, 0.7, n_days)
            mean_daily = base * weekly * trend * (1.0 + 0.4 * holiday) * discount ** -1.5
            mean_daily *= 1.0 + 0.2 * activity
            # Restock policy that occasionally runs out in the afternoon.
            stock_frac = rng.uniform(0.7, 1.6, n_days)
            for d in range(n_days):
                hourly_mean = mean_daily[d] * profile
                demand_h = rng.gamma(shape=2.0, scale=hourly_mean / 2.0)
                demand_h = np.round(demand_h, 1)
                avail = stock_frac[d] * mean_daily[d]
                cum = np.cumsum(demand_h)
                status = np.zeros(HOURS, dtype=np.int64)
                sales_h = demand_h.copy()
                over = np.nonzero(cum > avail)[0]
                if over.size:
                    first = over[0]
                    sold_before = cum[first - 1] if first > 0 else 0.0
                    sales_h[first] = max(0.0, np.round(avail - sold_before, 1))
                    sales_h[first + 1:] = 0.0
                    status[first + 1:] = 1
                rows.append({
                    "city_id": city,
                    "store_id": s,
                    "management_group_id": cat1 // 2,
                    "first_category_id": cat1,
                    "second_category_id": cat1 * 10 + p % 2,
                    "third_category_id": cat1 * 100 + p,
                    "product_id": p,
                    "dt": dates[d].isoformat(),
                    "sale_amount": float(np.round(sales_h.sum(), 2)),
                    "hours_sale": sales_h.tolist(),
                    "stock_hour6_22_cnt": int(status[6:22].sum()),
                    "hours_stock_status": status.tolist(),
                    "discount": float(np.round(discount[d], 3)),
                    "holiday_flag": int(holiday[d]),
                    "activity_flag": int(activity[d]),
                    "precpt": float(rng.gamma(0.5, 4.0)),
                    "avg_temperature": float(20 + 8 * np.sin(d / 15) + rng.normal()),
                    "avg_humidity": float(rng.uniform(40, 90)),
                    "avg_wind_level": float(rng.uniform(0.5, 4.0)),
                    "true_demand": float(demand_h.sum()),
                })
    df = pl.DataFrame(rows)
    return df if return_truth else df.drop("true_demand")
