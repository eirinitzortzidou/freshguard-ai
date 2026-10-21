"""Policy tuning, out-of-sample replenishment simulation, financial impact and sensitivity.

Protocol
--------
1. *Tuning simulation* on the validation window: buffers of traditional policies and the
   service level (quantile) of forecast policies are chosen to maximise base-case gross profit.
   Forecasts used here come from the validation model only.
2. *Test simulation* on the test window with frozen choices. Every policy sees the same demand
   realisations, prices, costs, lead time, shelf life, capacity and initial state. A warm-up
   period before the test window is simulated and excluded from KPIs.
3. Paired comparisons, store-cluster bootstrap CIs, segment breakdowns, scenario and
   sensitivity analyses. Policies are re-tuned on validation for every sensitivity setting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl

from freshguard.artifacts import load_store, write_json
from freshguard.config import Config
from freshguard.features.panel import PanelArrays
from freshguard.finance.economics import (
    Economics,
    bootstrap_compare,
    compare,
    series_ledger,
    summarize,
)
from freshguard.forecasting.backtest import ForecastStore
from freshguard.inventory.policies import (
    PolicyInputs,
    PolicySpec,
    build_targets,
    critical_fractile,
)
from freshguard.inventory.simulator import SimResult, SimSettings, simulate
from freshguard.logging_utils import get_logger, kv, timed
from freshguard.splits import resolve_splits

log = get_logger(__name__)

BUFFER_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
TAU_GRID = (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.925, 0.95,
            0.975)
BASELINE = "traditional_ma7_tuned"
AI = "lgbm_cost_optimal"


@dataclass(frozen=True)
class SimContext:
    """Everything that defines one operating environment for the simulation."""

    demand: np.ndarray        # (S, T) realised demand used by the simulator
    price: np.ndarray         # (S, T) EUR shelf price
    capacity: np.ndarray      # (S,)
    settings: SimSettings
    econ: Economics
    warmup: int


def _initial_state(inp: PolicyInputs, day0: int, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Same starting stock for every policy: 7-day mean of (causal) demand before day0."""
    m = np.nan_to_num(inp.demand_hist[:, max(day0 - 7, 0): day0].mean(axis=1), nan=0.0)
    return m, np.repeat(m[:, None], max(L - 1, 0), axis=1)


def run_policy(spec: PolicySpec, inp: PolicyInputs, ctx: SimContext, start: int, end: int
               ) -> SimResult:
    """Simulate days ``start - warmup .. end`` and return the KPI window ``start .. end``."""
    L = ctx.settings.lead_time
    day0 = start - ctx.warmup
    days = np.arange(day0, end + 1)
    targets, mid = build_targets(spec, inp, days, L)
    on_hand, pipe = _initial_state(inp, day0, L)
    res = simulate(ctx.demand[:, days], targets, ctx.settings,
                   mid_forecasts=mid if L > 1 else None, initial_on_hand=on_hand,
                   initial_pipeline=pipe if L > 1 else None, capacity=ctx.capacity)
    return res.window(ctx.warmup)


def evaluate_policy(spec: PolicySpec, inp: PolicyInputs, ctx: SimContext, start: int,
                    end: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    res = run_policy(spec, inp, ctx, start, end)
    led = series_ledger(res, ctx.price[:, start:end + 1], ctx.econ)
    return led, summarize(led)


def candidate_policies(econ: Economics, price_mean: float) -> dict[str, list[PolicySpec]]:
    """Policy families with their tuning grids (one spec per grid point)."""
    cost = price_mean * (1 - econ.gross_margin)
    cf = critical_fractile(price_mean, cost, econ.disposal_cost,
                           goodwill=econ.goodwill_share_of_margin * (price_mean - cost))
    return {
        "traditional_naive_tuned": [PolicySpec("traditional_naive_tuned", "traditional",
                                               "naive_sales", b) for b in BUFFER_GRID],
        "traditional_seasonal_tuned": [PolicySpec("traditional_seasonal_tuned", "traditional",
                                                  "seasonal_sales", b) for b in BUFFER_GRID],
        BASELINE: [PolicySpec(BASELINE, "traditional", "ma7_sales", b) for b in BUFFER_GRID],
        "traditional_ma7_rule20": [PolicySpec("traditional_ma7_rule20", "traditional",
                                              "ma7_sales", 0.2)],
        "ma7_reconstructed_tuned": [PolicySpec("ma7_reconstructed_tuned", "traditional",
                                               "ma7_demand", b) for b in BUFFER_GRID],
        "lgbm_point": [PolicySpec("lgbm_point", "forecast", "point")],
        "lgbm_service95_normal": [PolicySpec("lgbm_service95_normal", "forecast", "normal",
                                             0.95)],
        "lgbm_newsvendor_cf": [PolicySpec("lgbm_newsvendor_cf", "forecast", "quantile",
                                          round(cf, 3))],
        AI: [PolicySpec(AI, "forecast", "quantile", t) for t in TAU_GRID],
    }


LABELS = {
    "traditional_naive_tuned": "Traditional: yesterday's sales + buffer",
    "traditional_seasonal_tuned": "Traditional: same weekday last week + buffer",
    BASELINE: "Traditional: 7-day sales average + tuned buffer (baseline)",
    "traditional_ma7_rule20": "Traditional: 7-day sales average + 20% rule",
    "ma7_reconstructed_tuned": "Ablation: 7-day reconstructed-demand average + buffer",
    "lgbm_point": "LightGBM point forecast (no safety stock)",
    "lgbm_service95_normal": "LightGBM order-up-to, 95% service (normal safety stock)",
    "lgbm_newsvendor_cf": "LightGBM newsvendor quantile (critical fractile)",
    AI: "LightGBM cost-optimised quantile (FreshGuard)",
}


def tune_and_test(inp: PolicyInputs, ctx: SimContext, tune: tuple[int, int],
                  test: tuple[int, int], families: dict[str, list[PolicySpec]],
                  keep_ledgers: bool = True) -> dict[str, Any]:
    """Pick each family's grid point on the tuning window, then evaluate it on test."""
    chosen: dict[str, PolicySpec] = {}
    tuning_curves: dict[str, list[dict[str, float]]] = {}
    for fam, specs in families.items():
        best: tuple[float, PolicySpec] | None = None
        curve = []
        for spec in specs:
            _, summ = evaluate_policy(spec, inp, ctx, *tune)
            curve.append({"param": spec.param, "gross_profit": summ["gross_profit"],
                          "waste_units": summ["waste_units"], "fill_rate": summ["fill_rate"],
                          "availability": summ["availability"]})
            if best is None or summ["gross_profit"] > best[0]:
                best = (summ["gross_profit"], spec)
        assert best is not None
        chosen[fam] = replace(best[1], label=LABELS.get(fam, fam))
        tuning_curves[fam] = curve
    results: dict[str, Any] = {"chosen": {k: v.param for k, v in chosen.items()},
                               "tuning_curves": tuning_curves, "test": {}, "ledgers": {}}
    for fam, spec in chosen.items():
        led, summ = evaluate_policy(spec, inp, ctx, *test)
        results["test"][fam] = summ
        if keep_ledgers:
            results["ledgers"][fam] = led
    return results


def pareto_frontier(inp: PolicyInputs, ctx: SimContext, test: tuple[int, int]
                    ) -> list[dict[str, Any]]:
    """Full grids on the *test* window for the trade-off chart (descriptive, not selection)."""
    rows = []
    for fam, kind, grid in (("LightGBM quantile", "quantile", TAU_GRID),
                            ("Traditional MA7 + buffer", "ma7_sales", BUFFER_GRID)):
        for g in grid:
            family = "forecast" if kind == "quantile" else "traditional"
            spec = PolicySpec(fam, family, kind, g)
            _, s = evaluate_policy(spec, inp, ctx, *test)
            rows.append({"policy": fam, "param": g, "waste_units": s["waste_units"],
                         "waste_rate": s["waste_rate"], "fill_rate": s["fill_rate"],
                         "availability": s["availability"],
                         "stockout_events": s["stockout_events"],
                         "gross_profit": s["gross_profit"],
                         "avg_inventory_units": s["avg_inventory_units"]})
    return rows


def iso_service(pareto: list[dict[str, Any]], ref: dict[str, float]) -> dict[str, Any]:
    """Descriptive frontier comparison: interpolate the LightGBM frontier at the reference
    policy's availability and fill rate (uses test-window grids; not a selection step)."""
    rows = sorted((r for r in pareto if r["policy"] == "LightGBM quantile"),
                  key=lambda r: r["availability"])
    out: dict[str, Any] = {}
    for metric in ("availability", "fill_rate"):
        xs = np.array([r[metric] for r in rows])
        order = np.argsort(xs)
        xs = xs[order]
        target = ref[metric]
        if not xs[0] <= target <= xs[-1]:
            out[metric] = None
            continue
        res = {"matched_" + metric: target}
        for k in ("waste_units", "stockout_events", "gross_profit", "avg_inventory_units"):
            ys = np.array([rows[i][k] for i in order])
            res["lgbm_" + k] = float(np.interp(target, xs, ys))
            res["baseline_" + k] = ref[k]
        res["waste_reduction_pct"] = 100 * (1 - res["lgbm_waste_units"] / ref["waste_units"])
        res["profit_change_eur"] = res["lgbm_gross_profit"] - ref["gross_profit"]
        out[metric] = res
    return out


def build_context(cfg: Config, pa: PanelArrays, scenario: str = "base",
                  shelf_life: int | None = None, lead_time: int | None = None,
                  demand_mode: str = "reconstructed", noise_sigma: float = 0.0,
                  train_end: int | None = None) -> SimContext:
    icfg, fcfg = cfg["inventory"], cfg["finance"]
    econ = Economics.from_config(fcfg["scenarios"][scenario])
    if fcfg.get("price_source") == "data" and "price" in pa.m:
        price = pa.m["price"].astype(np.float64) * float(fcfg["usd_to_eur"])
    else:
        price = np.full((pa.n_series, pa.n_days), econ.unit_price)
    sales = pa.m["sale_amount"].astype(np.float64)
    recon = pa.m["demand"].astype(np.float64)
    if demand_mode == "reconstructed":
        demand = recon
    elif demand_mode == "observed_sales":
        demand = sales
    elif demand_mode == "reconstructed_high":
        demand = sales + 1.5 * (recon - sales)
    else:
        raise ValueError(demand_mode)
    if noise_sigma > 0:
        rng = np.random.default_rng(int(cfg["seed"]))
        demand = demand * rng.lognormal(-0.5 * noise_sigma ** 2, noise_sigma, demand.shape)
    te = train_end if train_end is not None else pa.day_index(cfg["splits"]["train_end"])
    cap = float(icfg["capacity_multiplier"]) * np.maximum(
        pa.m["demand_hist"][:, : te + 1].max(axis=1), 1e-3)
    settings = SimSettings(lead_time=lead_time or int(icfg["lead_time_days"]),
                           shelf_life=shelf_life or int(icfg["shelf_life_days"]),
                           pack_size=float(icfg["pack_size"]),
                           stockout_tolerance=float(icfg["stockout_tolerance"]), check=False)
    return SimContext(demand, price, cap, settings, econ, int(icfg["warmup_days"]))


def _inputs(pa: PanelArrays, store: ForecastStore) -> PolicyInputs:
    return PolicyInputs(sales=pa.m["sale_amount"].astype(np.float64),
                        demand_hist=pa.m["demand_hist"].astype(np.float64),
                        forecasts={h: v.astype(np.float64) for h, v in store.fc.items()},
                        residuals=store.residuals)


def _headline(res: dict[str, Any]) -> dict[str, float]:
    return compare(res["test"][BASELINE], res["test"][AI])


def run_experiments(cfg: Config, pa: PanelArrays) -> dict[str, Any]:
    sp = resolve_splits(cfg, pa)
    store = load_store(cfg)
    inp = _inputs(pa, store)
    warm = int(cfg["inventory"]["warmup_days"])
    tune = (sp.valid.start + min(warm, 4), sp.valid.end)
    test = (sp.test_start, sp.test_end)
    out: dict[str, Any] = {"windows": {"tune": [str(pa.dates[tune[0]]), str(pa.dates[tune[1]])],
                                       "test": [str(pa.dates[test[0]]), str(pa.dates[test[1]])]},
                           "n_series": pa.n_series}
    n_boot = int(cfg["finance"]["bootstrap_samples"])
    stores = pa.keys["store_id"].to_numpy()

    # Base case.
    with timed(log, "simulate.base"):
        ctx = build_context(cfg, pa, "base")
        fams = candidate_policies(ctx.econ, float(np.mean(ctx.price)))
        base = tune_and_test(inp, ctx, tune, test, fams)
    out["base"] = {"settings": ctx.settings.__dict__, "economics": ctx.econ.__dict__,
                   "chosen_params": base["chosen"], "tuning_curves": base["tuning_curves"],
                   "policies": base["test"], "labels": LABELS,
                   "vs_baseline": {f: compare(base["test"][BASELINE], s)
                                   for f, s in base["test"].items()},
                   "vs_rule20": {f: compare(base["test"]["traditional_ma7_rule20"], s)
                                 for f, s in base["test"].items()}}
    led_b, led_a = base["ledgers"][BASELINE], base["ledgers"][AI]
    out["base"]["bootstrap_store_cluster"] = bootstrap_compare(led_b, led_a, n_boot,
                                                               int(cfg["seed"]), stores)
    out["base"]["bootstrap_series"] = bootstrap_compare(led_b, led_a, n_boot, int(cfg["seed"]))
    n_test_days = test[1] - test[0] + 1
    out["base"]["per_store_day_net_benefit_eur"] = (
        out["base"]["vs_baseline"][AI]["net_benefit_eur"] / (len(np.unique(stores))
                                                             * n_test_days))

    # Breakdown by store, category id and SKU.
    seg = pa.keys.with_row_index("sid")
    ledger_frames = []
    for fam, led in base["ledgers"].items():
        ledger_frames.append(pl.DataFrame({k: v for k, v in led.items()}).with_columns(
            pl.lit(fam).alias("policy"), pl.Series("sid", np.arange(pa.n_series))))
    ledgers = pl.concat(ledger_frames).join(seg.with_columns(pl.col("sid").cast(pl.Int64)),
                                            on="sid")
    ledgers.write_parquet(cfg.artifacts_dir / "policy_ledgers_test.parquet")
    for col in ("store_id", "first_category_id", "city_id"):
        agg = _group_compare(ledgers, col)
        agg.write_csv(cfg.metrics_dir / f"impact_by_{col}.csv")
        out["base"][f"by_{col}_summary"] = {
            "groups": agg.height,
            "share_groups_waste_reduced": float((agg["waste_reduction_pct"] > 0).mean()),
            "share_groups_stockouts_reduced": float((agg["stockout_reduction_pct"] > 0).mean()),
            "share_groups_net_benefit_positive": float((agg["net_benefit_eur"] > 0).mean()),
        }
    sku = ledgers.filter(pl.col("policy").is_in([BASELINE, AI])).pivot(
        on="policy", index="sid", values="gross_profit")
    out["base"]["share_skus_profit_improved"] = float((sku[AI] > sku[BASELINE]).mean())

    # Save test trajectories of baseline vs AI for the dashboard.
    trajs = {}
    for fam in (BASELINE, AI):
        spec = PolicySpec(fam, "forecast" if fam == AI else "traditional",
                          "quantile" if fam == AI else "ma7_sales", base["chosen"][fam])
        traj = run_policy(spec, inp, ctx, *test)
        for k in ("opening", "sold", "lost", "expired", "closing", "ordered", "received"):
            trajs[f"{fam}__{k}"] = getattr(traj, k).astype(np.float32)
    np.savez_compressed(cfg.artifacts_dir / "test_trajectories.npz", **trajs)

    # Pareto trade-off (descriptive).
    with timed(log, "simulate.pareto"):
        out["pareto"] = pareto_frontier(inp, ctx, test)
    out["iso_service_vs_baseline"] = iso_service(out["pareto"], base["test"][BASELINE])

    # Economic scenarios (re-tuned per scenario).
    out["scenarios"] = {}
    for sc in cfg["finance"]["scenarios"]:
        c = build_context(cfg, pa, sc)
        r = tune_and_test(inp, c, tune, test,
                          candidate_policies(c.econ, float(np.mean(c.price))), keep_ledgers=True)
        out["scenarios"][sc] = {"economics": c.econ.__dict__, "chosen": r["chosen"],
                                "baseline": r["test"][BASELINE], "ai": r["test"][AI],
                                "impact": _headline(r),
                                "ci_store_cluster": bootstrap_compare(
                                    r["ledgers"][BASELINE], r["ledgers"][AI], n_boot,
                                    int(cfg["seed"]), stores)}

    # Operational sensitivity (base economics, re-tuned each time).
    sens = []
    grid: list[dict[str, Any]] = (
        [{"shelf_life": s} for s in (1, 2, 3, 4, 5, 7)]
        + [{"lead_time": lt} for lt in (1, 2, 3) if lt in store.fc]
        + [{"demand_mode": m} for m in ("observed_sales", "reconstructed", "reconstructed_high")]
        + [{"noise_sigma": s} for s in (0.25, 0.5)]
    )
    two = {k: v for k, v in fams.items() if k in (BASELINE, AI)}
    with timed(log, "simulate.sensitivity", runs=len(grid)):
        for g in grid:
            c = build_context(cfg, pa, "base", **g)
            r = tune_and_test(inp, c, tune, test, two, keep_ledgers=False)
            sens.append({**{"shelf_life": c.settings.shelf_life,
                            "lead_time": c.settings.lead_time,
                            "demand_mode": g.get("demand_mode", "reconstructed"),
                            "noise_sigma": g.get("noise_sigma", 0.0)},
                         "varied": next(iter(g)), "chosen_buffer": r["chosen"][BASELINE],
                         "chosen_tau": r["chosen"][AI],
                         "baseline_waste": r["test"][BASELINE]["waste_units"],
                         "ai_waste": r["test"][AI]["waste_units"],
                         "baseline_stockouts": r["test"][BASELINE]["stockout_events"],
                         "ai_stockouts": r["test"][AI]["stockout_events"],
                         "baseline_fill_rate": r["test"][BASELINE]["fill_rate"],
                         "ai_fill_rate": r["test"][AI]["fill_rate"],
                         **_headline(r)})
    out["sensitivity"] = sens
    pl.DataFrame(sens).write_csv(cfg.metrics_dir / "sensitivity.csv")
    write_json(cfg.metrics_dir / "simulation_results.json", out)
    h = out["base"]["vs_baseline"][AI]
    log.info(kv("simulate.done", waste_reduction_pct=h["waste_reduction_pct"],
                stockout_reduction_pct=h["stockout_reduction_pct"],
                net_benefit_eur=h["net_benefit_eur"]))
    return out


def _group_compare(ledgers: pl.DataFrame, col: str) -> pl.DataFrame:
    keys = ["waste_units", "stockout_events", "gross_profit", "sold_units", "demand_units",
            "lost_units", "days"]
    g = ledgers.filter(pl.col("policy").is_in([BASELINE, AI])).group_by([col, "policy"]).agg(
        [pl.col(k).sum() for k in keys])
    b = g.filter(pl.col("policy") == BASELINE).drop("policy")
    a = g.filter(pl.col("policy") == AI).drop("policy")
    j = b.join(a, on=col, suffix="_ai")
    return j.with_columns(
        ((pl.col("waste_units") - pl.col("waste_units_ai")) / pl.col("waste_units") * 100)
        .alias("waste_reduction_pct"),
        ((pl.col("stockout_events") - pl.col("stockout_events_ai"))
         / pl.col("stockout_events") * 100).alias("stockout_reduction_pct"),
        (pl.col("gross_profit_ai") - pl.col("gross_profit")).alias("net_benefit_eur"),
        (100 * (pl.col("sold_units_ai") / pl.col("demand_units_ai")
                - pl.col("sold_units") / pl.col("demand_units"))).alias("fill_rate_change_pp"),
    ).sort(col)
