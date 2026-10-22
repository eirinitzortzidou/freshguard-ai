"""Nested rolling-origin evaluation and quantile-calibration study.

The main backtest (``forecasting.backtest`` + ``inventory.experiment``) selects everything on
one validation window and reports one 14-day test window that was inspected more than once
(see ``docs/EVALUATION_PROTOCOL.md``). FreshRetailNet-50K has no later data, so a genuinely
untouched holdout cannot be created. This module instead replicates the whole decision
pipeline on several consecutive weeks, each evaluated with information from strictly
earlier days only:

1. **Walk-forward forecasts.** Every week ``s`` a LightGBM model is trained on targets
   ``< s`` with the *pre-specified* default hyperparameters of ``forecasting.lightgbm`` (not the
   validation-selected ones, which would be a look-ahead for early weeks). The number of
   boosting rounds is chosen by early stopping on the last ``es_days`` of those targets, then
   the model is refitted on all of them. It forecasts the targets of week ``s`` only.
2. **Calibration tables.** Three ways to turn point forecasts into quantiles are compared:

   * ``insample`` - the original method: residuals of the early-stopping model on its own
     early-stopping window (optimistic, because that window chose the number of rounds);
   * ``static`` - the first table fitted on ``calib_days`` of out-of-sample walk-forward
     residuals, then frozen;
   * ``rolling`` - refitted every week on the latest ``calib_days`` of out-of-sample residuals
     whose outcomes are known when the week is forecast.

3. **Outer folds.** For each outer week, policy parameters (traditional buffer, quantile
   level tau) are tuned by simulated profit on the ``tune_days`` before it, then frozen and
   simulated on the outer week. All policies share demand, costs, capacity and initial stock.

Uncertainty: pooled store-cluster bootstraps (stores resampled with all their days, which
keeps within-store temporal dependence) and the spread across folds. Folds overlap with
windows that earlier development inspected; every output is labelled accordingly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from freshguard.artifacts import load_store, write_json
from freshguard.config import Config
from freshguard.evaluation.forecast_eval import demand_segments
from freshguard.evaluation.metrics import paired_bootstrap_wape_diff, pinball, summary
from freshguard.features.build import history_features
from freshguard.features.panel import PanelArrays
from freshguard.finance.economics import bootstrap_compare, compare, summarize
from freshguard.forecasting.backtest import BASELINES, _feats, _rows, baseline_forecasts
from freshguard.forecasting.gbm import GBMForecaster
from freshguard.forecasting.intervals import ResidualQuantiles, RollingQuantiles
from freshguard.inventory.experiment import (
    AI,
    BASELINE,
    build_context,
    candidate_policies,
    tune_and_test,
)
from freshguard.inventory.policies import PolicyInputs
from freshguard.logging_utils import get_logger, kv, timed
from freshguard.splits import Splits, resolve_splits

log = get_logger(__name__)

CALIBRATIONS = ("insample", "static", "rolling")
COVERAGE_TAUS = (0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95)
PINBALL_TAUS = tuple(np.round(np.arange(0.05, 0.951, 0.05), 2))
TRADITIONAL = (BASELINE, "traditional_ma7_rule20", "ma7_reconstructed_tuned")


@dataclass(frozen=True)
class NestedPlan:
    horizon: int
    weeks: list[int]        # first target day of every walk-forward refit
    folds: list[int]        # first target day of every outer evaluation week
    fold_days: int
    tune_days: int
    calib_days: int
    min_calib_days: int
    es_days: int
    warmup: int
    primary: str


def make_plan(cfg: Config, pa: PanelArrays, sp: Splits) -> NestedPlan:
    n = cfg["nested"]
    h = int(n.get("horizon", 1))
    first = pa.day_index(n["first_week"])
    folds = [pa.day_index(d) for d in n["folds"]]
    fold_days = int(n.get("fold_days", 7))
    weeks = list(range(first, folds[-1] + 1, 7))
    plan = NestedPlan(h, weeks, folds, fold_days, int(n.get("tune_days", 14)),
                      int(n.get("calib_days", 14)), int(n.get("min_calib_days", 7)),
                      int(n.get("es_days", 7)), int(cfg["inventory"]["warmup_days"]),
                      str(n.get("primary_calibration", "insample")))
    if any(f not in weeks for f in folds):
        raise ValueError("every outer fold must start on a walk-forward week boundary")
    if folds[-1] + fold_days - 1 >= pa.n_days:
        raise ValueError("last outer fold exceeds the panel")
    if first - plan.es_days - h <= sp.train_start_origin:
        raise ValueError("first walk-forward week leaves no early-stopping training data")
    for f in folds:
        if f - plan.tune_days - plan.warmup < first:
            raise ValueError("tuning window + warm-up of an outer fold precede the first "
                             "walk-forward forecasts")
    if plan.primary not in CALIBRATIONS:
        raise ValueError(f"primary_calibration must be one of {CALIBRATIONS}")
    return plan


# ---------------------------------------------------------------------------- forecasts


def walk_forward(cfg: Config, pa: PanelArrays, sp: Splits, plan: NestedPlan
                 ) -> tuple[np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]],
                            list[dict[str, Any]]]:
    """Weekly-refit forecasts by origin day, plus each week's early-stopping residuals."""
    h = plan.horizon
    target = cfg["forecasting"]["target"]
    params = dict(cfg["forecasting"]["lightgbm"])
    seed = int(cfg["seed"])
    hist = history_features(pa, cfg["features"]["lags"], cfg["features"]["rolling_windows"],
                            target)
    fc = np.full((pa.n_series, pa.n_days), np.nan, dtype=np.float32)
    insample: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    report: list[dict[str, Any]] = []
    ends = [*plan.weeks[1:], pa.n_days]
    for s, nxt in zip(plan.weeks, ends, strict=True):
        tr = _rows(pa, hist, range(sp.train_start_origin, s - h), h, cfg, target)
        split = s - plan.es_days - h            # first origin of the early-stopping window
        inner_tr = tr.filter(pl.col("origin_idx") < split)
        inner_va = tr.filter(pl.col("origin_idx") >= split)
        feats = _feats(tr, cfg)
        with timed(log, "nested.week", week=str(pa.dates[s]), rows=tr.height):
            m = GBMForecaster(params, seed, features=feats).fit(inner_tr, inner_va)
            insample[s] = (m.predict(inner_va), inner_va["y"].to_numpy().astype(np.float64))
            mk = GBMForecaster(params, seed, features=feats).fit(
                tr, None, num_boost_round=m.best_iteration)
            mk.best_iteration = m.best_iteration
        serve = range(s - h, min(nxt, pa.n_days) - h)
        sv = _rows(pa, hist, serve, h, cfg, target)
        fc[sv["sid"].to_numpy(), sv["origin_idx"].to_numpy()] = mk.predict(sv)
        report.append({"week_start": str(pa.dates[s]), "train_rows": tr.height,
                       "early_stopping_window": [str(pa.dates[split + h]),
                                                 str(pa.dates[s - 1])],
                       "best_iteration": m.best_iteration,
                       "forecast_targets": [str(pa.dates[s]), str(pa.dates[min(nxt, pa.n_days)
                                                                           - 1])]})
        del tr, inner_tr, inner_va, sv
    return fc, insample, report


def by_target(fc_origin: np.ndarray, h: int) -> np.ndarray:
    out = np.full(fc_origin.shape, np.nan)
    out[:, h:] = fc_origin[:, :-h]
    return out


# ---------------------------------------------------------------------------- calibration


def rolling_tables(pred_t: np.ndarray, y: np.ndarray, starts: list[int], h: int,
                   calib_days: int, min_days: int, first_target: int) -> RollingQuantiles:
    """Refit a residual table at every ``start`` on targets ``[start-h-calib+1, start-h]``,
    the latest outcomes known when target ``start`` is forecast (origin ``start - h``)."""
    rq = RollingQuantiles()
    for s in starts:
        hi = s - h
        lo = max(first_target, hi - calib_days + 1)
        if hi - lo + 1 < min_days:
            continue
        p, a = pred_t[:, lo:hi + 1].ravel(), y[:, lo:hi + 1].ravel()
        rq.add(s, ResidualQuantiles().fit(p, a))
    return rq


def calibrators(pred_t: np.ndarray, y: np.ndarray, plan: NestedPlan,
                insample: dict[int, tuple[np.ndarray, np.ndarray]]
                ) -> dict[str, RollingQuantiles]:
    rolling = rolling_tables(pred_t, y, plan.weeks, plan.horizon, plan.calib_days,
                             plan.min_calib_days, plan.weeks[0])
    static = RollingQuantiles()
    frozen: ResidualQuantiles | None = None
    for s, table in zip(rolling.starts, rolling.tables, strict=True):
        full = s - plan.horizon - plan.weeks[0] + 1 >= plan.calib_days
        if frozen is None and full:
            frozen = table
        static.add(s, frozen if frozen is not None else table)
    ins = RollingQuantiles()
    for s in plan.weeks:
        ins.add(s, ResidualQuantiles().fit(*insample[s]))
    return {"insample": ins, "static": static, "rolling": rolling}


def coverage_table(pred_t: np.ndarray, y: np.ndarray, sales: np.ndarray, censored: np.ndarray,
                   days: np.ndarray, cal: dict[str, RollingQuantiles], seg: pl.DataFrame,
                   fold_of_day: dict[int, str]) -> pl.DataFrame:
    """Empirical coverage and pinball loss by method x grouping x tau on the outer days."""
    p = pred_t[:, days]
    yy, ss, cc = y[:, days], sales[:, days], censored[:, days] > 0
    groups: dict[str, np.ndarray] = {
        "all": np.full(p.shape, "all", dtype=object),
        "fold": np.broadcast_to(np.array([fold_of_day[int(d)] for d in days], dtype=object),
                                p.shape),
        "target_day_censored": np.where(cc, "censored (stockout hours)", "fully in stock"),
    }
    for col in ("volume_class", "demand_class"):
        groups[col] = np.broadcast_to(seg[col].to_numpy().astype(object)[:, None], p.shape)
    ok = np.isfinite(p) & np.isfinite(yy)
    rows: list[dict[str, Any]] = []
    for meth, rq in cal.items():
        qs = {t: rq.quantile_at(p, days, t) for t in sorted({*COVERAGE_TAUS, *PINBALL_TAUS})}
        for gname, g in groups.items():
            for val in np.unique(g[ok]):
                m = ok & (g == val)
                key = {"method": meth, "group_type": gname, "group": str(val), "n": int(m.sum())}
                for t in COVERAGE_TAUS:
                    rows.append({**key, "metric": "coverage", "tau": t,
                                 "value": float(np.mean(yy[m] <= qs[t][m]))})
                rows.append({**key, "metric": "mean_pinball", "tau": None, "value": float(
                    np.mean([pinball(yy[m], qs[t][m], t) for t in PINBALL_TAUS]))})
        # Against observed sales on fully in-stock days (no reconstruction involved).
        m = ok & ~cc
        for t in COVERAGE_TAUS:
            rows.append({"method": meth, "group_type": "observed_sales_uncensored",
                         "group": "sales, fully in stock", "n": int(m.sum()),
                         "metric": "coverage", "tau": t,
                         "value": float(np.mean(ss[m] <= qs[t][m]))})
    return pl.DataFrame(rows)


def horizon_calibration(cfg: Config, pa: PanelArrays, sp: Splits) -> list[dict[str, Any]]:
    """Bias and coverage by horizon for the *published* backtest forecasts on the test window:
    static validation tables (as published) vs. weekly rolling refits of the same forecasts."""
    store = load_store(cfg)
    target = cfg["forecasting"]["target"]
    y = pa.m[target].astype(np.float64)
    rows = []
    days = np.arange(sp.test_start, sp.test_end + 1)
    starts = [f.start for f in sp.test]
    for h in sorted(store.fc):
        pred_t = store.by_target(h).astype(np.float64)
        p, a = pred_t[:, days], y[:, days]
        ok = np.isfinite(p)
        static = RollingQuantiles([days[0]], [store.residuals[h]])
        rolling = rolling_tables(pred_t, y, starts, h, int(cfg["nested"].get("calib_days", 14)),
                                 int(cfg["nested"].get("min_calib_days", 7)), sp.valid.start)
        for name, rq in (("static_validation_table", static), ("rolling_14d", rolling)):
            for t in (0.5, 0.8, 0.9, 0.95):
                q = rq.quantile_at(p, days, t)
                rows.append({"horizon": h, "method": name, "tau": t,
                             "coverage": float(np.mean(a[ok] <= q[ok])),
                             "bias": summary(a[ok], p[ok])["bias"],
                             "wape": summary(a[ok], p[ok])["wape"]})
    return rows


# ---------------------------------------------------------------------------- evaluation


def _fold_label(sp: Splits, s: int) -> str:
    if sp.valid.start <= s <= sp.valid.end:
        return "original validation window (used for tuning in the original protocol)"
    if sp.test_start <= s <= sp.test_end:
        return "original test window (inspected before this analysis)"
    return "original training period"


def forecast_folds(pa: PanelArrays, plan: NestedPlan, pred_t: np.ndarray, target: str,
                   n_boot: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    h = plan.horizon
    y = pa.m[target].astype(np.float64)
    base = baseline_forecasts(pa, [h], target)
    stores = pa.keys["store_id"].to_numpy()
    rows: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {"y": [], "lgbm": [], "best": [], "store": []}
    for s in plan.folds:
        tune = np.arange(s - plan.tune_days, s)
        days = np.arange(s, s + plan.fold_days)
        # Nested baseline selection: best baseline on the tuning window before the fold.
        best = min(BASELINES, key=lambda m: _wape(y[:, tune], base[m][h][:, tune]))
        yy = y[:, days]
        rec: dict[str, Any] = {"fold_start": s, "baseline_selected_on_tuning": best}
        for name, mat in (("lightgbm", pred_t), *((m, base[m][h]) for m in BASELINES)):
            p = mat[:, days]
            ok = np.isfinite(p)
            r = summary(yy[ok], p[ok])
            rec[f"{name}_wape"], rec[f"{name}_bias"] = r["wape"], r["bias"]
        ok = np.isfinite(base[best][h][:, days]) & np.isfinite(pred_t[:, days])
        st = np.broadcast_to(stores[:, None], yy.shape)
        rec["wape_diff_vs_selected_baseline"] = paired_bootstrap_wape_diff(
            yy[ok], pred_t[:, days][ok], base[best][h][:, days][ok], st[ok], n_boot, seed)
        rows.append(rec)
        pooled["y"].append(yy[ok])
        pooled["lgbm"].append(pred_t[:, days][ok])
        pooled["best"].append(base[best][h][:, days][ok])
        pooled["store"].append(st[ok])
    cat = {k: np.concatenate(v) for k, v in pooled.items()}
    pooled_out = {
        "lightgbm_wape": _wape(cat["y"], cat["lgbm"]),
        "selected_baseline_wape": _wape(cat["y"], cat["best"]),
        "wape_diff_bootstrap": paired_bootstrap_wape_diff(cat["y"], cat["lgbm"], cat["best"],
                                                          cat["store"], n_boot, seed),
        "folds_lgbm_better": int(sum(
            r["lightgbm_wape"] < r[r["baseline_selected_on_tuning"] + "_wape"] for r in rows)),
    }
    return rows, pooled_out


def _wape(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(p) & np.isfinite(y)
    den = float(np.abs(y[ok]).sum())
    return float(np.abs(y[ok] - p[ok]).sum() / den) if den else float("nan")


def policy_folds(cfg: Config, pa: PanelArrays, plan: NestedPlan, fc: np.ndarray,
                 cal: dict[str, RollingQuantiles]) -> dict[str, Any]:
    h = plan.horizon
    sales = pa.m["sale_amount"].astype(np.float64)
    dh = pa.m["demand_hist"].astype(np.float64)
    inputs = {c: PolicyInputs(sales, dh, {h: fc.astype(np.float64)}, {h: cal[c]})
              for c in CALIBRATIONS}
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, list[dict[str, np.ndarray]]] = {}
    for s in plan.folds:
        tune = (s - plan.tune_days, s - 1)
        test = (s, s + plan.fold_days - 1)
        ctx = build_context(cfg, pa, "base", train_end=tune[0] - plan.warmup - 1)
        fams = candidate_policies(ctx.econ, float(np.mean(ctx.price)))
        results = {"traditional": tune_and_test(inputs[plan.primary], ctx, tune, test,
                                                {k: fams[k] for k in TRADITIONAL}),
                   "lgbm_point": tune_and_test(inputs[plan.primary], ctx, tune, test,
                                               {"lgbm_point": fams["lgbm_point"]})}
        for c in CALIBRATIONS:
            results[c] = tune_and_test(inputs[c], ctx, tune, test, {AI: fams[AI]})
        named: dict[str, tuple[float, dict[str, float], dict[str, np.ndarray]]] = {}
        for fam in TRADITIONAL:
            r = results["traditional"]
            named[fam] = (r["chosen"][fam], r["test"][fam], r["ledgers"][fam])
        r = results["lgbm_point"]
        named["lgbm_point"] = (0.0, r["test"]["lgbm_point"], r["ledgers"]["lgbm_point"])
        for c in CALIBRATIONS:
            r = results[c]
            named[f"{AI}__{c}"] = (r["chosen"][AI], r["test"][AI], r["ledgers"][AI])
        ref = named[BASELINE][1]
        for name, (param, summ, led) in named.items():
            rows.append({"fold_start": str(pa.dates[s]), "policy": name, "param": param,
                         **{k: summ[k] for k in ("waste_units", "stockout_events", "fill_rate",
                                                 "availability", "lost_units", "gross_profit",
                                                 "avg_inventory_units")},
                         **{f"vs_baseline_{k}": v for k, v in compare(ref, summ).items()
                            if k in ("waste_reduction_pct", "stockout_reduction_pct",
                                     "fill_rate_change_pp", "net_benefit_eur")}})
            ledgers.setdefault(name, []).append(led)
        log.info(kv("nested.fold", start=str(pa.dates[s]),
                    tau={c: named[f"{AI}__{c}"][0] for c in CALIBRATIONS},
                    buffer=named[BASELINE][0]))
    # Pooled over folds: per-series sums of the fold ledgers.
    pooled_led = {name: {k: np.sum([led[k] for led in leds], axis=0) for k in leds[0]}
                  for name, leds in ledgers.items()}
    stores = pa.keys["store_id"].to_numpy()
    n_boot, seed = int(cfg["finance"]["bootstrap_samples"]), int(cfg["seed"])
    pooled: dict[str, dict[str, Any]] = {}
    for name, led in pooled_led.items():
        summ = summarize(led)
        pooled[name] = {"kpis": summ, "vs_baseline": compare(summarize(pooled_led[BASELINE]),
                                                             summ)}
        if name != BASELINE:
            pooled[name]["bootstrap_store_cluster"] = bootstrap_compare(
                pooled_led[BASELINE], led, n_boot, seed, stores)
    pairs = [(f"{AI}__{a}", f"{AI}__{b}") for a, b in (("insample", "rolling"),
                                                        ("insample", "static"))]
    head_to_head = {f"{b}_vs_{a}": {"point": compare(summarize(pooled_led[a]),
                                                      summarize(pooled_led[b])),
                                     "bootstrap_store_cluster": bootstrap_compare(
                                         pooled_led[a], pooled_led[b], n_boot, seed, stores)}
                    for a, b in pairs}
    return {"by_fold": rows, "pooled": pooled, "calibration_head_to_head": head_to_head}


def run_nested(cfg: Config, pa: PanelArrays) -> dict[str, Any]:
    if not cfg.get("nested", {}).get("enabled", False):
        log.info(kv("nested.skip", reason="disabled in config"))
        return {}
    sp = resolve_splits(cfg, pa)
    plan = make_plan(cfg, pa, sp)
    out_dir = cfg.metrics_dir / "nested"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = cfg["forecasting"]["target"]
    h = plan.horizon
    with timed(log, "nested.walk_forward", weeks=len(plan.weeks)):
        fc, insample, wf_report = walk_forward(cfg, pa, sp, plan)
    np.savez_compressed(cfg.artifacts_dir / "nested_forecasts.npz", fc=fc)
    pred_t = by_target(fc.astype(np.float64), h)
    y = pa.m[target].astype(np.float64)
    cal = calibrators(pred_t, y, plan, insample)
    for c, rq in cal.items():
        write_json(cfg.artifacts_dir / f"nested_calibration_{c}.json",
                   {"starts": [str(pa.dates[s]) for s in rq.starts],
                    "tables": [{"edges": t.edges, "taus": t.taus, "table": t.table,
                                "sigma": t.sigma} for t in rq.tables]})

    n_boot, seed = int(cfg["finance"]["bootstrap_samples"]), int(cfg["seed"])
    fc_rows, fc_pooled = forecast_folds(pa, plan, pred_t, target, n_boot, seed)
    for r in fc_rows:
        r["fold_label"] = _fold_label(sp, r["fold_start"])
        r["fold_start"] = str(pa.dates[r["fold_start"]])
    days = np.concatenate([np.arange(s, s + plan.fold_days) for s in plan.folds])
    fold_of_day = {int(d): str(pa.dates[s]) for s in plan.folds
                   for d in range(s, s + plan.fold_days)}
    seg = demand_segments(pa, sp, target)
    cov = coverage_table(pred_t, y, pa.m["sale_amount"].astype(np.float64),
                         pa.m.get("censored_truth", pa.m["censored"]), days, cal, seg,
                         fold_of_day)
    cov.write_csv(out_dir / "calibration_coverage.csv")
    hz = horizon_calibration(cfg, pa, sp)
    pl.DataFrame(hz).write_csv(out_dir / "horizon_calibration.csv")
    with timed(log, "nested.policies", folds=len(plan.folds)):
        pol = policy_folds(cfg, pa, plan, fc, cal)
    pl.DataFrame(pol["by_fold"]).write_csv(out_dir / "policy_by_fold.csv")
    flat = []
    for r in fc_rows:
        b = r["wape_diff_vs_selected_baseline"]
        flat.append({**{k: v for k, v in r.items() if not isinstance(v, dict)},
                     "wape_diff": b["diff"], "wape_diff_ci_low": b["ci_low"],
                     "wape_diff_ci_high": b["ci_high"]})
    pl.DataFrame(flat).write_csv(out_dir / "forecast_by_fold.csv")
    all_cov = cov.filter((pl.col("group_type") == "all") & (pl.col("metric") == "coverage"))
    out: dict[str, Any] = {
        "plan": {**asdict(plan), "weeks": [str(pa.dates[w]) for w in plan.weeks],
                 "folds": [{"start": str(pa.dates[s]),
                            "end": str(pa.dates[s + plan.fold_days - 1]),
                            "label": _fold_label(sp, s)} for s in plan.folds],
                 "hyperparameters": "forecasting.lightgbm defaults (not validation-selected)"},
        "walk_forward": wf_report,
        "forecast": {"by_fold": fc_rows, "pooled": fc_pooled},
        "calibration": {
            "coverage_all": {m: {str(r["tau"]): r["value"]
                                 for r in all_cov.filter(pl.col("method") == m).to_dicts()}
                             for m in CALIBRATIONS},
            "mean_pinball_all": {r["method"]: r["value"] for r in cov.filter(
                (pl.col("group_type") == "all") & (pl.col("metric") == "mean_pinball")
            ).to_dicts()},
            "horizon_published_forecasts": hz},
        "policies": pol,
    }
    write_json(out_dir / "nested_results.json", out)
    from freshguard.visualization.nested_figures import nested_figures

    nested_figures(cfg, out, cov)
    p = pol["pooled"]
    log.info(kv("nested.done",
                **{c: round(p[f"{AI}__{c}"]["vs_baseline"]["net_benefit_eur"], 1)
                   for c in CALIBRATIONS}))
    return out
