"""Rolling-origin backtest: tuning on validation, frozen selection, expanding-window refits
on the test folds, plus baseline forecasts on identical rows.

Origin ownership (no leakage): the validation model (fitted on targets <= train_end) serves
origins up to the day before the first test fold; test fold ``k`` is served by a model refit
on all targets before the fold start, for origins from (fold start - 1) onwards.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from freshguard.config import Config
from freshguard.evaluation.metrics import summary
from freshguard.features.build import build_rows, history_features
from freshguard.features.panel import PanelArrays
from freshguard.forecasting.baselines import forecast_matrix
from freshguard.forecasting.gbm import GBMForecaster
from freshguard.forecasting.intervals import ResidualQuantiles
from freshguard.logging_utils import get_logger, kv, timed
from freshguard.splits import Splits

log = get_logger(__name__)

BASELINES = ("naive", "seasonal_naive", "ma7", "ma28", "dow_mean_4w", "ma28_dow_index",
             "ses", "croston_sba", "tsb")


@dataclass
class ForecastStore:
    """Point forecasts by horizon, indexed by origin day: ``fc[h][:, t]`` -> day ``t + h``."""

    n_series: int
    n_days: int
    fc: dict[int, np.ndarray] = field(default_factory=dict)
    residuals: dict[int, ResidualQuantiles] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def put(self, h: int, origins: np.ndarray, sid: np.ndarray, pred: np.ndarray) -> None:
        if h not in self.fc:
            self.fc[h] = np.full((self.n_series, self.n_days), np.nan, dtype=np.float32)
        self.fc[h][sid, origins] = pred

    def by_target(self, h: int) -> np.ndarray:
        """Re-index to target day: column T holds the forecast of T made at T - h."""
        out = np.full((self.n_series, self.n_days), np.nan, dtype=np.float32)
        out[:, h:] = self.fc[h][:, : self.n_days - h]
        return out


def _rows(pa: PanelArrays, hist: dict[str, np.ndarray], origins: range | list[int], h: int,
          cfg: Config, target: str) -> pl.DataFrame:
    f = cfg["features"]
    return build_rows(pa, list(origins), h, f["lags"], f["rolling_windows"],
                      f["known_future"], target_col=target, hist=hist)


def _feats(df: pl.DataFrame, cfg: Config) -> list[str]:
    from freshguard.features.build import feature_columns

    drop = set(cfg["features"].get("exclude", []))
    return [c for c in feature_columns(df) if c not in drop]


def run_backtest(cfg: Config, pa: PanelArrays, sp: Splits) -> tuple[ForecastStore,
                                                                    dict[str, Any]]:
    fcfg = cfg["forecasting"]
    target = fcfg["target"]
    horizons = sorted(set(fcfg["horizons"]))
    base_params = dict(fcfg["lightgbm"])
    seed = int(cfg["seed"])
    with timed(log, "features.history"):
        hist = history_features(pa, cfg["features"]["lags"], cfg["features"]["rolling_windows"],
                                target)
    store = ForecastStore(pa.n_series, pa.n_days)
    report: dict[str, Any] = {"tuning": [], "models": [], "feature_importance": {}}

    # 1. Hyperparameter selection (h = primary) on the validation window only.
    h0 = int(fcfg["primary_horizon"])
    params = dict(base_params)
    if fcfg["tuning"]["enabled"]:
        dev_start = max(sp.train_start_origin, sp.train_end - h0 - int(
            fcfg["tuning"].get("dev_origin_days", 120)))
        tr = _rows(pa, hist, range(dev_start, sp.train_end - h0 + 1), h0, cfg, target)
        va = _rows(pa, hist, range(sp.valid.start - h0, sp.valid.end - h0 + 1), h0, cfg,
                   target)
        best = None
        for cand in fcfg["tuning"]["grid"]:
            p = {**base_params, **cand}
            t0 = time.perf_counter()
            m = GBMForecaster(p, seed, features=_feats(tr, cfg)).fit(tr, va)
            pred = m.predict(va)
            res = summary(va["y"].to_numpy(), pred)
            rec = {"params": cand, "best_iteration": m.best_iteration,
                   "valid_wape": res["wape"], "valid_mae": res["mae"],
                   "seconds": round(time.perf_counter() - t0, 1),
                   "train_rows": tr.height, "dev_subset_origins": f"{dev_start}..."}
            report["tuning"].append(rec)
            log.info(kv("tuning.candidate", **{k: v for k, v in rec.items()
                                               if k != "params"}, params=json.dumps(cand)))
            if best is None or res["wape"] < best[0]:
                best = (res["wape"], cand)
        assert best is not None
        params = {**base_params, **best[1]}
        del tr, va
    report["selected_params"] = params

    # 2. Validation models per horizon: early stopping on validation targets.
    next_owner = sp.test[0].start - 1
    for h in horizons:
        tr = _rows(pa, hist, range(sp.train_start_origin, sp.train_end - h + 1), h, cfg, target)
        va_orig = range(sp.valid.start - h, sp.valid.end - h + 1)
        va = _rows(pa, hist, va_orig, h, cfg, target)
        with timed(log, "gbm.valid_fit", horizon=h, rows=tr.height):
            m = GBMForecaster(params, seed, features=_feats(tr, cfg)).fit(tr, va)
        m.save(cfg.models_dir / f"lgbm_h{h}_valid.txt")
        report["feature_importance"][h] = m.feature_importance().head(25).to_dicts()
        best_iter = m.best_iteration
        # Serve every origin owned by the validation model whose target is out-of-sample
        # (target > train_end); this also covers the warm-up of the test simulation.
        serve = range(sp.valid.start - h, next_owner)
        sv = _rows(pa, hist, serve, h, cfg, target)
        store.put(h, sv["origin_idx"].to_numpy(), sv["sid"].to_numpy(), m.predict(sv))
        vpred = store.fc[h][va["sid"].to_numpy(), va["origin_idx"].to_numpy()]
        store.residuals[h] = ResidualQuantiles().fit(vpred.astype(np.float64),
                                                     va["y"].to_numpy().astype(np.float64))
        report["models"].append({"horizon": h, "fold": "valid", "train_rows": tr.height,
                                 "best_iteration": best_iter})
        del tr, va, sv
        # 3. Test folds: expanding-window refit with the frozen configuration.
        for k, fold in enumerate(sp.test):
            tr = _rows(pa, hist, range(sp.train_start_origin, fold.start - h), h, cfg, target)
            with timed(log, "gbm.fold_fit", horizon=h, fold=fold.name, rows=tr.height):
                mk = GBMForecaster(params, seed, features=_feats(tr, cfg))
                mk.fit(tr, None, num_boost_round=best_iter)
            mk.best_iteration = best_iter
            mk.save(cfg.models_dir / f"lgbm_h{h}_{fold.name}.txt")
            last = sp.test[k + 1].start - 1 if k + 1 < len(sp.test) else pa.n_days
            serve = range(fold.start - 1, min(last, pa.n_days))
            sv = _rows(pa, hist, serve, h, cfg, target)
            store.put(h, sv["origin_idx"].to_numpy(), sv["sid"].to_numpy(), mk.predict(sv))
            report["models"].append({"horizon": h, "fold": fold.name, "train_rows": tr.height,
                                     "num_boost_round": best_iter})
            del tr, sv
    store.meta = {"params": params, "horizons": horizons, "target": target}
    return store, report


def baseline_forecasts(pa: PanelArrays, horizons: list[int], target: str,
                       ses_alpha: float = 0.2) -> dict[str, dict[int, np.ndarray]]:
    """Baseline forecasts indexed by *target* day for every method and horizon."""
    from freshguard.features.build import history_column

    y = pa.m[history_column(pa, target)].astype(np.float64)
    out: dict[str, dict[int, np.ndarray]] = {}
    for meth in BASELINES:
        out[meth] = {h: forecast_matrix(y, meth, h, alpha=ses_alpha if meth == "ses" else 0.1)
                     for h in horizons}
    return out


def tune_ses_alpha(pa: PanelArrays, sp: Splits, target: str, h: int = 1) -> float:
    from freshguard.evaluation.metrics import wape
    from freshguard.features.build import history_column

    y = pa.m[history_column(pa, target)].astype(np.float64)
    truth = pa.m[target][:, sp.valid.start: sp.valid.end + 1]
    best = (np.inf, 0.2)
    for a in (0.05, 0.1, 0.2, 0.3, 0.5):
        f = forecast_matrix(y, "ses", h, alpha=a)[:, sp.valid.start: sp.valid.end + 1]
        w = wape(truth.ravel(), f.ravel())
        best = min(best, (w, a))
    return float(best[1])
