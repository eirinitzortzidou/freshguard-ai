"""Pipeline stages. Each stage reads the previous stage's artifacts, so stages can be rerun
independently (``freshguard <stage>``); ``freshguard all`` runs them in order."""

from __future__ import annotations

import polars as pl

from freshguard.artifacts import save_store, write_json
from freshguard.config import Config
from freshguard.data.ingest import load_prepared, prepare_dataset
from freshguard.features.panel import PanelArrays, to_arrays
from freshguard.forecasting.backtest import run_backtest
from freshguard.logging_utils import get_logger, kv, timed
from freshguard.splits import resolve_splits

log = get_logger(__name__)


def load_arrays(cfg: Config) -> PanelArrays:
    return to_arrays(load_prepared(cfg))


def stage_data(cfg: Config) -> pl.DataFrame:
    cfg.ensure_dirs()
    with timed(log, "stage.data"):
        return prepare_dataset(cfg)


def stage_forecast(cfg: Config) -> None:
    cfg.ensure_dirs()
    pa = load_arrays(cfg)
    sp = resolve_splits(cfg, pa)
    with timed(log, "stage.forecast", series=pa.n_series, days=pa.n_days):
        store, report = run_backtest(cfg, pa, sp)
    save_store(cfg, store)
    write_json(cfg.metrics_dir / "training_report.json", report)
    log.info(kv("stage.forecast.saved", dir=cfg.artifacts_dir))


def stage_evaluate(cfg: Config) -> None:
    from freshguard.evaluation.forecast_eval import evaluate_forecasts

    evaluate_forecasts(cfg, load_arrays(cfg))


def stage_simulate(cfg: Config) -> None:
    from freshguard.inventory.experiment import run_experiments

    run_experiments(cfg, load_arrays(cfg))


def stage_eda(cfg: Config) -> None:
    from freshguard.visualization.eda import run_eda

    run_eda(cfg, load_prepared(cfg))


def stage_report(cfg: Config) -> None:
    from freshguard.visualization.report_figures import build_report

    build_report(cfg)


STAGES = {"data": stage_data, "eda": stage_eda, "forecast": stage_forecast,
          "evaluate": stage_evaluate, "simulate": stage_simulate, "report": stage_report}


def run_all(cfg: Config) -> None:
    for name, fn in STAGES.items():
        log.info(kv("pipeline.stage", name=name))
        fn(cfg)
