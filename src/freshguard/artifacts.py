"""Persistence of intermediate artifacts shared by pipeline stages and the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from freshguard.config import Config
from freshguard.forecasting.backtest import ForecastStore
from freshguard.forecasting.intervals import ResidualQuantiles


def save_store(cfg: Config, store: ForecastStore) -> Path:
    d = cfg.artifacts_dir
    d.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {f"h{h}": v for h, v in store.fc.items()}
    np.savez_compressed(d / "forecasts.npz", **arrays)
    for h, rq in store.residuals.items():
        rq.to_json(d / f"residuals_h{h}.json")
    (d / "forecast_meta.json").write_text(json.dumps(
        {**store.meta, "n_series": store.n_series, "n_days": store.n_days}, indent=2,
        default=str), encoding="utf-8")
    return d


def load_store(cfg: Config) -> ForecastStore:
    d = cfg.artifacts_dir
    meta = json.loads((d / "forecast_meta.json").read_text(encoding="utf-8"))
    store = ForecastStore(meta["n_series"], meta["n_days"], meta=meta)
    with np.load(d / "forecasts.npz") as npz:
        for k in npz.files:
            store.fc[int(k[1:])] = npz[k]
    for h in store.fc:
        p = d / f"residuals_h{h}.json"
        if p.exists():
            store.residuals[h] = ResidualQuantiles.from_json(p)
    return store


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def _default(o: Any) -> Any:
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
