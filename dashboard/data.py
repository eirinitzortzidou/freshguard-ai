"""Cached artifact loaders for the dashboard. Everything shown is read from pipeline outputs;
nothing is hardcoded.

Two data modes:

* **full** - the git-ignored artifacts of ``make all`` exist locally (all 50,000 series);
* **demo** - only the committed files exist (hosted demo, fresh clone). Aggregate results come
  from ``reports/metrics`` (computed on all series by the pipeline); the per-SKU explorer and
  the what-if simulator use the versioned sample in ``demo/<dataset>/`` built by
  ``scripts/build_demo_bundle.py`` from the same run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import streamlit as st

from freshguard.config import Config, load_config


class MissingArtifactError(RuntimeError):
    pass


@st.cache_resource(show_spinner=False)
def config() -> Config:
    return load_config()


def demo_dir() -> Path:
    return config().root / "demo" / str(config().get("config_name", "freshretailnet"))


def data_mode() -> str:
    cfg = config()
    if (cfg.processed_dir / "panel.parquet").exists() and (
            cfg.artifacts_dir / "forecasts.npz").exists():
        return "full"
    if (demo_dir() / "panel_sample.npz").exists():
        return "demo"
    return "missing"


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MissingArtifactError(f"`{path.name}` not found. Run `make all` first.")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def simulation() -> dict[str, Any]:
    return _json(config().metrics_dir / "simulation_results.json")


@st.cache_data(show_spinner=False)
def forecast_metrics() -> dict[str, Any]:
    return _json(config().metrics_dir / "forecast_metrics.json")


@st.cache_data(show_spinner=False)
def nested() -> dict[str, Any]:
    return _json(config().metrics_dir / "nested" / "nested_results.json")


@st.cache_data(show_spinner=False)
def forecast_overall() -> pl.DataFrame:
    p = config().metrics_dir / "forecast_overall.csv"
    if not p.exists():
        raise MissingArtifactError("forecast_overall.csv missing. Run `make evaluate`.")
    return pl.read_csv(p)


@st.cache_data(show_spinner=False)
def csv(name: str) -> pl.DataFrame:
    p = config().metrics_dir / name
    if not p.exists():
        raise MissingArtifactError(f"{name} missing.")
    return pl.read_csv(p)


@st.cache_data(show_spinner=False)
def demo_manifest() -> dict[str, Any]:
    p = demo_dir() / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_resource(show_spinner="Loading panel ...")
def arrays() -> Any:
    from freshguard.features.panel import PanelArrays

    mode = data_mode()
    if mode == "full":
        from freshguard.data.ingest import load_prepared
        from freshguard.features.panel import to_arrays

        return to_arrays(load_prepared(config()))
    if mode == "missing":
        raise MissingArtifactError("No pipeline artifacts and no demo bundle found. Run "
                                   "`make all` (full data) or `python scripts/"
                                   "build_demo_bundle.py`.")
    import datetime as dt

    with np.load(demo_dir() / "panel_sample.npz", allow_pickle=False) as z:
        keys = pl.DataFrame({c[4:]: z[c] for c in z.files if c.startswith("key_")})
        dates = [dt.date.fromisoformat(d) for d in z["dates"].tolist()]
        mats = {c[2:]: z[c] for c in z.files if c.startswith("m_")}
    return PanelArrays(keys, dates, mats)


@st.cache_resource(show_spinner="Loading forecasts ...")
def forecasts() -> Any:
    from freshguard.artifacts import load_store

    if data_mode() == "full":
        return load_store(config())
    if data_mode() == "missing":
        raise MissingArtifactError("Forecast artifacts missing. Run `make forecast`.")
    return load_store(config().with_overrides({"paths": {"artifacts": str(demo_dir())}}))


@st.cache_resource(show_spinner=False)
def trajectories() -> dict[str, np.ndarray]:
    p = (config().artifacts_dir / "test_trajectories.npz" if data_mode() == "full"
         else demo_dir() / "test_trajectories.npz")
    if not p.exists():
        raise MissingArtifactError("test_trajectories.npz missing. Run `make simulate`.")
    with np.load(p) as npz:
        return {k: npz[k] for k in npz.files}


@st.cache_resource(show_spinner=False)
def nested_forecasts() -> np.ndarray:
    p = (config().artifacts_dir / "nested_forecasts.npz" if data_mode() == "full"
         else demo_dir() / "nested_forecasts.npz")
    if not p.exists():
        raise MissingArtifactError("nested_forecasts.npz missing. Run `make nested`.")
    with np.load(p) as npz:
        return npz["fc"]


@st.cache_data(show_spinner=False)
def ledgers() -> pl.DataFrame:
    p = config().artifacts_dir / "policy_ledgers_test.parquet"
    if not p.exists():
        raise MissingArtifactError("policy_ledgers_test.parquet missing.")
    return pl.read_parquet(p)


def series_lookup() -> pl.DataFrame:
    return arrays().keys.with_row_index("sid")
