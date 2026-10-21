"""Cached artifact loaders for the dashboard. Everything shown is read from pipeline outputs;
nothing is hardcoded."""

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


@st.cache_resource(show_spinner="Loading panel ...")
def arrays() -> Any:
    from freshguard.data.ingest import load_prepared
    from freshguard.features.panel import to_arrays

    return to_arrays(load_prepared(config()))


@st.cache_resource(show_spinner="Loading forecasts ...")
def forecasts() -> Any:
    from freshguard.artifacts import load_store

    return load_store(config())


@st.cache_resource(show_spinner=False)
def trajectories() -> dict[str, np.ndarray]:
    p = config().artifacts_dir / "test_trajectories.npz"
    if not p.exists():
        raise MissingArtifactError("test_trajectories.npz missing. Run `make simulate`.")
    with np.load(p) as npz:
        return {k: npz[k] for k in npz.files}


@st.cache_data(show_spinner=False)
def ledgers() -> pl.DataFrame:
    p = config().artifacts_dir / "policy_ledgers_test.parquet"
    if not p.exists():
        raise MissingArtifactError("policy_ledgers_test.parquet missing.")
    return pl.read_parquet(p)


def series_lookup() -> pl.DataFrame:
    return arrays().keys.with_row_index("sid")
