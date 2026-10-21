"""Shared fixtures. Everything here is synthetic and only exercises code paths."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from freshguard.data.synthetic import make_synthetic_frn


@pytest.fixture(scope="session")
def synthetic_raw() -> pl.DataFrame:
    return make_synthetic_frn(n_stores=3, n_products=4, n_days=70, seed=3, return_truth=True)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
