"""Schema validation, panel construction, quality checks and demand reconstruction."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from freshguard.data.demand import (
    ReconstructionParams,
    ZeroRunParams,
    reconstruct_demand,
    zero_run_reconstruction,
)
from freshguard.data.ingest import build_panel
from freshguard.data.m5 import zero_run_lengths
from freshguard.data.schema import SchemaError, check_schema, require_schema
from freshguard.data.validate import quality_report


def test_synthetic_matches_raw_schema(synthetic_raw):
    rep = check_schema(synthetic_raw.drop("true_demand").schema)
    assert rep.ok, rep


def test_schema_detects_missing_and_wrong_types(synthetic_raw):
    with pytest.raises(SchemaError):
        require_schema(synthetic_raw.drop(["sale_amount"]).schema)
    bad = synthetic_raw.with_columns(pl.col("store_id").cast(pl.Utf8))
    assert "store_id" in check_schema(bad.schema).dtype_mismatches


def test_build_panel_sorted_typed_and_aligned(synthetic_raw):
    shuffled = synthetic_raw.sample(fraction=1.0, shuffle=True, seed=1)
    panel, hs, hst = build_panel(shuffled)
    assert panel["dt"].dtype == pl.Date
    assert hs.shape == (panel.height, 24) and hst.shape == (panel.height, 24)
    keys = panel.select(["store_id", "product_id", "dt"])
    assert keys.equals(keys.sort(["store_id", "product_id", "dt"]))
    np.testing.assert_allclose(hs.sum(1), panel["sale_amount"].to_numpy(), atol=1e-3)


def test_quality_report_flags_duplicates_gaps_and_invalid(synthetic_raw):
    panel, hs, hst = build_panel(synthetic_raw)
    rep = quality_report(panel, hs, hst)
    assert rep["duplicate_keys"] == 0 and rep["series_with_gaps"] == 0
    assert rep["stock_count_mismatch_rows"] == 0
    dup = pl.concat([panel, panel.head(3)])
    gap = panel.filter(~((pl.col("store_id") == 0) & (pl.col("product_id") == 0)
                         & (pl.col("dt") == pl.col("dt").min().dt.offset_by("5d"))))
    neg = panel.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(-1.0)
                             .otherwise(pl.col("sale_amount")).alias("sale_amount"))
    assert quality_report(dup, np.vstack([hs, hs[:3]]), np.vstack([hst, hst[:3]]))[
        "duplicate_keys"] == 3
    assert quality_report(gap, None, None)["series_with_gaps"] == 1
    assert quality_report(neg, None, None)["negative_sales_rows"] == 1


def test_reconstruction_never_below_sales_and_reduces_error(synthetic_raw):
    truth = synthetic_raw.sort(["store_id", "product_id", "dt"])["true_demand"].to_numpy()
    panel, hs, hst = build_panel(synthetic_raw.drop("true_demand"))
    out = reconstruct_demand(panel, hs, hst, "2024-05-20", ReconstructionParams())
    sales = out["sale_amount"].to_numpy()
    dem = out["demand"].to_numpy()
    assert np.all(dem >= sales - 1e-6)
    unc = out["censored"].to_numpy() == 0
    np.testing.assert_allclose(dem[unc], sales[unc])
    cen = ~unc
    assert cen.any()
    err_sales = np.abs(sales[cen] - truth[cen]).sum()
    err_recon = np.abs(dem[cen] - truth[cen]).sum()
    assert err_recon < 0.7 * err_sales  # reconstruction recovers most censored demand


def test_zero_run_lengths():
    s = np.array([[1, 0, 0, 0, 2, 0, 3]], float)
    so_far, full, start = zero_run_lengths(s)
    np.testing.assert_array_equal(so_far[0], [0, 1, 2, 3, 0, 1, 0])
    np.testing.assert_array_equal(full[0], [0, 3, 3, 3, 0, 1, 0])
    np.testing.assert_array_equal(start[0], [-1, 1, 1, 1, -1, 5, -1])


def test_zero_run_flags_only_implausible_runs_and_hist_is_causal():
    rate = np.full(40, 5.0)
    rate[30:34] = 0.0           # 4 zero days after selling 5/day: P = e^-20, flagged
    slow = np.zeros(40)
    slow[::10] = 1.0            # 0.1/day seller: zero runs are normal, never flagged
    s = np.vstack([rate, slow])
    out = zero_run_reconstruction(s, ZeroRunParams(window=28, min_rate=0.5, p_threshold=0.01))
    assert out["censored_truth"][0, 30:34].all() and out["censored_truth"][1].sum() == 0
    np.testing.assert_allclose(out["demand"][0, 30:34], 5.0)
    # Causal flag on the first zero day: exp(-5*1) < 0.01 -> flagged already; prefix of the
    # run must not depend on the future.
    s2 = s.copy()
    s2[0, 32:] = 5.0
    out2 = zero_run_reconstruction(s2)
    np.testing.assert_array_equal(out["demand_hist"][:, :32], out2["demand_hist"][:, :32])
