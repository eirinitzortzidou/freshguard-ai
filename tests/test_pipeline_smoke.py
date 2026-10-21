"""End-to-end smoke test on the synthetic fixture (never used for reported results)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from freshguard.config import PROJECT_ROOT, load_config
from freshguard.pipeline import run_all


@pytest.mark.integration
def test_full_pipeline_runs_on_synthetic(tmp_path: Path):
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    for f in ("base.yaml", "synthetic.yaml"):
        shutil.copy(PROJECT_ROOT / "configs" / f, root / "configs" / f)
    cfg = load_config(root / "configs" / "synthetic.yaml", root=root,
                      overrides={"finance": {"bootstrap_samples": 20}})
    run_all(cfg)
    sim = json.loads((cfg.metrics_dir / "simulation_results.json").read_text())
    fm = json.loads((cfg.metrics_dir / "forecast_metrics.json").read_text())
    assert "lgbm_cost_optimal" in sim["base"]["policies"]
    d = sim["base"]["vs_baseline"]["lgbm_cost_optimal"]
    assert set(d) >= {"waste_reduction_pct", "stockout_reduction_pct", "net_benefit_eur"}
    assert fm["best_baseline_by_valid"]
    assert (cfg.figures_dir / "01_daily_trend.png").exists()
    # Physical sanity of every policy's KPIs.
    for s in sim["base"]["policies"].values():
        assert 0 <= s["fill_rate"] <= 1 + 1e-9
        assert s["sold_units"] <= s["demand_units"] + 1e-6


def test_simulation_reproducible(tmp_path: Path):
    """Same seed + inputs -> identical KPI ledger."""
    import numpy as np

    from freshguard.finance.economics import Economics, series_ledger
    from freshguard.inventory.simulator import SimSettings, simulate

    rng = np.random.default_rng(5)
    dem = rng.gamma(2, 2, (30, 20))
    tgt = rng.uniform(0, 8, (30, 20))
    econ = Economics(0.3, 0.1, 0.25, 0.5, 0.0)
    a = series_ledger(simulate(dem, tgt, SimSettings()), np.full((30, 20), 2.0), econ)
    b = series_ledger(simulate(dem, tgt, SimSettings()), np.full((30, 20), 2.0), econ)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
