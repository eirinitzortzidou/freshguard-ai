"""Provenance, dataset checksums, model persistence and the rerun comparison tool."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from freshguard.config import PROJECT_ROOT, load_config
from freshguard.forecasting.gbm import GBMForecaster
from freshguard.provenance import (
    DataIntegrityError,
    record_stage,
    sha256_file,
    verify_checksums,
)
from freshguard.verify import compare_dirs


def _project(tmp_path: Path, name: str = "synthetic.yaml") -> Path:
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    for f in ("base.yaml", "synthetic.yaml", "freshretailnet.yaml"):
        shutil.copy(PROJECT_ROOT / "configs" / f, root / "configs" / f)
    return root


def test_gbm_save_load_roundtrip(tmp_path: Path, rng):
    n = 2000
    df = pl.DataFrame({"f1": rng.uniform(0, 5, n), "city_id": rng.integers(0, 4, n),
                       "y": rng.gamma(2.0, 1.0, n)})
    params = {"objective": "tweedie", "tweedie_variance_power": 1.2, "learning_rate": 0.1,
              "num_leaves": 7, "min_data_in_leaf": 20, "verbose": -1, "num_boost_round": 30,
              "num_threads": 1}
    m = GBMForecaster(params, seed=3).fit(df)
    m.save(tmp_path / "m.txt")
    loaded = GBMForecaster.load(tmp_path / "m.txt")
    assert loaded.features == m.features and loaded.best_iteration == m.best_iteration
    np.testing.assert_array_equal(loaded.predict(df), m.predict(df))


def test_checksum_verification(tmp_path: Path):
    root = _project(tmp_path)
    raw = root / "data" / "raw" / "freshretailnet"
    raw.mkdir(parents=True)
    (raw / "train.parquet").write_bytes(b"train-bytes")
    (raw / "eval.parquet").write_bytes(b"eval-bytes")
    good = {"train": sha256_file(raw / "train.parquet"),
            "eval": sha256_file(raw / "eval.parquet")}
    cfg = load_config(root / "configs" / "freshretailnet.yaml", root=root,
                      overrides={"data": {"sha256": good}})
    assert verify_checksums(cfg) == {"train": True, "eval": True}
    bad = load_config(root / "configs" / "freshretailnet.yaml", root=root,
                      overrides={"data": {"sha256": {**good, "eval": "0" * 64}}})
    with pytest.raises(DataIntegrityError):
        verify_checksums(bad)


def test_published_checksums_are_configured():
    cfg = load_config(PROJECT_ROOT / "configs" / "freshretailnet.yaml")
    assert set(cfg["data"]["sha256"]) == {"train", "eval"}
    assert all(len(v) == 64 for v in cfg["data"]["sha256"].values())


def test_run_manifest_records_provenance(tmp_path: Path):
    import datetime as dt

    root = _project(tmp_path)
    cfg = load_config(root / "configs" / "synthetic.yaml", root=root)
    cfg.ensure_dirs()
    (cfg.metrics_dir / "x.json").write_text('{"a": 1}')
    t0 = dt.datetime.now(dt.UTC)
    record_stage(cfg, "data", t0)
    record_stage(cfg, "forecast", t0)
    man = json.loads((cfg.metrics_dir / "run_manifest.json").read_text())
    assert set(man["stages"]) == {"data", "forecast"}
    assert man["seed"] == cfg["seed"]
    assert man["dataset"]["source"] == "synthetic"
    assert man["packages"]["numpy"] == np.__version__
    assert man["config_snapshot"]["splits"] == cfg["splits"]
    assert man["environment"]["python"]
    key = next(iter(man["outputs"]))
    assert key.endswith("x.json") and len(man["outputs"][key]["sha256"]) == 64


def test_compare_dirs_detects_differences(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "m.json").write_text(json.dumps({"wape": 0.3, "n": 5, "tag": "x"}))
    (b / "m.json").write_text(json.dumps({"wape": 0.3 + 1e-12, "n": 5, "tag": "x"}))
    pl.DataFrame({"k": ["a"], "v": [1.0]}).write_csv(a / "t.csv")
    pl.DataFrame({"k": ["a"], "v": [1.5]}).write_csv(b / "t.csv")
    diffs, missing = compare_dirs(a, b)
    by = {d.name: d for d in diffs}
    assert not missing
    assert by["m.json"].differing == 0 and by["m.json"].compared == 3
    assert by["t.csv"].differing == 1
    (a / "only_ref.json").write_text("{}")
    assert compare_dirs(a, b)[1] == ["only_ref.json"]
