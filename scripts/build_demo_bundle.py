"""Build the small, versioned demo bundle used by the hosted dashboard.

Usage (after ``make all``):
    python scripts/build_demo_bundle.py --config configs/freshretailnet.yaml --n 400

Writes ``demo/<dataset>/``: a uniformly random sample (fixed seed, not cherry-picked) of
store-SKU series with their daily sales, reconstructed demand and stockout flags, the
pipeline's forecasts, residual tables, test-window simulation trajectories and nested
walk-forward forecasts for those series. Aggregate dashboard numbers never come from this
sample; they are read from ``reports/metrics`` (all series).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATRICES = ("sale_amount", "demand", "demand_hist", "censored_truth", "censored",
            "instock_share")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/freshretailnet.yaml")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    from freshguard.artifacts import load_store
    from freshguard.config import load_config
    from freshguard.data.schema import HIERARCHY
    from freshguard.pipeline import load_arrays
    from freshguard.provenance import git_info, sha256_file

    cfg = load_config(ROOT / args.config)
    pa = load_arrays(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))
    idx = np.sort(rng.choice(pa.n_series, size=min(args.n, pa.n_series), replace=False))
    out = ROOT / "demo" / cfg["config_name"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    keys = pa.keys[idx]
    np.savez_compressed(
        out / "panel_sample.npz",
        dates=np.array([d.isoformat() for d in pa.dates]),
        **{f"key_{c}": keys[c].to_numpy() for c in HIERARCHY},
        **{f"m_{c}": pa.m[c][idx].astype(np.float32) for c in MATRICES if c in pa.m})
    store = load_store(cfg)
    np.savez_compressed(out / "forecasts.npz", **{f"h{h}": v[idx] for h, v in store.fc.items()})
    for h in store.residuals:
        shutil.copy(cfg.artifacts_dir / f"residuals_h{h}.json", out / f"residuals_h{h}.json")
    (out / "forecast_meta.json").write_text(json.dumps(
        {**store.meta, "n_series": int(idx.size), "n_days": store.n_days}, indent=2,
        default=str), encoding="utf-8")
    with np.load(cfg.artifacts_dir / "test_trajectories.npz") as z:
        np.savez_compressed(out / "test_trajectories.npz", **{k: z[k][idx] for k in z.files})
    nf = cfg.artifacts_dir / "nested_forecasts.npz"
    if nf.exists():
        with np.load(nf) as z:
            np.savez_compressed(out / "nested_forecasts.npz", fc=z["fc"][idx])
    manifest = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": cfg["config_name"], "n_series": int(idx.size), "of_series": pa.n_series,
        "selection": f"uniform random without replacement, numpy default_rng(seed={cfg['seed']})",
        "series_index": idx.tolist(), "git": git_info(ROOT),
        "source_run_manifest": str((cfg.metrics_dir / "run_manifest.json").relative_to(ROOT)),
        "files": {p.name: sha256_file(p) for p in sorted(out.iterdir())},
        "licence": "Derived from FreshRetailNet-50K (Dingdong-Inc), CC BY 4.0",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"demo bundle: {out} ({sum(p.stat().st_size for p in out.iterdir()) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
