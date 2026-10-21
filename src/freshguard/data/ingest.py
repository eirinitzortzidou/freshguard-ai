"""Dataset acquisition and conversion into a sorted, typed analytical panel (Parquet).

FreshRetailNet-50K is published on the Hugging Face Hub under CC BY 4.0. The raw files are
downloaded into ``data/raw/freshretailnet`` (git-ignored) and never redistributed by this
repository. A manual import path is supported: place the original ``train`` and ``eval``
Parquet files in the raw directory and run ``make data``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from freshguard.config import Config
from freshguard.data.schema import HOURS, PANEL_DTYPES, SERIES_KEYS, require_schema
from freshguard.data.synthetic import make_synthetic_frn
from freshguard.logging_utils import get_logger, kv

log = get_logger(__name__)


class DataUnavailableError(RuntimeError):
    """Raised when the raw dataset is neither present locally nor downloadable."""


def download_freshretailnet(cfg: Config) -> Path:
    """Download the dataset Parquet files from the Hugging Face Hub (idempotent)."""
    raw_dir = cfg.path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    if _find_split_files(raw_dir, "train") and _find_split_files(raw_dir, "eval"):
        log.info(kv("download.skip", reason="files_present", dir=raw_dir))
        return raw_dir
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=cfg["data"]["hf_repo_id"],
            repo_type="dataset",
            local_dir=str(raw_dir),
            allow_patterns=["*.parquet", "README.md"],
        )
    except Exception as exc:  # network policy, missing package, auth, ...
        raise DataUnavailableError(
            "Could not download FreshRetailNet-50K from the Hugging Face Hub "
            f"({type(exc).__name__}: {exc}). Manual import: download the 'train' and 'eval' "
            f"Parquet files from https://huggingface.co/datasets/{cfg['data']['hf_repo_id']} "
            f"and place them in {raw_dir}."
        ) from exc
    log.info(kv("download.done", dir=raw_dir))
    return raw_dir


def _find_split_files(raw_dir: Path, split: str) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("*.parquet") if split in p.name)


def load_raw(cfg: Config) -> pl.DataFrame:
    """Load train + eval splits (or the synthetic fixture) as one raw table."""
    if cfg["data"]["source"] == "synthetic":
        log.info(kv("load_raw", source="synthetic"))
        return make_synthetic_frn(seed=cfg["seed"])
    raw_dir = cfg.path(cfg["data"]["raw_dir"])
    frames = []
    for split in ("train", "eval"):
        files = _find_split_files(raw_dir, split)
        if not files:
            raise DataUnavailableError(f"no '{split}' parquet files under {raw_dir}")
        lf = pl.scan_parquet(files)
        require_schema(lf.collect_schema())
        frames.append(lf.collect().with_columns(pl.lit(split).alias("split")))
        log.info(kv("load_raw.split", split=split, rows=frames[-1].height, files=len(files)))
    return pl.concat(frames, how="vertical_relaxed")


def build_panel(raw: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Normalise dtypes, sort by series/date and split hourly arrays off the scalar panel.

    Returns
    -------
    panel : scalar columns, one row per store-SKU-day, sorted by (store, product, dt)
    hours_sale : float32 array (rows, 24) aligned with ``panel``
    hours_stock : int8 array (rows, 24); 1 = out of stock during that hour
    """
    if "split" not in raw.columns:
        raw = raw.with_columns(pl.lit("train").alias("split"))
    df = raw.with_columns(
        pl.col("dt").cast(pl.Utf8).str.to_date(strict=False).alias("dt")
    ).sort([*SERIES_KEYS, "dt"])
    casts = [pl.col(c).cast(t) for c, t in PANEL_DTYPES.items() if c in df.columns]
    df = df.with_columns(casts)
    hours_sale = _list_to_array(df["hours_sale"], np.float32)
    hours_stock = _list_to_array(df["hours_stock_status"], np.int8)
    panel = df.drop(["hours_sale", "hours_stock_status"])
    return panel, hours_sale, hours_stock


def _list_to_array(col: pl.Series, dtype: type[np.generic]) -> np.ndarray:
    lengths = col.list.len()
    if lengths.min() != HOURS or lengths.max() != HOURS:
        raise ValueError(f"{col.name}: expected {HOURS} hourly values per row")
    flat = col.explode().fill_null(0).to_numpy()
    return flat.astype(dtype, copy=False).reshape(-1, HOURS)


def save_panel(cfg: Config, panel: pl.DataFrame, hours_sale: np.ndarray,
               hours_stock: np.ndarray) -> None:
    out = cfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(out / "panel_raw.parquet", compression="zstd")
    np.savez_compressed(out / "hourly.npz", hours_sale=hours_sale, hours_stock=hours_stock)
    log.info(kv("save_panel", rows=panel.height, dir=out))


def load_panel(cfg: Config) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    out = cfg.processed_dir
    panel = pl.read_parquet(out / "panel_raw.parquet")
    with np.load(out / "hourly.npz") as npz:
        return panel, npz["hours_sale"], npz["hours_stock"]


def prepare_dataset(cfg: Config) -> pl.DataFrame:
    """Ingest the configured source and write the canonical ``panel.parquet``.

    The canonical panel has one row per store-SKU-day with hierarchy ids, observed
    ``sale_amount``, reconstructed ``demand`` (evaluation target), ``demand_hist`` (causal
    version for features), ``censored`` flags, promotion/calendar fields and, when the source
    has it, ``price``.
    """
    from freshguard.data.demand import (
        ReconstructionParams,
        ZeroRunParams,
        reconstruct_demand,
        zero_run_reconstruction,
    )
    from freshguard.data.validate import quality_report, write_quality_report

    source = cfg["data"]["source"]
    out = cfg.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    dcfg = cfg["demand"]
    if source == "m5":
        from freshguard.data.m5 import download_m5, load_m5_panel

        raw_dir = download_m5(cfg.path(cfg["data"]["raw_dir"]))
        panel, labels, audit = load_m5_panel(raw_dir, cfg["data"]["start"], cfg["data"]["end"],
                                             cfg["data"]["categories"])
        labels.write_parquet(out / "labels.parquet")
        rep = quality_report(panel, None, None)
        rep["ingest_audit"] = audit
        n_s = panel.select(["store_id", "product_id"]).n_unique()
        sales = panel["sale_amount"].to_numpy().reshape(n_s, -1)
        zr = zero_run_reconstruction(sales, ZeroRunParams(**dcfg.get("zero_run", {})))
        panel = panel.with_columns(
            pl.Series(k, v.reshape(-1)) for k, v in zr.items()
        ).with_columns(
            (1 - pl.col("censored").cast(pl.Float32)).alias("instock_share"),
            (pl.col("demand") - pl.col("sale_amount")).alias("lost_demand"),
        )
    else:
        if source == "freshretailnet":
            download_freshretailnet(cfg)
        raw = load_raw(cfg)
        panel, hours_sale, hours_stock = build_panel(raw)
        rep = quality_report(panel, hours_sale, hours_stock, cfg["data"]["open_hour_start"],
                             cfg["data"]["open_hour_end"])
        params = ReconstructionParams(
            open_start=cfg["data"]["open_hour_start"], open_end=cfg["data"]["open_hour_end"],
            min_instock_share=dcfg["min_instock_share"],
            profile_shrinkage=dcfg["profile_shrinkage"],
            max_uplift_factor=dcfg["max_uplift_factor"])
        panel = reconstruct_demand(panel, hours_sale, hours_stock, cfg["splits"]["train_end"],
                                   params)
        panel = panel.with_columns(pl.col("demand").alias("demand_hist"),
                                   pl.col("censored").alias("censored_truth"))
        if "split" in panel.columns:
            panel = panel.drop("split")
    rep["demand_reconstruction"] = {
        "censored_truth_share": float(panel["censored_truth"].cast(pl.Float64).mean() or 0),
        "observed_sales_total": float(panel["sale_amount"].sum()),
        "reconstructed_demand_total": float(panel["demand"].sum()),
        "uplift_pct": float(100 * (panel["demand"].sum() / max(panel["sale_amount"].sum(),
                                                               1e-9) - 1)),
    }
    panel.write_parquet(out / "panel.parquet", compression="zstd")
    write_quality_report(rep, cfg.metrics_dir / "data_quality.json",
                         cfg.path("reports") / "data_quality.md")
    log.info(kv("prepare_dataset.done", source=source, rows=panel.height))
    return panel


def load_prepared(cfg: Config) -> pl.DataFrame:
    path = cfg.processed_dir / "panel.parquet"
    if not path.exists():
        raise DataUnavailableError(f"{path} missing - run `make data` first")
    return pl.read_parquet(path)
