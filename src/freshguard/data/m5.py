"""M5 (Walmart) loader: FOODS category mapped onto the canonical FreshGuard panel.

M5 provides daily unit sales for 3,049 products in 10 US stores (2011-01-29 .. 2016-06-19),
weekly shelf prices in USD, a calendar of events and SNAP days. It has *no* inventory,
stockout or shelf-life information and no "fresh" product labels: departments are
anonymised (FOODS_1/2/3). FreshGuard therefore uses the FOODS departments as a stand-in for
perishable assortment *under explicit shelf-life assumptions*, and never labels them as
fruit, vegetables, meat or bakery.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from freshguard.logging_utils import get_logger, kv

log = get_logger(__name__)

M5_URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def download_m5(raw_dir: Path) -> Path:
    """Fetch the public M5 archive (mirror maintained by Nixtla) if not already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    needed = ["sales_train_evaluation.csv", "sales_test_evaluation.csv", "calendar.csv",
              "sell_prices.csv"]
    if all((raw_dir / f).exists() for f in needed):
        return raw_dir
    import urllib.request
    import zipfile

    archive = raw_dir / "m5.zip"
    if not archive.exists():
        log.info(kv("m5.download", url=M5_URL))
        urllib.request.urlretrieve(M5_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        for name in needed:
            zf.extract(name, raw_dir)
    return raw_dir


def load_m5_panel(raw_dir: Path, start: str, end: str, categories: list[str],
                  require_full_price_history: bool = True
                  ) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    """Return (long panel, label mapping, audit counts) for ``start..end`` inclusive."""
    cal = pl.read_csv(raw_dir / "calendar.csv", null_values=["NA"]).with_columns(
        pl.col("date").str.to_date()
    )
    if "d" not in cal.columns:  # the Nixtla mirror drops the "d_<n>" key; d_1 = 2011-01-29
        cal = cal.with_columns(
            ("d_" + ((pl.col("date") - pl.date(2011, 1, 29)).dt.total_days() + 1)
             .cast(pl.Utf8)).alias("d"))
    cal = cal.filter((pl.col("date") >= dt.date.fromisoformat(start))
                     & (pl.col("date") <= dt.date.fromisoformat(end)))
    dcols = cal["d"].to_list()
    train = pl.read_csv(raw_dir / "sales_train_evaluation.csv")
    test = pl.read_csv(raw_dir / "sales_test_evaluation.csv")
    wide = train.join(test, on=ID_COLS, how="left")
    audit: dict[str, int] = {"series_total": wide.height}
    wide = wide.filter(pl.col("cat_id").is_in(categories))
    audit["series_in_categories"] = wide.height
    missing = [c for c in dcols if c not in wide.columns]
    if missing:
        raise ValueError(f"M5 sales file lacks days {missing[:3]}...")
    wide = wide.select([*ID_COLS, *dcols])

    long = wide.unpivot(index=ID_COLS, on=dcols, variable_name="d", value_name="sale_amount")
    long = long.join(cal.select(["d", "date", "wm_yr_wk", "event_name_1", "event_type_1",
                                 "snap_CA", "snap_TX", "snap_WI"]), on="d", how="left")
    prices = pl.read_csv(raw_dir / "sell_prices.csv")
    long = long.join(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # Series must have a shelf price on every day of the window (launched before the window
    # and never delisted), otherwise revenue and discount are undefined. Count, don't hide.
    priced = long.group_by(["store_id", "item_id"]).agg(
        pl.col("sell_price").is_null().sum().alias("n_missing_price"))
    keep = priced.filter(pl.col("n_missing_price") == 0) if require_full_price_history \
        else priced
    audit["series_dropped_missing_price"] = priced.height - keep.height
    long = long.join(keep.select(["store_id", "item_id"]), on=["store_id", "item_id"],
                     how="inner")
    audit["series_kept"] = keep.height

    snap = (pl.when(pl.col("state_id") == "CA").then(pl.col("snap_CA"))
            .when(pl.col("state_id") == "TX").then(pl.col("snap_TX"))
            .otherwise(pl.col("snap_WI")))
    long = long.sort(["store_id", "item_id", "date"]).with_columns(
        # Discount proxy: price relative to the series' max price over the trailing 8 weeks.
        (pl.col("sell_price") / pl.col("sell_price").rolling_max(56, min_samples=1)
         .over(["store_id", "item_id"])).alias("discount"),
        pl.col("event_name_1").is_not_null().cast(pl.Int8).alias("holiday_flag"),
        snap.cast(pl.Int8).alias("activity_flag"),
    )

    labels = long.select(["state_id", "store_id", "cat_id", "dept_id", "item_id"]).unique()
    maps = {c: {v: i for i, v in enumerate(sorted(labels[c].unique().to_list()))}
            for c in ("state_id", "store_id", "cat_id", "dept_id", "item_id")}
    panel = long.select(
        pl.col("state_id").replace_strict(maps["state_id"]).cast(pl.Int16).alias("city_id"),
        pl.col("store_id").replace_strict(maps["store_id"]).cast(pl.Int32).alias("store_id"),
        pl.col("cat_id").replace_strict(maps["cat_id"]).cast(pl.Int16)
        .alias("management_group_id"),
        pl.col("dept_id").replace_strict(maps["dept_id"]).cast(pl.Int32)
        .alias("first_category_id"),
        pl.col("dept_id").replace_strict(maps["dept_id"]).cast(pl.Int32)
        .alias("second_category_id"),
        pl.col("dept_id").replace_strict(maps["dept_id"]).cast(pl.Int32)
        .alias("third_category_id"),
        pl.col("item_id").replace_strict(maps["item_id"]).cast(pl.Int32).alias("product_id"),
        pl.col("date").alias("dt"),
        pl.col("sale_amount").cast(pl.Float32),
        pl.col("sell_price").cast(pl.Float32).alias("price"),
        pl.col("discount").cast(pl.Float32),
        pl.col("holiday_flag"),
        pl.col("activity_flag"),
        pl.lit(0, dtype=pl.Int8).alias("stock_hour6_22_cnt"),
    ).sort(["store_id", "product_id", "dt"])
    label_map = labels.with_columns(
        pl.col("store_id").replace_strict(maps["store_id"]).cast(pl.Int32).alias("store_idx"),
        pl.col("item_id").replace_strict(maps["item_id"]).cast(pl.Int32).alias("product_idx"),
        pl.col("dept_id").replace_strict(maps["dept_id"]).cast(pl.Int32).alias("dept_idx"),
    ).rename({"store_id": "store_name", "item_id": "product_name", "dept_id": "dept_name",
              "state_id": "state_name", "cat_id": "cat_name"})
    audit["rows"] = panel.height
    audit["days"] = len(dcols)
    log.info(kv("m5.load", **audit))
    return panel, label_map, audit


def zero_run_lengths(sales: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a (S, T) matrix return run-so-far length, full run length and run start index
    of the zero-sales run containing each day (0 / -1 for days with positive sales)."""
    s, t = sales.shape
    zero = sales <= 0
    so_far = np.zeros((s, t), dtype=np.int32)
    start = np.full((s, t), -1, dtype=np.int32)
    run = np.zeros(s, dtype=np.int32)
    cur_start = np.full(s, -1, dtype=np.int32)
    for j in range(t):
        z = zero[:, j]
        cur_start = np.where(z & (run == 0), j, cur_start)
        run = np.where(z, run + 1, 0)
        so_far[:, j] = run
        start[:, j] = np.where(z, cur_start, -1)
    full = np.zeros((s, t), dtype=np.int32)
    nxt = np.zeros(s, dtype=np.int32)
    for j in range(t - 1, -1, -1):
        z = zero[:, j]
        nxt = np.where(z, np.maximum(nxt, so_far[:, j]), 0)
        full[:, j] = nxt
    return so_far, full, start
