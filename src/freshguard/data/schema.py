"""Expected schema of FreshRetailNet-50K (and of the synthetic fixture that mimics it)."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

SERIES_KEYS: tuple[str, str] = ("store_id", "product_id")
HIERARCHY: tuple[str, ...] = (
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
)

RAW_SCHEMA: dict[str, pl.DataType] = {
    "city_id": pl.Int64(),
    "store_id": pl.Int64(),
    "management_group_id": pl.Int64(),
    "first_category_id": pl.Int64(),
    "second_category_id": pl.Int64(),
    "third_category_id": pl.Int64(),
    "product_id": pl.Int64(),
    "dt": pl.Utf8(),
    "sale_amount": pl.Float64(),
    "hours_sale": pl.List(pl.Float64()),
    "stock_hour6_22_cnt": pl.Int64(),
    "hours_stock_status": pl.List(pl.Int64()),
    "discount": pl.Float64(),
    "holiday_flag": pl.Int64(),
    "activity_flag": pl.Int64(),
    "precpt": pl.Float64(),
    "avg_temperature": pl.Float64(),
    "avg_humidity": pl.Float64(),
    "avg_wind_level": pl.Float64(),
}

# Normalised analytical dtypes (compact, Parquet-friendly).
PANEL_DTYPES: dict[str, pl.DataType] = {
    "city_id": pl.Int16(),
    "store_id": pl.Int32(),
    "management_group_id": pl.Int16(),
    "first_category_id": pl.Int32(),
    "second_category_id": pl.Int32(),
    "third_category_id": pl.Int32(),
    "product_id": pl.Int32(),
    "dt": pl.Date(),
    "sale_amount": pl.Float32(),
    "stock_hour6_22_cnt": pl.Int8(),
    "discount": pl.Float32(),
    "holiday_flag": pl.Int8(),
    "activity_flag": pl.Int8(),
    "precpt": pl.Float32(),
    "avg_temperature": pl.Float32(),
    "avg_humidity": pl.Float32(),
    "avg_wind_level": pl.Float32(),
}

HOURS = 24


class SchemaError(ValueError):
    """Raised when an input table does not match the expected schema."""


@dataclass(frozen=True)
class SchemaReport:
    missing_columns: list[str]
    extra_columns: list[str]
    dtype_mismatches: dict[str, tuple[str, str]]

    @property
    def ok(self) -> bool:
        return not self.missing_columns and not self.dtype_mismatches


def _compatible(actual: pl.DataType, expected: pl.DataType) -> bool:
    if actual == expected:
        return True
    if isinstance(expected, pl.List):
        return isinstance(actual, (pl.List, pl.Array)) and (
            actual.inner.is_numeric()
        )
    if expected.is_integer():
        return actual.is_integer()
    if expected.is_float():
        return actual.is_numeric()
    if expected == pl.Utf8():
        return actual in (pl.Utf8(), pl.Date(), pl.Datetime())
    return False


def check_schema(schema: pl.Schema | dict[str, pl.DataType],
                 expected: dict[str, pl.DataType] = RAW_SCHEMA) -> SchemaReport:
    """Compare a frame schema against the expected raw schema (dtype families, not widths)."""
    actual = dict(schema)
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    mismatches = {
        c: (str(actual[c]), str(t))
        for c, t in expected.items()
        if c in actual and not _compatible(actual[c], t)
    }
    return SchemaReport(missing, extra, mismatches)


def require_schema(schema: pl.Schema | dict[str, pl.DataType]) -> None:
    rep = check_schema(schema)
    if not rep.ok:
        raise SchemaError(
            f"schema mismatch: missing={rep.missing_columns} dtype={rep.dtype_mismatches}"
        )
