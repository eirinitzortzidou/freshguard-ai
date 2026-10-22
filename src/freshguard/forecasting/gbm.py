"""Global LightGBM demand model (one direct model per horizon)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from freshguard.features.build import CATEGORICAL, feature_columns
from freshguard.logging_utils import get_logger, kv

log = get_logger(__name__)


def _to_pandas(df: pl.DataFrame, cols: list[str]) -> Any:
    pdf = df.select(cols).to_pandas()
    for c in CATEGORICAL:
        if c in pdf.columns:
            pdf[c] = pdf[c].astype("category")
    return pdf


@dataclass
class GBMForecaster:
    """Thin wrapper that fixes feature handling, seeding and non-negative outputs."""

    params: dict[str, Any]
    seed: int = 2025
    features: list[str] = field(default_factory=list)
    booster: lgb.Booster | None = None
    best_iteration: int = 0
    categories: dict[str, list[Any]] = field(default_factory=dict)

    def _lgb_params(self) -> dict[str, Any]:
        p = {k: v for k, v in self.params.items()
             if k not in ("num_boost_round", "early_stopping_rounds")}
        if p.get("objective") != "tweedie":
            p.pop("tweedie_variance_power", None)
        p.update(seed=self.seed, deterministic=True, force_row_wise=True,
                 bagging_seed=self.seed, feature_fraction_seed=self.seed)
        return p

    def _frame(self, df: pl.DataFrame) -> Any:
        pdf = _to_pandas(df, self.features)
        for c, cats in self.categories.items():
            pdf[c] = pdf[c].cat.set_categories(cats)
        return pdf

    def fit(self, train: pl.DataFrame, valid: pl.DataFrame | None = None,
            num_boost_round: int | None = None) -> GBMForecaster:
        train = train.filter(pl.col("y").is_not_null())
        self.features = self.features or feature_columns(train)
        xtr = _to_pandas(train, self.features)
        self.categories = {c: list(xtr[c].cat.categories) for c in CATEGORICAL
                           if c in xtr.columns}
        dtrain = lgb.Dataset(xtr, label=train["y"].to_numpy(), free_raw_data=True)
        rounds = num_boost_round or int(self.params.get("num_boost_round", 1000))
        callbacks: list[Any] = [lgb.log_evaluation(period=0)]
        valid_sets = []
        if valid is not None and valid.height:
            valid = valid.filter(pl.col("y").is_not_null())
            dvalid = lgb.Dataset(self._frame(valid), label=valid["y"].to_numpy(),
                                 reference=dtrain)
            valid_sets = [dvalid]
            callbacks.append(lgb.early_stopping(int(self.params.get("early_stopping_rounds",
                                                                     50)), verbose=False))
        self.booster = lgb.train(self._lgb_params(), dtrain, num_boost_round=rounds,
                                 valid_sets=valid_sets, callbacks=callbacks)
        self.best_iteration = int(self.booster.best_iteration or self.booster.num_trees())
        log.info(kv("gbm.fit", rows=train.height, features=len(self.features),
                    best_iteration=self.best_iteration))
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        pred = self.booster.predict(self._frame(df), num_iteration=self.best_iteration)
        return np.clip(np.asarray(pred, dtype=np.float64), 0.0, None)

    def feature_importance(self) -> pl.DataFrame:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        gain = self.booster.feature_importance(importance_type="gain")
        return pl.DataFrame({"feature": self.features, "gain": gain}).sort("gain",
                                                                          descending=True)

    def save(self, path: Path) -> None:
        if self.booster is None:
            raise RuntimeError("model is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path), num_iteration=self.best_iteration)
        meta = {"params": self.params, "seed": self.seed, "features": self.features,
                "best_iteration": self.best_iteration,
                "categories": {k: [int(x) for x in v] for k, v in self.categories.items()}}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> GBMForecaster:
        """Restore a model written by :meth:`save` (booster + feature/category metadata)."""
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        m = cls(meta["params"], int(meta["seed"]), features=list(meta["features"]),
                booster=lgb.Booster(model_file=str(path)),
                best_iteration=int(meta["best_iteration"]),
                categories={k: list(v) for k, v in meta["categories"].items()})
        return m
