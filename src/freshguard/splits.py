"""Chronological split definitions resolved to panel day indices."""

from __future__ import annotations

from dataclasses import dataclass

from freshguard.config import Config
from freshguard.features.panel import PanelArrays


@dataclass(frozen=True)
class Fold:
    name: str
    start: int   # first target day (inclusive)
    end: int     # last target day (inclusive)


@dataclass(frozen=True)
class Splits:
    train_start_origin: int
    train_end: int
    valid: Fold
    test: list[Fold]

    @property
    def test_start(self) -> int:
        return self.test[0].start

    @property
    def test_end(self) -> int:
        return self.test[-1].end

    def check(self, n_days: int, min_history: int) -> None:
        if self.train_start_origin < min_history - 1:
            raise ValueError("train_start_origin leaves less history than min_history_days")
        if not self.train_end < self.valid.start <= self.valid.end < self.test_start:
            raise ValueError("splits must be strictly chronological: train < valid < test")
        for a, b in zip(self.test, self.test[1:], strict=False):
            if b.start != a.end + 1:
                raise ValueError("test folds must be contiguous")
        if self.test_end >= n_days:
            raise ValueError("test window exceeds panel")


def resolve_splits(cfg: Config, pa: PanelArrays) -> Splits:
    s = cfg["splits"]
    sp = Splits(
        train_start_origin=pa.day_index(s["train_start_origin"]),
        train_end=pa.day_index(s["train_end"]),
        valid=Fold("valid", pa.day_index(s["valid_start"]), pa.day_index(s["valid_end"])),
        test=[Fold(f"test_{i + 1}", pa.day_index(a), pa.day_index(b))
              for i, (a, b) in enumerate(s["test_folds"])],
    )
    sp.check(pa.n_days, int(s["min_history_days"]))
    return sp
