"""Structured (key=value) logging used across the pipeline."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_FORMAT = "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once; level from argument or ``FRESHGUARD_LOG_LEVEL``."""
    lvl = (level or os.environ.get("FRESHGUARD_LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if not any(getattr(h, "_freshguard", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
        handler._freshguard = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(lvl)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def kv(event: str, **fields: Any) -> str:
    """Render a structured log line: ``event=... key=value``."""
    parts = [f"event={event}"]
    for key, value in fields.items():
        if isinstance(value, float):
            value = f"{value:.6g}"
        parts.append(f"{key}={value}")
    return " ".join(parts)


@contextmanager
def timed(logger: logging.Logger, event: str, **fields: Any) -> Iterator[None]:
    start = time.perf_counter()
    logger.info(kv(f"{event}.start", **fields))
    yield
    logger.info(kv(f"{event}.done", seconds=round(time.perf_counter() - start, 2), **fields))
