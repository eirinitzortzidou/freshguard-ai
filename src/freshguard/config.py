"""Configuration loading and project path resolution."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "freshretailnet.yaml"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass(frozen=True)
class Config:
    """Thin wrapper around the YAML configuration with path helpers."""

    raw: dict[str, Any]
    root: Path = PROJECT_ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, relative: str | os.PathLike[str]) -> Path:
        """Resolve a project-relative path (never an absolute hardcoded path)."""
        p = Path(relative)
        return p if p.is_absolute() else self.root / p

    @property
    def processed_dir(self) -> Path:
        return self.path(self.raw["data"]["processed_dir"])

    @property
    def artifacts_dir(self) -> Path:
        return self.path(self.raw["paths"]["artifacts"])

    @property
    def figures_dir(self) -> Path:
        return self.path(self.raw["paths"]["figures"])

    @property
    def metrics_dir(self) -> Path:
        return self.path(self.raw["paths"]["metrics"])

    @property
    def models_dir(self) -> Path:
        return self.path(self.raw["paths"]["models"])

    def with_overrides(self, overrides: dict[str, Any], root: Path | None = None) -> Config:
        return Config(_deep_update(self.raw, overrides), root or self.root)

    def ensure_dirs(self) -> None:
        for d in (self.processed_dir, self.artifacts_dir, self.figures_dir, self.metrics_dir,
                  self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


def _read_with_extends(cfg_path: Path) -> dict[str, Any]:
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)
    if parent:
        base = _read_with_extends(cfg_path.parent / parent)
        raw = _deep_update(base, raw)
    raw["config_name"] = cfg_path.stem
    return raw


def load_config(path: str | os.PathLike[str] | None = None,
                overrides: dict[str, Any] | None = None,
                root: Path | None = None) -> Config:
    """Load a YAML config. ``FRESHGUARD_CONFIG`` overrides the default location."""
    cfg_path = Path(path or os.environ.get("FRESHGUARD_CONFIG") or DEFAULT_CONFIG)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = PROJECT_ROOT / cfg_path
    raw = _read_with_extends(cfg_path)
    cfg = Config(raw, root or PROJECT_ROOT)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg
