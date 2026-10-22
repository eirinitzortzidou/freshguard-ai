"""Run provenance: dataset checksums and a per-stage run manifest.

Every pipeline stage records, in ``<metrics_dir>/run_manifest.json``, what produced its
outputs: git commit, dataset SHA-256, package versions, seed, the resolved configuration,
the execution environment, timestamps and the SHA-256 of every file in the metrics
directory. A published number can therefore be traced to the exact code, data and settings.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from freshguard.config import PROJECT_ROOT, Config
from freshguard.logging_utils import get_logger, kv

log = get_logger(__name__)

PACKAGES = ("freshguard", "numpy", "pandas", "polars", "pyarrow", "lightgbm", "scipy",
            "matplotlib", "plotly", "streamlit", "pyyaml")
MANIFEST = "run_manifest.json"


class DataIntegrityError(RuntimeError):
    """Raised when a dataset file does not match its published checksum."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_info(root: Path) -> dict[str, Any]:
    """Commit SHA, branch and dirty flag; ``None`` values outside a git checkout."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                                 timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def package_versions(packages: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def environment() -> dict[str, Any]:
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count()}


def dataset_files(cfg: Config) -> list[Path]:
    if cfg["data"]["source"] == "synthetic":
        return []
    raw = cfg.path(cfg["data"]["raw_dir"])
    return sorted(p for p in raw.rglob("*") if p.is_file() and p.suffix in (".parquet", ".csv"))


def dataset_identity(cfg: Config) -> dict[str, Any]:
    """Dataset name, source and SHA-256 of every raw file (synthetic: generator + seed)."""
    d = cfg["data"]
    if d["source"] == "synthetic":
        return {"source": "synthetic", "generator": "freshguard.data.synthetic",
                "seed": cfg["seed"]}
    raw = cfg.path(d["raw_dir"])
    files = {str(p.relative_to(raw)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
             for p in dataset_files(cfg)}
    return {"source": d["source"], "hf_repo_id": d.get("hf_repo_id"), "files": files,
            "expected_sha256": d.get("sha256")}


def verify_checksums(cfg: Config) -> dict[str, bool]:
    """Compare raw files against ``data.sha256`` ({split: sha256}); raise on mismatch.

    A file matches a split when the split name occurs in its file name (``train``,
    ``eval``), the convention used by the loader. Splits stored in several files are not
    checked (the published checksum refers to a single file).
    """
    expected: dict[str, str] = cfg["data"].get("sha256") or {}
    if not expected:
        return {}
    files = dataset_files(cfg)
    result: dict[str, bool] = {}
    for split, digest in expected.items():
        matches = [p for p in files if split in p.name]
        if len(matches) != 1:
            log.warning(kv("checksum.skip", split=split, files=len(matches)))
            continue
        actual = sha256_file(matches[0])
        result[split] = actual == digest
        if actual != digest:
            raise DataIntegrityError(
                f"{matches[0].name}: sha256 {actual} does not match the published dataset "
                f"checksum {digest}. Results would not be comparable with the reported ones.")
    log.info(kv("checksum.ok", splits=",".join(result)))
    return result


def _outputs(cfg: Config) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    d = cfg.metrics_dir
    if not d.exists():
        return out
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != MANIFEST:
            out[os.path.relpath(p, cfg.root)] = {"sha256": sha256_file(p),
                                                 "bytes": p.stat().st_size}
    return out


def record_stage(cfg: Config, stage: str, started: dt.datetime,
                 extra: dict[str, Any] | None = None) -> Path:
    """Add (or replace) this stage's entry in the run manifest."""
    path = cfg.metrics_dir / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    finished = dt.datetime.now(dt.UTC)
    manifest.setdefault("stages", {})[stage] = {
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": finished.isoformat(timespec="seconds"),
        "seconds": round((finished - started).total_seconds(), 1),
        "git": git_info(PROJECT_ROOT),   # the code that ran, wherever outputs are written
        **(extra or {}),
    }
    manifest.update({
        "config_name": cfg.get("config_name"),
        "seed": cfg["seed"],
        "dataset": manifest.get("dataset") if stage != "data" and manifest.get("dataset")
        else dataset_identity(cfg),
        "packages": package_versions(),
        "environment": environment(),
        "config_snapshot": cfg.raw,
        "outputs": _outputs(cfg),
        "updated_utc": finished.isoformat(timespec="seconds"),
    })
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
