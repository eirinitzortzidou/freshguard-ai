"""Compare a rerun's metrics with the published ones.

``python -m freshguard.verify <reference_dir> <candidate_dir> [--rtol 1e-6] [--atol 1e-9]``

Every JSON number and every numeric CSV cell in files present in both directories is
compared (wall-clock ``seconds`` fields are ignored). The report lists, per file, how many
values were compared, how many differ beyond tolerance and the largest differences, so a
discrepancy is documented instead of hidden.
Exit code 0 = identical within tolerance, 1 = differences found, 2 = files missing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class FileDiff:
    name: str
    compared: int = 0
    differing: int = 0
    structural: list[str] = field(default_factory=list)
    worst: list[tuple[float, str, Any, Any]] = field(default_factory=list)

    def add(self, key: str, a: Any, b: Any, rtol: float, atol: float) -> None:
        self.compared += 1
        if _close(a, b, rtol, atol):
            return
        self.differing += 1
        try:
            mag = abs(float(a) - float(b)) / max(abs(float(a)), 1e-12)
        except (TypeError, ValueError):
            mag = math.inf
        self.worst.append((mag, key, a, b))
        self.worst.sort(key=lambda t: -t[0])
        del self.worst[5:]


def _close(a: Any, b: Any, rtol: float, atol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not (
            isinstance(a, int | float) and isinstance(b, int | float)):
        return bool(a == b)
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def _walk(a: Any, b: Any, key: str, diff: FileDiff, rtol: float, atol: float) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                diff.structural.append(f"{key}/{k}: only in {'reference' if k in a else 'rerun'}")
                continue
            if k in IGNORE_KEYS:
                continue
            _walk(a[k], b[k], f"{key}/{k}", diff, rtol, atol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diff.structural.append(f"{key}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _walk(x, y, f"{key}[{i}]", diff, rtol, atol)
    else:
        diff.add(key, a, b, rtol, atol)


def compare_json(ref: Path, cand: Path, rtol: float, atol: float) -> FileDiff:
    d = FileDiff(ref.name)
    _walk(json.loads(ref.read_text()), json.loads(cand.read_text()), "", d, rtol, atol)
    return d


def compare_csv(ref: Path, cand: Path, rtol: float, atol: float) -> FileDiff:
    d = FileDiff(ref.name)
    a, b = pl.read_csv(ref), pl.read_csv(cand)
    if a.columns != b.columns or a.height != b.height:
        d.structural.append(f"shape {a.shape} vs {b.shape}")
        return d
    for c in a.columns:
        for i, (x, y) in enumerate(zip(a[c].to_list(), b[c].to_list(), strict=True)):
            d.add(f"{c}[{i}]", x, y, rtol, atol)
    return d


SKIP = {"run_manifest.json"}
# Wall-clock timings legitimately differ between runs.
IGNORE_KEYS = {"seconds"}


def compare_dirs(ref: Path, cand: Path, rtol: float = 1e-6, atol: float = 1e-9
                 ) -> tuple[list[FileDiff], list[str]]:
    diffs: list[FileDiff] = []
    missing: list[str] = []
    for p in sorted(ref.iterdir()):
        if p.name in SKIP or p.suffix not in (".json", ".csv"):
            continue
        q = cand / p.name
        if not q.exists():
            missing.append(p.name)
            continue
        diffs.append(compare_json(p, q, rtol, atol) if p.suffix == ".json"
                     else compare_csv(p, q, rtol, atol))
    return diffs, missing


def render(diffs: list[FileDiff], missing: list[str]) -> str:
    lines = ["| File | Values compared | Differing | Largest relative differences |",
             "|---|---|---|---|"]
    for d in diffs:
        worst = "; ".join(f"`{k}` {a!r} → {b!r}" for _, k, a, b in d.worst[:3]) or "none"
        if d.structural:
            worst = f"structure: {'; '.join(d.structural[:3])}. " + worst
        lines.append(f"| {d.name} | {d.compared} | {d.differing} | {worst} |")
    if missing:
        lines.append(f"\nMissing in rerun: {', '.join(missing)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    args = ap.parse_args(argv)
    diffs, missing = compare_dirs(args.reference, args.candidate, args.rtol, args.atol)
    print(render(diffs, missing))
    if missing:
        return 2
    return 1 if any(d.differing or d.structural for d in diffs) else 0


if __name__ == "__main__":
    sys.exit(main())
