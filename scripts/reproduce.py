"""Independent rerun of the full pipeline, compared against the committed results.

Usage: ``make reproduce`` or
    python scripts/reproduce.py --config configs/freshretailnet.yaml --out runs/reproduce

The rerun uses an isolated project root (``--out``) that shares only ``configs/`` (copied)
and ``data/raw`` (linked) with the repository, so committed metrics and figures are never
overwritten. Afterwards every JSON/CSV metric file is compared with
``reports/metrics/<dataset>`` and the comparison is written to ``<out>/comparison.md``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/freshretailnet.yaml")
    ap.add_argument("--out", default="runs/reproduce")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--skip-run", action="store_true", help="only compare an existing rerun")
    args = ap.parse_args()

    from freshguard.config import load_config
    from freshguard.logging_utils import setup_logging
    from freshguard.pipeline import run_all
    from freshguard.verify import compare_dirs, render

    setup_logging()
    out = (ROOT / args.out).resolve()
    cfg_src = (ROOT / args.config).resolve()
    if not args.skip_run:
        (out / "configs").mkdir(parents=True, exist_ok=True)
        for f in cfg_src.parent.glob("*.yaml"):
            shutil.copy(f, out / "configs" / f.name)
        (out / "data").mkdir(exist_ok=True)
        link = out / "data" / "raw"
        if not link.exists():
            link.symlink_to(ROOT / "data" / "raw", target_is_directory=True)
        (out / "reports").mkdir(exist_ok=True)
        run_all(load_config(out / "configs" / cfg_src.name, root=out))
    ref_cfg = load_config(cfg_src)
    rerun_cfg = load_config(out / "configs" / cfg_src.name, root=out)
    diffs, missing = compare_dirs(ref_cfg.metrics_dir, rerun_cfg.metrics_dir, args.rtol)
    report = render(diffs, missing)
    (out / "comparison.md").write_text(
        f"# Rerun vs committed metrics ({cfg_src.stem}, rtol={args.rtol})\n\n{report}\n",
        encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
