"""Command-line entry point: ``freshguard <stage> [--config path]``."""

from __future__ import annotations

import argparse

from freshguard.config import load_config
from freshguard.logging_utils import setup_logging
from freshguard.pipeline import STAGES, run_all


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="freshguard", description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--config", default=None, help="YAML config (default: FreshRetailNet)")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    if args.stage == "all":
        run_all(cfg)
    else:
        STAGES[args.stage](cfg)


if __name__ == "__main__":
    main()
