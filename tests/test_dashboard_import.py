"""The dashboard modules import cleanly (no Streamlit server needed)."""

from __future__ import annotations

import ast
from pathlib import Path

from freshguard.config import PROJECT_ROOT

DASH = PROJECT_ROOT / "dashboard"


def test_dashboard_sources_parse_and_helpers_import():
    for p in [DASH / "app.py", DASH / "data.py", DASH / "ui.py", *DASH.glob("views/*.py")]:
        ast.parse(Path(p).read_text(encoding="utf-8"), filename=str(p))
    import importlib.util
    import sys

    sys.path.insert(0, str(DASH))
    try:
        for name in ("ui", "data"):
            spec = importlib.util.spec_from_file_location(name, DASH / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        assert callable(mod.simulation)
    finally:
        sys.path.remove(str(DASH))


def test_dashboard_views_reference_existing_modules():
    for p in DASH.glob("views/*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
                    "freshguard"):
                __import__(node.module)
