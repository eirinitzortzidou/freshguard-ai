"""Capture dashboard screenshots for the README (requires Playwright + Chromium).

Usage: start the app (`make dashboard`), then
    python scripts/capture_dashboard.py --url http://localhost:8501 --out reports/figures/dashboard
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PAGES = {
    "overview": "",
    "forecasting": "forecasting",
    "inventory": "inventory",
    "financial": "financial",
    "evaluation": "evaluation",
    "whatif": "whatif",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default="reports/figures/dashboard")
    ap.add_argument("--run-whatif", action="store_true")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exe = os.environ.get("CHROMIUM_PATH")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        for name, slug in PAGES.items():
            page.goto(f"{args.url}/{slug}", wait_until="networkidle")
            page.wait_for_timeout(9000)
            if name == "whatif" and args.run_whatif:
                page.get_by_role("button", name="Run simulation").click()
                page.wait_for_timeout(45000)
            page.screenshot(path=str(out / f"{name}.png"), full_page=True)
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
