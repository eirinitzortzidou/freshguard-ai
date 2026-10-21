"""Shared chart style: a validated categorical palette and recessive axes."""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

# Categorical slots in fixed order (validated for colour-vision deficiency; see README).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQUENTIAL = "Blues"

POLICY_COLORS = {"baseline": SERIES[1], "ai": SERIES[0], "other": "#9a9893"}


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "font.size": 10,
        "lines.linewidth": 2.0, "legend.frameon": False, "figure.dpi": 110,
        "axes.prop_cycle": cycler(color=SERIES),
    })
