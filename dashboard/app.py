"""FreshGuard AI decision-support dashboard.

Run with ``make dashboard`` (``streamlit run dashboard/app.py``). Reads the artifacts of
``make all``; set ``FRESHGUARD_CONFIG`` to switch dataset configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="FreshGuard AI", page_icon="🥬", layout="wide")

PAGES = [
    st.Page("views/overview.py", title="Executive overview", icon="📊", default=True),
    st.Page("views/forecasting.py", url_path="forecasting", title="Demand forecasting", icon="📈"),
    st.Page("views/inventory.py", url_path="inventory", title="Inventory optimization", icon="📦"),
    st.Page("views/financial.py", url_path="financial", title="Financial impact", icon="💶"),
    st.Page("views/evaluation.py", url_path="evaluation", title="Model evaluation", icon="🧪"),
    st.Page("views/whatif.py", url_path="whatif", title="What-if simulator", icon="🎛️"),
]

nav = st.navigation(PAGES)
with st.sidebar:
    st.markdown("### FreshGuard AI")
    st.caption("Demand forecasting and perishable replenishment for fresh food. "
               "Offline simulation on FreshRetailNet-50K.")
nav.run()
