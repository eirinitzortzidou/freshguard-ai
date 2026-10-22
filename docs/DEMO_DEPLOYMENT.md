# Interactive demo: running and deploying

The dashboard (`dashboard/app.py`, Streamlit) has eight pages: executive overview, demand
forecasting, inventory optimisation, financial impact, model evaluation, robustness and
calibration, what-if simulator, and methodology and limitations.

## Data modes

| Mode | When | What is shown |
|---|---|---|
| **full** | `make all` has been run locally (git-ignored `data/processed/` artifacts exist) | everything for all 50,000 store-SKUs |
| **demo** | fresh clone or hosted deployment | aggregate pages read the committed metrics in `reports/metrics/freshretailnet/` (computed on all 50,000 series); the per-SKU explorer, inventory trajectories and what-if simulator use the committed sample in `demo/freshretailnet/` |

The demo bundle is a **uniform random sample** of 400 store-SKUs (fixed seed 2025, not
selected for good results), built by `scripts/build_demo_bundle.py` from the same pipeline
run as the committed metrics. `demo/freshretailnet/manifest.json` records the sample indices,
git commit, SHA-256 of every bundle file and the licence of the source data (FreshRetailNet-50K,
CC BY 4.0, attribution in `demo/README.md`). Nothing in the dashboard is hardcoded: every
number comes from these files, and the what-if page reruns the real tuning and simulation
code on the sample. Pages that need a missing artifact show a warning instead of failing.

No credentials, private data, model training or downloads are needed at page load. A cold
start loads < 2 MB of demo files.

## Run locally

```bash
make setup
make dashboard                      # http://localhost:8501 (full mode after `make all`)
```

## Deploy on Streamlit Community Cloud (recommended)

Streamlit Community Cloud fits this app: it is free for public repositories, needs no server
management, and the demo mode needs no secrets or large files.

1. Push the repository to GitHub (public).
2. On <https://share.streamlit.io>, choose **Create app → Deploy a public app from GitHub**.
3. Repository `eirinitzortzidou/freshguard-ai`, branch `main`, main file path
   `dashboard/app.py`.
4. **Advanced settings → Python version 3.13** (the locked versions in
   `requirements/lock.txt` were verified on 3.13). No secrets are needed.
5. Deploy. Streamlit installs `requirements.txt` (which points to the lock file); the app adds
   `src/` to the import path itself, so the package does not need to be installed.

After deployment, add the app URL to the README "Interactive demo" section and to the GitHub
repository's *About → Website* field.

Alternatives: Hugging Face Spaces (Streamlit SDK, same files) or any container host
(`streamlit run dashboard/app.py --server.port $PORT`). Both need nothing beyond the
repository contents.

## Status

The app has been verified locally in demo mode (screenshots in
`reports/figures/dashboard/`). It has **not** been deployed: deploying requires the
repository owner's Streamlit Community Cloud account, which is outside this repository.

## Refreshing screenshots and the demo bundle

```bash
make all                                        # full artifacts
python scripts/build_demo_bundle.py --n 400     # demo/freshretailnet/
pip install -e ".[screenshots]"                 # Playwright
make dashboard &                                # in another shell
python scripts/capture_dashboard.py --url http://localhost:8501 --out reports/figures/dashboard
```
