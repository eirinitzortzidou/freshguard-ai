# Reproducibility report

This report records how the published FreshRetailNet results were regenerated from scratch, what
matched, and what was not verified. Every number below comes from a file committed in
`reports/reproducibility/`.

## 1. How to reproduce

```bash
make setup                       # installs the exact versions in requirements/lock.txt
# place train.parquet and eval.parquet in data/raw/freshretailnet/
make reproduce                   # clean rerun in runs/reproduce/, compared with committed metrics
```

`make reproduce` (`scripts/reproduce.py`) runs every stage (data, eda, forecast, evaluate,
simulate, nested, report) into an isolated root under `runs/reproduce/`. The committed results
are not touched. It then compares every metrics file with the committed one at a relative
tolerance of 1e-6 (`python -m freshguard.verify REF CAND`) and writes `comparison.md`.

## 2. What is recorded for each run

Every stage writes or updates `reports/metrics/<config>/run_manifest.json`
(`src/freshguard/provenance.py`):

| Field | Content |
|---|---|
| `stages.<name>` | start and finish time (UTC), duration, git commit, branch, dirty flag |
| `seed` | global seed (2025) |
| `dataset` | SHA-256 and size of each raw file, plus the expected checksums from the config |
| `packages` | installed versions of the key libraries |
| `environment` | Python version, platform, CPU count |
| `config_snapshot` | the fully resolved configuration |
| `outputs` | SHA-256 of every file written to the metrics directory |

Before ingesting FreshRetailNet, the data stage checks the raw files against the checksums in
`configs/freshretailnet.yaml` and stops with `DataIntegrityError` on a mismatch.

## 3. Final rerun (2026-10-22)

| Item | Value |
|---|---|
| Code | commit `190f69c` |
| Dataset | `train.parquet` SHA-256 `6706832d…5a3d` (106,436,287 bytes); `eval.parquet` `1b118840…960e` (8,440,124 bytes); both match the config |
| Environment | CPython 3.13.16, Linux x86_64 (glibc 2.39), 4 CPUs, 15 GB RAM |
| Packages | numpy 2.5.4, pandas 3.1.0, polars 2.0.0, pyarrow 26.0.0, lightgbm 4.7.0, scipy 1.18.1, matplotlib 3.11.2, plotly 7.1.0, streamlit 1.66.1, pyyaml 6.0.3 |
| Seed | 2025 |
| Wall time | 29 min (20:06 to 20:35 UTC) |

Stage durations from the manifest: data 28.5 s, eda 4.1 s, forecast 609.8 s, evaluate 9.7 s,
simulate 227.1 s, nested 854.0 s, report 1.3 s.

The manifest records `dirty: true` from the forecast stage onward. That is because
`PROJECT_PROGRESS.md` was edited and `docs/REPOSITORY_METADATA.md` was created while the run was
in progress. No code or configuration changed. Files:
`reports/reproducibility/final_rerun_2026-10-22_manifest.json` and
`final_rerun_2026-10-22_comparison.md`.

### Results

| Output set | Values compared | Differing at rtol 1e-6 |
|---|---|---|
| Original pipeline metrics (17 files: data quality, EDA, forecasts, simulation, sensitivity, segment impacts) | 46,638 | 29 |
| Nested evaluation (5 files) | 4,427 | 0 |

All 29 differing values are in two segment breakdowns: `forecast_by_city_id.csv` (10) and
`forecast_by_first_category_id.csv` (19). The largest relative difference is about 2.5e-6 (for
example city 1 WAPE 0.3072062 → 0.3072054). Every headline file is identical within tolerance:
`forecast_metrics.json`, `forecast_overall.csv`, `simulation_results.json`, `sensitivity.csv`,
`training_report.json` and all nested outputs.

**Probable cause (not proven):** these columns are float32 sums of about 10^5 values per group.
An interim rerun of the original code (`rerun_2026-10-22_comparison.md`) also differed only in
these two files, but in different cells (4 and 9). The variation therefore looks like it comes
from the summation order in multi-threaded aggregation, not from the models or the simulation.
Model predictions themselves are bit-for-bit deterministic; `tests/test_reproducibility.py`
checks this with a save/load round trip.

## 4. Earlier rerun of the original code

Before any methodological change, the pipeline as published at `d5f6d8a` was rerun with the
locked environment. That rerun matched within the same tolerance, apart from 13 float32 segment
cells and the recorded `seconds` timings, which are excluded from comparisons
(`reports/reproducibility/rerun_2026-10-22_comparison.md`). The published numbers therefore did
not depend on the newer, unpinned package versions.

## 5. Verified and not verified

| Verified | Not verified |
|---|---|
| Full rerun from raw files on Python 3.13 with the lock file reproduces every headline metric | Bit-identical results on other CPUs, operating systems or BLAS builds (LightGBM `deterministic=True` targets same-machine determinism) |
| Dataset checksums enforced at ingestion | Python 3.11 full-data run (CI runs 3.11 with latest packages on synthetic data only) |
| Model save/load gives identical predictions (test) | Reproduction of the M5 benchmark with the new tooling (not rerun) |
| Synthetic end-to-end pipeline, manifest and nested stage in CI | Whether a fresh Hugging Face download is byte-identical to the uploaded files (Hugging Face is blocked in this environment; checksums would catch a mismatch) |

## 6. Known limitations

* The lock file is a `pip freeze` for Linux, Python 3.13. Other platforms install the closest
  wheels and may differ in the last float32 digits.
* Timings vary with machine load. LightGBM slowed 5 to 10 times when another CPU-heavy job ran at
  the same time.
* The full run needs the 115 MB raw dataset, which is not stored in the repository (CC BY 4.0,
  available from Hugging Face `Dingdong-Inc/FreshRetailNet-50K`).
