# Final project review

Review of the 2026-10-22 enhancement of FreshGuard AI, compared with `main` at `d5f6d8a`. Every
result quoted here was measured and is traceable to a committed file.

## 1. Summary of improvements

| Area | Before | After |
|---|---|---|
| Dependencies | `>=` ranges only; a clean install no longer passed `mypy` | `requirements/lock.txt`; CI tests the lock (3.13) and the latest versions (3.11) |
| Provenance | none | `run_manifest.json` per run: git SHA, dataset SHA-256, seed, versions, config, timings, output hashes; checksums enforced |
| Reproduction | manual | `make reproduce` plus automated comparison; full rerun matches (`docs/REPRODUCIBILITY_REPORT.md`) |
| Evaluation | one 14-day test window, inspected more than once and described as "untouched" | contamination audit plus nested rolling-origin evaluation over 4 weeks, with fold-local tuning (`docs/EVALUATION_PROTOCOL.md`) |
| Calibration | in-sample residual table, never updated | coverage by group, regime and censoring; bias by horizon; rolling vs static vs in-sample study with decision impact (`docs/CALIBRATION_STUDY.md`) |
| Dashboard | 6 pages, 3 needed git-ignored artifacts | 8 pages; demo bundle so every page works without the dataset; deployment config (`docs/DEMO_DEPLOYMENT.md`) |
| Documentation | README with one-window headline numbers | restructured README showing nested and single-split results side by side, an architecture diagram, and limitations up front |
| Bugs fixed | `data_quality.md` held the M5 report; EDA stage could not write outside the repo; model could not be reloaded | fixed and tested |

## 2. Files modified

* **New modules:** `src/freshguard/provenance.py`, `verify.py`, `evaluation/nested.py`,
  `visualization/nested_figures.py`.
* **Modified modules:**
  * `pipeline.py` and `cli.py`: nested stage, provenance recording, checksum check.
  * `forecasting/intervals.py`: `RollingQuantiles`.
  * `forecasting/gbm.py`: `load`.
  * `inventory/policies.py`: time-indexed quantiles.
  * `data/ingest.py`: per-config quality report.
  * `data/demand.py`, `finance/economics.py`: typing fixes for numpy 2.5.
  * `visualization/eda.py`, `features/build.py`.
* **Configs:** `base.yaml` (`nested:` block), `freshretailnet.yaml` (checksums), `m5.yaml`,
  `synthetic.yaml`.
* **Build and CI:** `requirements/lock.txt`, `requirements.txt`, `pyproject.toml`, `Makefile`
  (`setup`, `setup-latest`, `check`, `nested`, `reproduce`), `.gitignore`,
  `.github/workflows/ci.yml`, `.streamlit/config.toml`.
* **Scripts:** `scripts/reproduce.py`, `build_demo_bundle.py`, `capture_dashboard.py`.
* **Dashboard:**
  * `app.py`, `data.py`.
  * New pages: `views/robustness.py`, `views/methodology.py`.
  * Updated pages: `views/overview.py`, `views/whatif.py`.
  * `demo/` bundle: 400 series, 0.77 MB, CC BY 4.0 attribution.
* **Tests:** `tests/test_reproducibility.py` (5), `tests/test_nested.py` (7),
  `tests/test_pipeline_smoke.py`.
* **Results:**
  * Added: `reports/metrics/freshretailnet/nested/` (5 files), figures 16–18,
    `reports/reproducibility/`, `reports/data_quality_m5.md`.
  * Corrected: `reports/data_quality.md` now holds the FreshRetailNet report.
  * Refreshed: the 8 dashboard screenshots.
  * All previously published metrics files are unchanged.
* **Docs:**
  * New: `ENHANCEMENT_AUDIT.md`, `EVALUATION_PROTOCOL.md`, `CALIBRATION_STUDY.md`,
    `DEMO_DEPLOYMENT.md`, `REPRODUCIBILITY_REPORT.md`, `REPOSITORY_METADATA.md`, this file.
  * Updated: `README.md`, `docs/methodology.md`, `PROJECT_PROGRESS.md`.

## 3. Tests executed and results

| Check | Result |
|---|---|
| `pytest` (locked env, Python 3.13) | 54 passed (was 42), ~12 s |
| `ruff check`, `ruff format --check` | clean |
| `mypy src` | clean (2 errors under numpy 2.5 fixed) |
| GitHub Actions on `190f69c` | both jobs green: locked 3.13 (lint, types, CLI and config checks, tests) and compat 3.11 with the latest packages |
| Full-data rerun (`make reproduce`) | 51,065 values compared; 29 differ by at most ~2.5e-6 relative, all in float32 segment sums |
| Dashboard | all 8 pages render in full and demo mode; screenshots taken with Playwright from the running app |
| `pip-audit` on the lock file | no known vulnerabilities |
| Licences of key dependencies | permissive (BSD, MIT, Apache-2.0, PSF); dataset CC BY 4.0, attributed in `demo/README.md` |

## 4. Reproducibility status

**Reproducible.** On the same platform with the lock file, a clean rerun from the raw files
reproduces every headline metric and every nested result. The only differences are in the last
float32 digits of two segment tables. Cross-platform bit-identity was not tested. See
`docs/REPRODUCIBILITY_REPORT.md`.

## 5. Methodological improvements

1. **Contamination audit.** The original test window had been inspected more than once: the
   baseline grid was extended after the first test run. The README and dashboard no longer call
   it untouched. The single-split numbers are kept unchanged and labelled as inspected.
2. **Nested rolling-origin evaluation** over 4 weeks (2024-06-05 to 07-02):
   * Weekly walk-forward refits with pre-specified hyperparameters.
   * Early stopping only on past targets.
   * Policy parameters tuned on the 14 days before each week.
   * Results:
     * WAPE 31.3% vs 34.8% (−3.54 pp, CI −3.66 to −3.42); LightGBM better in all 4 weeks.
     * Waste −15.3%, stockouts −35.7%, hypothetical profit +€36.4k (CI 34.5k–38.3k).
     * Profit gain in every week; waste rose in one week (+18.5% in the week of 06-19).
3. **Comparability check.** On the two original test weeks, the nested design gives waste +0.9%
   and stockouts −43.3%. The original split reported waste −56.5% and stockouts +12.4%. The
   original trade-off came mostly from the baseline buffer chosen in that one window (1.25 vs
   1.0 in every nested fold). This is documented, not hidden.
4. **Calibration study (negative result).**
   * Rolling recalibration did not beat the published in-sample table on coverage, pinball loss
     or profit (−€649, CI −814 to −508). The existing method is kept.
   * New finding: censored (stockout) days are under-covered, at 78.1% vs 96.3% for in-stock
     days at τ = 0.9.
   * Forecast bias grows with horizon (−6.6% at 1 day to −9.9% at 7 days).
5. **Oracle labelling.** The iso-service comparison point is now labelled an oracle, not a
   selectable policy.

## 6. Dashboard status

* **Pages:** Executive overview, Demand forecasting, Inventory optimization, Financial impact,
  Model evaluation, Robustness and calibration (new), What-if simulator, and Methodology and
  limitations (new).
* **Demo mode:** runs from the committed demo bundle when the dataset is absent, and a badge marks
  it.
* **Deployment:** ready for Streamlit Community Cloud (`dashboard/app.py`, Python 3.13, no secrets
  needed). **It is not deployed.** Deployment needs the owner's Streamlit account; the steps are
  in `docs/DEMO_DEPLOYMENT.md`.

## 7. Remaining limitations

* Euro figures are hypothetical: FreshRetailNet has no prices or costs, so scenario economics are
  used.
* 4 evaluation weeks in one season (June 2024). The bootstraps resample store clusters, not days,
  so the CIs do not cover common week-level shocks.
* Latent demand on stockout days is an estimate. Results with observed sales are reported as a
  sensitivity case.
* Under-coverage on censored days and a negative bias that grows with horizon.
* Lead time 1 day and shelf life 3 days are assumptions; the sensitivity analysis covers
  alternatives.
* M5 secondary benchmark: no WAPE gain over TSB (documented negative result, not rerun with the
  new tooling).

## 8. Unresolved issues

* Live dashboard URL missing (needs the owner's account).
* Rerun differences in the last float32 digits of two segment tables. They could be removed by
  aggregating in float64; this was not changed, to keep published files identical.
* Repository description and topics are proposed in `docs/REPOSITORY_METADATA.md` but must be set
  by the owner on GitHub.

## 9. Future research directions

1. Censoring-aware calibration (separate or Tobit-style residual models for stockout days).
2. Bias correction by horizon, or direct quantile objectives in LightGBM.
3. Longer and multi-season evaluation, and day-block bootstraps.
4. Real cost and price data, or a partner dataset, to replace scenario economics.
5. Multi-echelon or lead-time-uncertain extensions of the order-up-to policy.

## 10. Portfolio readiness

**Ready to present, with one manual step: deploying the dashboard.** The project now shows:

* An end-to-end pipeline on a real 50,000-series dataset.
* Leakage tests.
* A reproducible run with provenance.
* An evaluation protocol that admits and corrects its own earlier contamination.
* Two documented negative results.
* Business results stated with confidence intervals and a hypothetical-currency caveat.

Main risk in an interview setting: the headline trade-off changed between the original split
(large waste cut, more stockouts) and the nested evaluation (smaller waste cut, far fewer
stockouts). Present the nested result as primary and explain why. The README and
`docs/EVALUATION_PROTOCOL.md` already do this.
