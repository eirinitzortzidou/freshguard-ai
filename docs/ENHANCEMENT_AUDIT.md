# Enhancement audit (Phase 0)

Audit of the repository as published on `main` at commit `d5f6d8a` (2026-10-21), carried out
before any change. Every statement below was checked against the code, configs, tests and
committed artifacts; file references are given so each claim can be verified.

## 1. Current capabilities (verified in code)

| Area | What exists | Where |
|---|---|---|
| Ingestion | Polars Parquet loader, raw schema contract (19 columns, dtype families), rectangular-panel check | `data/ingest.py`, `data/schema.py`, `features/panel.py` |
| Data quality | duplicates, gaps, nulls, hourly-vs-daily totals, stock-count consistency, outliers (flagged, not removed) | `data/validate.py`, `reports/data_quality.md` |
| Censored demand | intraday-profile ratio estimator, profiles fitted on training days only; M5 zero-run variant with a causal `demand_hist` | `data/demand.py` |
| Features | 56 origin-indexed features, target-day inputs limited to calendar and planned discount/activity; future-perturbation leakage test | `features/build.py`, `tests/test_features.py::test_no_future_leakage` |
| Forecasting | global LightGBM per horizon (1, 2, 3, 7), deterministic mode, 9 baselines on identical rows | `forecasting/gbm.py`, `forecasting/baselines.py` |
| Backtest | validation tuning (4-point grid), early stopping on validation, expanding-window refits for two weekly test folds | `forecasting/backtest.py` |
| Uncertainty | binned empirical residual quantiles fitted on validation residuals; paired store-cluster bootstraps | `forecasting/intervals.py`, `evaluation/metrics.py`, `finance/economics.py` |
| Inventory | vectorised FEFO lost-sales simulator with lead time, shelf life, pack size, capacity, asserted invariants | `inventory/simulator.py` |
| Policies | 9 policy families; parameters tuned on validation by simulated profit | `inventory/policies.py`, `inventory/experiment.py` |
| Business analysis | exact profit decomposition, 3 economic scenarios, 13 sensitivity settings, segment breakdowns | `finance/economics.py`, `inventory/experiment.py` |
| Dashboard | six Streamlit pages reading pipeline artifacts | `dashboard/` |
| Engineering | 42 tests, ruff, mypy, GitHub Actions CI on synthetic data | `tests/`, `.github/workflows/ci.yml` |

There is no `PROJECT_STATUS.md` or `DECISIONS.md`; `PROJECT_PROGRESS.md` holds status,
decisions and protocol notes.

## 2. Verified strengths

* **Leakage controls are real and tested.** History features only read columns `<= origin`;
  the test perturbs every value after the origin and checks that only declared
  known-in-advance columns change.
* **Demand reconstruction is fitted on training days only** (`reconstruct_demand(..., fit_end=train_end)`),
  and the simulation is repeated with observed sales and a high-demand variant.
* **Strong, tuned baseline.** The traditional MA7 policy has its buffer tuned on validation, so
  the comparison is not against a straw man; a fixed "+20%" rule is reported separately.
* **Honest reporting of negative results** (stockouts +12.4% at the selected point, M5 WAPE
  not improved, observed-sales sensitivity negative).
* **Exact profit decomposition** with a unit test, and simulator invariants asserted in tests.
* **Determinism**: fixed seeds, LightGBM `deterministic=True`, seeded bootstraps.

## 3. Identified weaknesses

### Methodological risks

| # | Finding | Evidence | Severity |
|---|---|---|---|
| M1 | **The test window was inspected more than once.** After the first test run the baseline buffer grid was extended (edge of grid at 1.0 → 2.5) and the iso-service comparison was added. The README and the dashboard still say the test window was "used once" / "untouched". | `PROJECT_PROGRESS.md` "Protocol notes"; `README.md` §8; `dashboard/views/overview.py` subtitle | High |
| M2 | **Quantiles are calibrated in-sample relative to the tuning window.** The validation model is early-stopped on the validation targets, its residuals on the same days define the quantile table, and τ is then tuned by simulation on validation days 06-09..06-18 using that table. Optimistic residuals → τ chosen too low for genuinely out-of-sample errors. Test coverage of the τ = 0.9 quantile was 87.5%. | `backtest.py` lines building `store.residuals`; `experiment.py::run_experiments` (`tune` window); `forecast_metrics.json` `quantile_coverage_test` | High |
| M3 | **Only one 14-day evaluation window**, so policy conclusions rest on 2 weeks and one demand regime (late-June growth). No rolling-origin replication of the policy comparison. | `configs/base.yaml` `splits` | High |
| M4 | Quantile tables are never updated after validation (no recalibration), although the test window shows drift (bias −6.6%). | `intervals.py`, `README.md` §9 | Medium |
| M5 | The "−49% waste at equal availability" figure interpolates the test-window frontier at the baseline's test availability: it is an oracle operating point, not a selectable policy. It is labelled "descriptive" in code but presented as a headline in the README. | `experiment.py::iso_service`; `README.md` §9 | Medium |
| M6 | Store-cluster bootstraps capture cross-sectional dependence but not common day-level shocks; with 14 days the CIs are likely too narrow for "will this generalise to another fortnight" questions. | `finance/economics.py::bootstrap_compare` | Medium |
| M7 | Hyperparameters were selected on validation; for any earlier rolling fold this would be a look-ahead. The 4 grid points differ by < 0.1 pp WAPE, so the practical effect is small but undocumented. | `training_report.json` `tuning` | Low |

### Engineering risks

| # | Finding | Evidence | Severity |
|---|---|---|---|
| E1 | **Dependencies are unpinned** (`>=` only). A clean install on 2026-10-22 pulls numpy 2.5 / pandas 3.1 / polars 2.0 and `mypy` now fails with 2 errors that did not exist when CI last passed (2026-10-21). | `pyproject.toml`; `mypy` output | High |
| E2 | **No run provenance.** Metrics files do not record git SHA, dataset checksum, package versions, config snapshot or timestamps. Dataset checksums are only in prose. | `pipeline.py`, `PROJECT_PROGRESS.md` | High |
| E3 | No single documented reproduction command, and no automated comparison of a rerun against published metrics. | `Makefile` | Medium |
| E4 | Model artifacts are saved but there is **no loader** and no save/load round-trip test. | `forecasting/gbm.py::save` | Medium |
| E5 | `reports/data_quality.md` is written to one shared path by every config. The committed file actually contained the **M5** report (13,760 series), not FreshRetailNet as the README implies. | `ingest.py::prepare_dataset`; `git show d5f6d8a:reports/data_quality.md` | Medium |
| E6 | EDA stage assumes the metrics directory sits inside the project root (`relative_to`), so outputs cannot be redirected elsewhere. | `visualization/eda.py` | Low |
| E7 | Docstring references a non-existent `tests/test_leakage.py`. | `features/build.py` | Low |
| E8 | Three dashboard pages need git-ignored artifacts (`panel.parquet`, `forecasts.npz`, ledgers), so a hosted demo would show warnings on half the pages. No deployment config. | `dashboard/data.py`, `dashboard/views/*` | Medium |
| E9 | CI tests Python 3.11 only; the measured run used Python 3.13. | `ci.yml` | Low |

## 4. Prioritised improvements

| Priority | Improvement | Files expected to change | Acceptance criteria |
|---|---|---|---|
| P1 | Lock dependencies (`requirements/lock.txt`), fix numpy-2.5 typing errors, CI installs from the lock and tests 3.11 + 3.13 | `pyproject.toml`, `requirements/`, `Makefile`, `ci.yml`, `data/demand.py`, `finance/economics.py` | `ruff`, `mypy`, `pytest` pass on a clean install from the lock |
| P1 | Run manifest per stage (git SHA, dirty flag, dataset SHA-256, versions, seed, config snapshot, platform, timestamps, outputs) and dataset checksum verification | `provenance.py` (new), `pipeline.py`, `configs/freshretailnet.yaml` | `run_manifest.json` written by every stage; checksum mismatch is reported |
| P1 | `make reproduce` + metric comparison tool against committed results | `Makefile`, `freshguard/verify.py` (new) | Clean rerun compared to published metrics; differences documented in `docs/REPRODUCIBILITY_REPORT.md` |
| P1 | Model loader + round-trip test; more unit/integration tests | `forecasting/gbm.py`, `tests/` | Loaded model reproduces predictions bit-for-bit |
| P1 | Contamination audit and a **nested rolling-origin evaluation** in which every forecast, calibration and policy choice uses only data before each evaluation week; correct "untouched test" wording | `evaluation/nested.py` (new), `pipeline.py`, `configs/base.yaml`, `README.md`, dashboard | `docs/EVALUATION_PROTOCOL.md`; existing results kept unchanged in place; new outputs in a separate directory |
| P2 | Calibration study: bias by horizon, coverage by group/regime/censoring, rolling residual recalibration vs static, decision impact | `forecasting/intervals.py`, `evaluation/calibration.py` (new), `tests/` | `docs/CALIBRATION_STUDY.md`; recalibration adopted only if decision metrics improve |
| P2 | Dashboard: methodology/limitations page, calibration and nested-evaluation views, small versioned demo bundle so all pages work without the dataset, deployment config | `dashboard/`, `scripts/`, `.streamlit/`, `requirements.txt` | Every page renders from committed artifacts; `docs/DEMO_DEPLOYMENT.md` |
| P2 | README restructure with context on every headline metric, architecture/data-flow diagrams, repository metadata | `README.md`, `docs/` | Every number traceable to a committed file |
| P3 | Final QA review | `docs/FINAL_PROJECT_REVIEW.md` | All checks executed and recorded |

Out of scope (would add complexity without a clear benefit): new model families, deep
learning, per-category parameters (category ids are anonymised), changing the reconstruction
estimator.
