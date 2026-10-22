# Evaluation protocol

This document defines how FreshGuard results are evaluated, audits every past use of the
test window, and explains the nested rolling-origin evaluation added on 2026-10-22. Results
of the nested evaluation are in section 5; the calibration part is in
[`CALIBRATION_STUDY.md`](CALIBRATION_STUDY.md).

## 1. Data available for evaluation

FreshRetailNet-50K covers 97 consecutive days, 2024-03-28 to 2024-07-02: 90 "train" days and
the official 7-day "eval" week. No later public data exists for these 50,000 store-SKUs. **A
genuinely untouched final holdout therefore cannot be created**: every day of the dataset was
either used for development or already inspected as test data. This is disclosed wherever
results are reported.

| Window | Dates | Role in the original protocol |
|---|---|---|
| Feature burn-in | 03-28 .. 04-23 | history for 28-day features only |
| Training targets | up to 06-04 | model fitting; demand-profile, capacity and segment estimation |
| Validation | 06-05 .. 06-18 | hyperparameters, early stopping, baseline selection, SES α, residual tables, policy tuning (06-09 .. 06-18) |
| Test fold 1 | 06-19 .. 06-25 | refit on all earlier targets, evaluation |
| Test fold 2 | 06-26 .. 07-02 | official FreshRetailNet eval week, refit, evaluation |

## 2. Contamination and leakage audit

Each item was checked in the code and in the project log (`PROJECT_PROGRESS.md`).

### 2.1 Uses of the test window

| # | Use | Contaminates? | Assessment |
|---|---|---|---|
| T1 | First full test run (forecast + simulation) | – | Intended use. |
| T2 | The traditional baseline's buffer had been chosen at the edge of its grid (1.0). After seeing the first test results, the grid was extended to 2.5 and everything rerun; the baseline then chose 1.25 on validation. | **Yes** (design change after inspecting test) | The selection itself still happened on validation, and the change made the *baseline* stronger, so it works against FreshGuard's headline. It is nonetheless a test-informed design change. |
| T3 | The iso-service comparison ("−49% waste at equal availability") was added after the first run. It interpolates the LightGBM frontier **on the test window** at the baseline's **test** availability. | **Yes** (oracle operating point) | Descriptive only: no policy could select this point in advance. Must not be presented as an achievable result. |
| T4 | Pareto frontier figure (full parameter grids on test). | Descriptive | Labelled as a trade-off chart, not used for selection. |
| T5 | Dashboard what-if page reruns tuning and simulation interactively. | Descriptive | Tuning still uses validation days; results shown are exploratory. |
| T6 | README, dashboard and methodology stated the test window was "used once" / "untouched". | Reporting error | Corrected in this release. |

### 2.2 Selection steps and the information they used

| Decision | Information used | Leak into test? |
|---|---|---|
| LightGBM hyperparameters (4-point grid) | validation window | No. Grid points differ by < 0.1 pp WAPE (32.25 – 32.34%). |
| Number of boosting rounds | early stopping on validation | No. Test folds refit with that fixed count. |
| Excluding `store_id` / `product_id` as categoricals | "development runs" (not logged which windows) | Unknown; disclosed as a possible soft contamination. |
| Best baseline (Croston-SBA), SES α | validation | No. |
| Residual quantile tables | validation residuals of the model early-stopped on those same days | No test leak, but **in-sample** for the tuning window (see 2.4). |
| Policy parameters (buffers, τ) | simulation on validation days 06-09 .. 06-18 | No test leak. |
| Demand reconstruction profiles, capacity caps, segments | training targets ≤ 06-04 | No. |

### 2.3 Temporal leakage checks

* **Features**: every history feature reads only days ≤ origin; target-day inputs are limited
  to calendar fields and planned discount/activity flags (assumed known when ordering).
  Enforced by `tests/test_features.py::test_no_future_leakage`.
* **Origin ownership**: no prediction comes from a model that saw its target. For h = 7 the
  first six test days are forecast by the validation model (origins ≤ 06-17), which was trained
  on targets ≤ 06-04, so this is not a leak.
* **Overlapping windows**: direct models for h > 1 create rows whose target windows overlap
  across origins, but every training target precedes the first served target of its fold, so
  overlap does not cross the train / evaluation boundary.
* **Reconstruction**: target-day stockout flags are used to reconstruct the *target*, never as
  a feature of the same row. Profiles are estimated on training days only.

### 2.4 In-sample calibration feeding policy tuning

The residual table that converts forecasts into quantiles is fitted on validation residuals of
the model whose number of boosting rounds was chosen on those same days, and τ is then tuned by
simulation on those days. The residuals are therefore optimistic relative to genuine
out-of-sample errors, which biases the chosen τ downward. The 0.9 quantile covered 87.5% of
test outcomes. Section 5 and `CALIBRATION_STUDY.md` measure whether this matters for decisions.

## 3. Final evaluation protocol

Because no untouched data exists, results are reported at two levels, and neither is
presented as an independent holdout.

1. **Original single-split results (retained, unchanged).** The published metrics in
   `reports/metrics/freshretailnet/` are kept as they were, labelled "test window inspected
   more than once (T2, T3)". They remain the most detailed analysis (scenarios, sensitivity,
   segments).
2. **Nested rolling-origin evaluation (new, primary evidence for generalisation).**
   Implemented in `src/freshguard/evaluation/nested.py`, run with `make nested` (also part of
   `make all`). Outputs in `reports/metrics/freshretailnet/nested/`.

### 3.1 Nested design

* **Walk-forward forecasts.** Weekly refits from 2024-05-08. For week *s*, LightGBM is trained
  on all targets before *s* with the pre-specified default hyperparameters from
  `configs/base.yaml` (`forecasting.lightgbm`), not the validation-selected grid point, so
  that no early fold depends on later data. Boosting rounds: early stopping on the last 7 days
  of the training targets, then a refit on all of them. Each model forecasts only its own week.
* **Outer folds.** Four 7-day weeks starting 06-05, 06-12, 06-19 and 06-26 (28 days, 1.4 M
  store-SKU-days). For each, the baseline buffer and FreshGuard's τ are tuned by simulated
  base-case profit on the 14 days before the week (with a 7-day warm-up before that), frozen,
  and evaluated on the week. Capacity caps use only data before the tuning warm-up.
  Baseline forecasters are selected per fold on the same 14 tuning days.
* **Quantile tables.** Three calibrations are compared in every fold (details in the
  calibration study): `insample` (the original method, primary and pre-registered as the
  headline), `static` and `rolling` (out-of-sample residuals).
* **Uncertainty.** Paired store-cluster bootstraps on the pooled 28 days (stores resampled
  with all their days, preserving within-store temporal dependence), plus per-fold results to
  show variation between weeks. Four folds are too few for a fold-level significance test, so
  none is claimed; common day-level shocks shared by all stores are not captured by the
  bootstrap, which therefore understates uncertainty about a different period.
* **What it does not fix.** The outer weeks overlap the original validation and test windows
  that were inspected during development (feature set, policy families, grids and the cost
  model were designed with them in view). The nested evaluation removes *parameter* selection
  on evaluation data; it cannot remove *design* knowledge.

### 3.2 Metrics reported

Forecasts: WAPE, bias, MAE (reconstructed demand), paired WAPE difference with store-cluster
CI. Decisions: expired units (waste), stockout events (store-SKU-days with lost demand above
0.05 units), lost units, unit fill rate, availability, average stock, simulated gross profit
in hypothetical EUR, all against the tuned traditional MA7 baseline.

## 4. Reproducing

```bash
make forecast evaluate simulate   # original protocol (unchanged)
make nested                       # nested rolling-origin evaluation + calibration study
```

## 5. Results of the nested evaluation

Source: `reports/metrics/freshretailnet/nested/` (`nested_results.json`, `forecast_by_fold.csv`,
`policy_by_fold.csv`), run on 2026-10-22 (provenance in
[`REPRODUCIBILITY_REPORT.md`](REPRODUCIBILITY_REPORT.md)). Simulated profit is in
hypothetical EUR (base scenario). All outer weeks were inspected during earlier development.

### 5.1 Forecast accuracy (walk-forward LightGBM, next day, reconstructed demand)

| Outer week | Status | LightGBM WAPE | Baseline selected on the 14 tuning days | Its WAPE | Difference (95% store-cluster CI) | LightGBM bias |
|---|---|---|---|---|---|---|
| 06-05 .. 06-11 | original validation | 33.9% | Croston-SBA | 35.2% | −1.30 pp (−1.45, −1.15) | −0.3% |
| 06-12 .. 06-18 | original validation | 30.2% | Croston-SBA | 35.0% | −4.78 pp (−4.93, −4.63) | −1.8% |
| 06-19 .. 06-25 | original test fold 1 | 30.0% | Croston-SBA | 34.5% | −4.51 pp (−4.78, −4.30) | −5.7% |
| 06-26 .. 07-02 | original test fold 2 | 31.1% | MA7 | 34.5% | −3.42 pp (−3.53, −3.30) | −5.7% |
| **Pooled, 28 days** | | **31.3%** | | **34.8%** | **−3.54 pp (−3.66, −3.42)** | |

LightGBM beats the nested-selected baseline in all 4 weeks. The walk-forward model (default
hyperparameters, weekly refits) reaches 30.0% / 31.1% on the two original test weeks, in line
with the published 30.6% for the same 14 days. Under-forecasting grows in late June in both
evaluations.

### 5.2 Replenishment decisions (pooled over the 4 outer weeks, 1.4 M store-SKU-days)

Every policy parameter below was chosen on the 14 days before each week.

| Policy | Expired units | Stockout events | Fill rate | Availability | Gross profit (hyp. €) | vs tuned baseline |
|---|---|---|---|---|---|---|
| Traditional MA7 + tuned buffer (chose +100% in all 4 weeks) | 23,633 | 192,107 | 94.0% | 86.3% | 1,281,143 | – |
| **FreshGuard, published calibration (τ = 0.9, 0.9, 0.925, 0.925)** | **20,018** | **123,540** | **96.0%** | **91.2%** | **1,317,576** | waste **−15.3%** (CI −12.9 to −17.4), stockouts **−35.7%** (−34.8 to −36.6), **+€36.4k** (34.5k to 38.3k) |
| Ablation: MA7 of reconstructed demand + tuned buffer | 23,294 | 170,479 | 95.0% | 87.8% | 1,296,497 | waste −1.4%, stockouts −11.3%, +€15.4k |
| MA7 + fixed 20% rule | – | – | – | – | – | stockouts +216%, −€164.9k |
| LightGBM point forecast, no safety stock | – | – | – | – | – | stockouts +201%, −€122.7k |

Per week, FreshGuard's net benefit was +€6.6k, +€11.5k, +€8.8k and +€9.6k (positive in all
4 weeks). Stockout events fell in all 4 weeks (−16% to −43%). **Waste did not fall in every
week**: −37%, −21%, **+18%** (06-19 week) and −15%.

### 5.3 How this compares with the original single-split result

The original protocol (test 06-19 .. 07-02, policies tuned on 06-09 .. 06-18) reported waste
−56.5%, stockout events **+12.4%** and +€13.4k over 14 days. The nested evaluation over the
same two weeks gives waste **+0.9%** (slightly more), stockout events −43.3% and +€18.4k
(sums of the two test-week rows of `policy_by_fold.csv`). The main difference is the baseline's tuned buffer: +125%
in the original 10-day tuning window, +100% in every nested 14-day window; FreshGuard's τ is
similar (0.9 vs 0.9 to 0.925).

**What is robust across both protocols and all weeks:** the forecast-accuracy gain
(≈ 3.5 pp WAPE) and a positive simulated net benefit against a tuned traditional policy.
**What is not robust:** how the benefit splits between less waste and fewer stockouts. That
split depends on where the profit-tuned operating points of both policies land in a given
tuning window, so neither "−56% waste" nor "−36% stockouts" should be quoted on its own.

### 5.4 Limitations of this evidence

* Four weekly folds from one 97-day dataset and one season; weeks are consecutive and share
  the late-June growth regime.
* The bootstrap CIs resample stores and are narrow; they describe uncertainty over stores
  for these weeks, not over future weeks. With 4 folds no fold-level test is meaningful; the
  per-week signs are reported instead.
* The outer weeks were seen during development (section 3.1).
* Demand on stockout days is reconstructed; decision metrics depend on that estimate
  (see the observed-sales sensitivity in the README).
