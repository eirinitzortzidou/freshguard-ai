# Methodology

Code references are given as `module:function`. Every number in the README is produced by
the pipeline described here (`make all`). The evaluation protocol and contamination audit are
in [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md); the calibration analysis in
[`CALIBRATION_STUDY.md`](CALIBRATION_STUDY.md).

## 1. Data pipeline

`freshguard.data.ingest:prepare_dataset`

1. **Ingestion**: Parquet files are read with Polars; schema is checked against
   `freshguard.data.schema.RAW_SCHEMA` (19 columns, dtype families) and fails loudly on
   mismatch.
2. **Normalisation**: dates parsed, numeric columns downcast (Int16/Int32/Float32), rows sorted
   by `(store_id, product_id, dt)`, hourly lists split into aligned `(rows, 24)` NumPy arrays.
3. **Quality checks** (`freshguard.data.validate:quality_report`): duplicate keys, nulls/NaN,
   temporal gaps, series length, negative sales, hourly-vs-daily totals, stock-count
   consistency, discount range, sales during stockout hours, robust outliers (flagged, never
   removed), promotion prevalence. Output: `reports/data_quality.md`.
   FreshRetailNet-50K passes all integrity checks (0 duplicates, 0 gaps, hourly sums equal
   daily totals, stock counts consistent). Flagged but kept: 17,339 rows with a discount ratio
   of 0 or above 1.0, 7,973 robust outliers, and 831,726 hour cells marked out of stock that
   still recorded sales (the flag is hourly, so stock can run out within the hour).
4. **Panel**: one row per store-SKU-day, written to `data/processed/<dataset>/panel.parquet`.

## 2. Censored demand reconstruction

`freshguard.data.demand:reconstruct_demand`

FreshRetailNet-50K annotates, for each of the 16 opening hours (06:00-22:00), whether the
product was out of stock. 44% of store-SKU-days have at least one stockout hour; 19.8% of all
opening hours are out of stock. Observed sales are therefore a *lower bound* on demand.

* For each series an intraday demand profile `p_h` (share of daily sales per hour) is
  estimated from fully in-stock **training** days, shrunk towards its third-level category
  profile with 20 pseudo-days.
* For a censored day, the expected share of demand in in-stock hours is
  `s = 1 - Σ_h w_h p_h`, where `w_h = 1` for stockout hours without sales and `0.5` for
  stockout hours that still recorded sales.
* If `s >= 0.25`: `demand = sales / s`, capped at 4 × sales.
  Otherwise (almost the whole day out of stock): `demand = sales + (1 - s) × level`, where
  `level` is the series' mean daily sales on fully in-stock training days.
* Uncensored days keep `demand = sales`.

On the synthetic fixture with known true demand, this cuts the absolute error on censored days
by more than 30% relative to using sales (`tests/test_data.py`). On FreshRetailNet it adds
26.1% to total units (4.91M observed → 6.19M estimated). The estimate is uncertain, so the
business simulation is repeated with (a) observed sales as demand (no lost demand at all,
an extreme lower bound) and (b) 1.5 × the estimated lost demand.

The M5 fallback has no stock data; there, stockouts are inferred from zero-sales runs that
are implausible under a Poisson rate estimated over the 28 previous days
(`zero_run_reconstruction`), with a causal variant used for features.

## 3. Forecasting problem design

* **Unit**: store-SKU-day; target = reconstructed demand of day `t + h`, forecast at the
  close of day `t`.
* **Horizon**: the order placed at the close of day `t` arrives before opening on `t + L`
  (L = 1 in the base case) and must cover that day, so `h = L = 1` is primary. Horizons 2, 3
  and 7 are trained and evaluated (2 and 3 feed the lead-time sensitivity). The horizon was
  fixed by the operating setting before seeing any accuracy numbers.
* **Splits** (strictly chronological, FreshRetailNet dates):
  training targets 2024-03-28..06-04 (origins from 04-24 so that 28-day windows are complete),
  validation 06-05..06-18 (all tuning and selection), test 06-19..07-02 in two
  expanding-window folds; fold 2 is the official FreshRetailNet-50K evaluation week.
* **Origin ownership**: the validation model serves origins up to 06-17; fold *k* is served by a
  model refit on all targets before the fold start. No prediction ever comes from a model that
  saw its target.

## 4. Features (`freshguard.features.build`)

| Group | Features |
|---|---|
| Demand history | lags 1-7, 14, 21, 28; rolling mean/std (3, 7, 14, 28), rolling median/max (7, 28); trend 7/28; CV 28 |
| Weekly seasonality | demand on the same weekday as the target, 1-4 weeks back (only if ≤ origin), their mean |
| Intermittency | zero-sales share 7/28; days since last sale |
| Stockouts | stockout hours on the origin day and last 7 days; censored-day share 7/28; in-stock share; lost-demand share 7 |
| Promotions / price | discount at origin, 7-day mean; activity share 7 |
| Known at ordering | target-day discount, discount relative to recent mean, activity flag, holiday flag, weekday, day of month |
| Weather | precipitation and temperature **of the origin day** only (target-day weather is not known) |
| Hierarchy | city, management group, category ids (categorical). Store and product ids are *excluded* as categoricals (in development runs they caused overfitting after <100 boosting rounds) |

Leakage tests (`tests/test_features.py::test_no_future_leakage`) perturb every value after the
origin and assert that only the target and the declared known-in-advance columns change.
Assumption: discounts and activity flags are planned by the retailer and known when ordering.

## 5. Models

* **Baselines** (`forecasting.baselines`): naive, seasonal naive, 7- and 28-day moving
  averages, 4-week same-weekday mean, 28-day mean × weekday index, SES (α tuned on
  validation), Croston-SBA, TSB. All computed on the same causal demand signal.
* **LightGBM** (`forecasting.gbm`): global model, one per horizon; tweedie objective
  (power 1.2), early stopping on validation L2. A 4-point grid (leaf count / minimum leaf size /
  tweedie vs L2) was evaluated on validation; differences were tiny (WAPE 32.25-32.34%) and
  the 255-leaf tweedie model was selected. Test folds refit with the validation-selected number
  of rounds. Seeds fixed and `deterministic=True`.
* **Predictive quantiles** (`forecasting.intervals`): validation residuals binned by forecast
  level (12 bins); quantile = forecast + residual quantile of its bin. Applied unchanged to test.
* Not used: CatBoost/XGBoost (no expected gain over LightGBM for this tabular setup, extra
  dependency), deep learning (90 days of history, CPU-only budget; the FreshRetailNet authors'
  TFT/DLinear baselines need a GPU), reinforcement learning (no justification over a tuned
  base-stock policy for daily lost-sales replenishment).

## 6. Replenishment policies (`inventory.policies`)

All policies output an **order-up-to level** for the arrival day; the simulator turns it into
an order quantity `q = max(0, min(S, capacity) - projected usable stock)`.

| Policy | Order-up-to level S | Tuned on validation |
|---|---|---|
| Traditional naive | (1 + b) × yesterday's sales | buffer b |
| Traditional seasonal | (1 + b) × sales on the same weekday last week | b |
| **Traditional MA7 (baseline)** | (1 + b) × 7-day mean of observed sales | b |
| Traditional MA7 rule-of-thumb | 1.2 × 7-day mean of observed sales | fixed |
| Ablation: MA7 on reconstructed demand | (1 + b) × 7-day mean of reconstructed demand | b |
| LightGBM point | forecast | none |
| LightGBM 95% service | forecast + 1.645 σ(bin) | fixed |
| LightGBM newsvendor | τ-quantile, τ = Cu / (Cu + Co) | fixed by costs |
| **LightGBM cost-optimal (FreshGuard)** | τ-quantile of demand | τ |

Newsvendor costs: Cu = price − cost + goodwill, Co = cost + disposal. With a 3-day shelf life a
unit unsold today can still be sold tomorrow, so the single-period fractile overstates the
overage cost; the cost-optimal policy instead chooses τ by simulated profit.

## 7. Simulation (`inventory.simulator`)

Chronological, vectorised over all 50,000 store-SKUs. State: stock by remaining shelf life and
an order pipeline. Each day: arrivals (full shelf life) → FEFO sales up to available stock,
unmet demand lost → expiry of units at zero remaining life → ageing → order at close. Orders
arrive exactly `L` days later. For `L > 1` the inventory position projects current stock and
open orders forward with the policy's own point forecasts and removes units that would expire
before the arrival day.

Invariants are unit-tested (`tests/test_simulator.py`): non-negative stock, sales ≤ stock and
≤ demand, day-level and global conservation, exact expiry timing, FEFO order, exact lead
times, pack-size rounding, capacity cap, determinism.

**Common conditions**: every policy uses the same realised demand, prices, costs, capacity
(5 × max historical daily demand), lead time, shelf life and initial stock (7-day mean demand
before the warm-up, fresh). Seven warm-up days are simulated and excluded from KPIs.

**Tuning and testing**: each policy's parameter is chosen on validation days 06-09..06-18 by
base-case gross profit; the test window 06-19..07-02 is simulated with frozen choices. The
test window was inspected more than once during development, and a nested rolling-origin
evaluation complements it: see [`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).

## 8. KPIs and financial accounting (`finance.economics`)

* Waste = expired units; waste rate = expired / received.
* **Stockout event** = a store-SKU-day with lost demand > 0.05 units (half the smallest sales
  increment). Availability = 1 − events / store-SKU-days. Fill rate = sold / demand.
* Waste reduction % = (baseline − AI) / baseline × 100; stockout reduction likewise.
* Gross profit = revenue − cost of goods consumed (sold + expired) − disposal − holding −
  ordering. Valuing consumed rather than purchased units neutralises end-of-window stock
  differences. The profit difference decomposes **exactly** into recovered-sales margin,
  avoided waste at cost, avoided disposal, holding and ordering effects (tested), so nothing is
  double-counted. Goodwill (non-cash lost-sales penalty) is reported separately.
* Currency: all values in EUR. FreshRetailNet has no prices, so a hypothetical EUR price per
  normalised unit is assumed (2.00 / 2.50 / 3.00 in the three scenarios). The M5 path converts
  observed USD prices at 0.9034 EUR/USD (ECB 2016 average).

## 9. Uncertainty and sensitivity

* Forecast comparison: store-cluster paired bootstrap of the WAPE difference; daily wins.
* Business impact: store-cluster paired bootstrap (500 resamples) of waste and stockout
  reductions, fill-rate change and net benefit; series-level bootstrap as a check.
* Scenarios: conservative / base / optimistic economics, policies re-tuned per scenario.
* Sensitivity (policies re-tuned each time): shelf life 1-7 days, lead time 1-3 days, demand
  definition (observed sales / reconstructed / high), multiplicative demand noise
  (σ = 0.25, 0.5) that forecasts cannot anticipate.
