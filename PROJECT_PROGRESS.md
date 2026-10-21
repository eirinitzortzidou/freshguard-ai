# PROJECT_PROGRESS

Living log so work can resume after an interruption. Newest state first.

## Status (2026-10-20)

| Phase | State | Evidence |
|---|---|---|
| 1 Research & feasibility | done | `docs/data_selection.md`, `docs/research_review.md` |
| 2 Repository & data pipeline | done | `freshguard data`; `reports/data_quality.md` |
| 3 EDA & features | done | `reports/eda_report.md`, `reports/figures/freshretailnet/01-09`, leakage tests |
| 4 Forecasting | done | `reports/metrics/freshretailnet/forecast_*.csv`, `forecast_metrics.json` |
| 5 Inventory optimisation | done | `inventory/`, `tests/test_simulator.py` |
| 6 Business simulation | done | `simulation_results.json`, `sensitivity.csv`, `impact_by_*.csv`, figures 10-15 |
| 7 Dashboard | done | `dashboard/`, screenshots in `reports/figures/dashboard/` |
| 8 Engineering quality | done | 42 tests pass; ruff and mypy clean; CI workflow |
| 9 Documentation | done | README, `docs/*`, `reports/eda_report.md` |
| 10 Final audit | done for FRN | numbers in README cross-checked with metrics files |
| M5 secondary benchmark | done | `docs/m5_benchmark.md`: WAPE 53.4% vs TSB 53.2% (no gain, negative result); waste −11.2%, stockouts −27.7%, +€41.8k |

## Headline results (FRN test window 2024-06-19..07-02, measured)
- Next-day WAPE: LightGBM 30.6% vs Croston-SBA 34.3% (validation-selected baseline), MA7 34.0%.
- Base case vs tuned MA7 (+125% buffer): waste −56.5%, stockout events +12.4% (worse),
  fill rate 95.9% → 95.1%, net benefit +€13,373 (hypothetical, CI 11,806-14,681).
- At equal availability: −49.2% waste, +€14.5k. LightGBM 95% policy: waste −15.6%, stockouts −37.7%.
- Negative results: observed-sales demand −€3.9k; noise σ = 0.5 −€1.0k; newsvendor fractile −€140.6k.

## Decisions
- **Dataset**: FreshRetailNet-50K (CC BY 4.0), files uploaded by Eirini on 2026-10-20 (Hugging Face blocked
  by the environment's network policy). sha256 train `6706832d…5a3d`, eval `1b118840…960e`.
  M5 FOODS is implemented as a documented fallback and secondary benchmark (`configs/m5.yaml`).
- **Splits** (FRN): train targets <= 2024-06-04; validation 06-05..06-18; test fold 1 06-19..06-25,
  fold 2 = official eval week 06-26..07-02. Expanding-window refits per fold.
- **Target**: latent demand reconstructed from hourly stockout flags (intraday-profile ratio
  estimator, half weight for stockout hours that still recorded sales); sensitivity runs use
  observed sales and a high-demand variant.
- **Model**: global LightGBM (tweedie, early stopping on L2), one direct model per horizon
  (1, 2, 3, 7); 255 leaves / min leaf 400 chosen on validation. `store_id`/`product_id` excluded.
- **Inventory**: daily review, lead time 1 day, shelf life 3 days (base); FEFO lost-sales
  simulator; policies tuned on validation days 06-09..06-18 by gross profit; stockout event =
  lost demand > 0.05.
- **Finance**: hypothetical EUR (FRN has no prices); scenarios in `configs/base.yaml`.

## Protocol notes (disclosed)
- After the first test run, the baseline buffer had been tuned at the edge of its grid (1.0), so
  the grid was extended to 2.5 and everything rerun; the baseline then chose 1.25. The iso-service
  comparison was also added after that first run. All parameter choices are still made on
  validation only, but the test window has been seen more than once.
- `holiday_flag` mostly marks weekends (82% of flagged days are Sat/Sun); the weekday index uses all days.

## Blockers
- huggingface.co not on the environment allowlist (worked around via manual upload).
- No GitHub repo created yet: waiting for Eirini's explicit approval to publish.

## Next steps
- Publish to GitHub once approved.
- Optional: rolling quantile recalibration to fix test under-coverage; category-specific parameters.
