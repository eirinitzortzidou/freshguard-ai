# Secondary Benchmark: M5 Walmart FOODS

FreshRetailNet-50K is the primary dataset. M5 FOODS is kept as a documented fallback and a
second benchmark. It shows how the same code behaves on a different retailer **without
stock data**. Reproduce it with `make all CONFIG=configs/m5.yaml`. Outputs are in
`reports/metrics/m5/` and `reports/figures/m5/`.

**Setup:** 13,760 FOODS store-items (10 stores), 2015-01-01 to 2016-06-19. Tuning uses
2016-04-29 to 05-22, and the 28-day test window is 2016-05-23 to 06-19, in two refit folds.
Horizons are 1 and 2 days. Stockouts are **inferred** from implausible zero-sales runs, which
adds +7.5% demand; 6.6% of days are flagged. Prices are the observed USD shelf prices
converted at 0.9034 EUR/USD. Costs, shelf life (3 days) and lead time (1 day) are the same
assumptions as for FreshRetailNet. M5 has no perishable flag, so a 3-day shelf life is a
stress test, not a fact about these items.

## Forecast accuracy (test, next day)

| Method | WAPE | Bias |
|---|---|---|
| TSB (best baseline on validation) | **53.2%** | −4.6% |
| LightGBM | 53.4% | −3.4% |
| SES | 53.4% | −4.3% |
| MA28 | 53.5% | −5.1% |
| MA7 | 55.1% | −3.2% |

**Negative result:** on the test window LightGBM does not beat the validation-selected
baseline (TSB). It was better on validation (54.4% vs 57.2%), but that edge disappears on
test. Daily M5 FOODS demand at store-item level is very noisy and intermittent, so smoothing
methods are hard to beat.

## Business simulation (base economics, vs tuned MA7 with a +50% buffer)

| KPI | Change | 95% CI (store bootstrap) |
|---|---|---|
| Waste | −11.2% | −8.1 to −14.3% |
| Stockout events | −27.7% | −26.3 to −29.2% |
| Fill rate | +4.1 pp | +3.5 to +4.8 pp |
| Net benefit (EUR, assumed costs) | +€41.8k over 28 days | +€34.8k to +€48.0k |

At the baseline's availability, the LightGBM policy wastes 45.9% less. On M5 the chosen point
improves **both** waste and stockouts, unlike on FreshRetailNet. Even with no accuracy gain on
WAPE, LightGBM quantiles still help the decision. Scenarios range from +€37.1k (conservative)
to +€52.0k (optimistic). The benefit turns **negative (−€10.8k)** if observed sales are taken
as demand. Here too the result depends on the stockout assumption, and on M5 that assumption is
weaker because stockouts are inferred rather than observed.

This benchmark supports the FreshRetailNet conclusions qualitatively (forecast-driven quantile
ordering dominates the traditional frontier, and censoring matters). It should not be quoted as
evidence about fresh food.
