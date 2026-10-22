# Rerun vs committed metrics (freshretailnet, rtol=1e-06)

| File | Values compared | Differing | Largest relative differences |
|---|---|---|---|
| data_quality.json | 60 | 0 | none |
| eda_category_summary.csv | 160 | 0 | none |
| eda_stats.json | 58 | 0 | none |
| forecast_by_city_id.csv | 432 | 10 | `wape[1]` 0.3072061836719513 → 0.3072054386138916; `abs_err[1]` 165138.125 → 165137.828125; `wape[10]` 0.3763306438922882 → 0.37633004784584045 |
| forecast_by_demand_class.csv | 96 | 0 | none |
| forecast_by_first_category_id.csv | 768 | 19 | `wape[13]` 0.28445670008659363 → 0.28445741534233093; `wape[16]` 0.47965654730796814 → 0.4796557128429413; `bias[46]` 0.003408168675377965 → 0.0034081630874425173 |
| forecast_by_store_id.csv | 21552 | 0 | none |
| forecast_by_volume_class.csv | 72 | 0 | none |
| forecast_metrics.json | 1521 | 0 | none |
| forecast_overall.csv | 640 | 0 | none |
| forecast_uncensored_sales.csv | 640 | 0 | none |
| impact_by_city_id.csv | 342 | 0 | none |
| impact_by_first_category_id.csv | 608 | 0 | none |
| impact_by_store_id.csv | 17062 | 0 | none |
| sensitivity.csv | 406 | 0 | none |
| simulation_results.json | 1926 | 0 | none |
| training_report.json | 295 | 0 | none |

## Nested evaluation outputs (rtol=1e-06)

| File | Values compared | Differing | Largest relative differences |
|---|---|---|---|
| calibration_coverage.csv | 2499 | 0 | none |
| forecast_by_fold.csv | 104 | 0 | none |
| horizon_calibration.csv | 192 | 0 | none |
| nested_results.json | 1240 | 0 | none |
| policy_by_fold.csv | 392 | 0 | none |
