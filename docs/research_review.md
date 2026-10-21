# Research Review and Operational Setting

This note summarises the literature that shaped FreshGuard's design and states the
operational setting, separating **dataset facts** from **modelling assumptions**.

## 1. Why the problem matters

Food waste is large at every stage of the chain; UNEP's *Food Waste Index Report 2024*
estimates about 1.05 billion tonnes of food waste in 2022 across households, food service and
retail. For a grocer, fresh categories combine short shelf lives with volatile demand, so the
ordering decision trades off spoilage (overage) against empty shelves (underage) every day
(van Donselaar et al., 2006; Fildes, Ma & Kolassa, 2022).

## 2. Methods reviewed and how they were used

| Topic | Key references | Decision in FreshGuard |
|---|---|---|
| Retail forecasting practice | Fildes, Ma & Kolassa (2022); Hyndman & Athanasopoulos (2021) | Daily store-SKU forecasts, simple benchmarks first, rolling-origin evaluation (Tashman, 2000). |
| Global tree models | Ke et al. (2017) LightGBM; Januschowski et al. (2022); Makridakis, Spiliotis & Assimakopoulos (2022) M5 results | One global LightGBM per horizon with lag/rolling/calendar/promotion features; tweedie loss for non-negative skewed demand. Deep learning not used: 90 days of history and CPU budget; the M5 winners were predominantly LightGBM-based. |
| Intermittent demand | Croston (1972); Syntetos & Boylan (2005); Syntetos, Boylan & Croston (2005); Teunter, Syntetos & Babai (2011) | Croston-SBA and TSB implemented as baselines; ADI/CV² classification used to segment errors. FreshRetailNet is mostly smooth (88% of series), so these are benchmarks rather than the primary model. |
| Accuracy metrics | Hyndman & Koehler (2006) | MAE, RMSE, WAPE (zero-safe), weighted bias; no MAPE (undefined at zero). |
| Censored demand / stockouts | Nahmias (1994); Conlon & Mortimer (2013); Wang et al. (2025, FreshRetailNet-50K) | Latent demand reconstructed from **hourly** stock flags with an intraday-profile ratio estimator; results re-run on observed sales (lower bound) and a high-demand variant to bound the impact of the estimate. |
| Newsvendor | Arrow, Harris & Marschak (1951); Qin et al. (2011); Ban & Rudin (2019) | Critical fractile Cu/(Cu+Co) implemented and reported, but treated as a single-period approximation. |
| Lost-sales, lead times | Bijvank & Vis (2011); Huh et al. (2009) | Order-up-to policies on the inventory position are near-optimal in lost-sales systems; FreshGuard uses forecast-driven order-up-to levels with an inventory position that removes projected expiries. |
| Perishable inventory | Nahmias (1982, 2011); Broekmeulen & van Donselaar (2009) | Age-tracked FEFO simulation with fixed shelf life; order quantity subtracts stock projected to expire before the delivery is used (in the spirit of the EWA heuristic). |
| Predictive intervals | Vovk, Gammerman & Shafer (2005); Angelopoulos & Bates (2021) | Mondrian split-conformal style empirical residual quantiles by forecast-level bin, fitted on validation, checked for coverage on test. |
| Cost-sensitive decisions | Ban & Rudin (2019) | The decision parameter (service quantile τ) is tuned by simulated profit on validation instead of forecasting accuracy. |

## 3. Selected approach (summary)

1. Reconstruct latent daily demand from stockout-censored sales (hourly flags).
2. Forecast it with a global LightGBM per horizon; compare with 9 baselines on identical rows.
3. Convert forecasts into demand quantiles with validation residuals.
4. Turn quantiles into order-up-to levels; simulate a lost-sales, FEFO, fixed-shelf-life
   system chronologically, with orders arriving after the lead time.
5. Tune each policy's single parameter on the validation window by gross profit; evaluate
   once on the test window; quantify waste, stockouts, service and EUR impact with
   store-cluster bootstrap intervals and sensitivity analysis.

## 4. Operational setting

| Element | Value | Real or assumed |
|---|---|---|
| Store | 898 front-warehouse stores in 18 cities | Real ids (anonymised) |
| SKU | 50,000 store-product series, 865 products | Real ids (anonymised) |
| Product category | 7 management groups; 32 / 84 / 233 category ids | Real ids, **no names**; results reported by id |
| Data frequency | Daily (hourly sales and stock flags inside) | Real |
| Ordering frequency | Daily review at store close (R = 1) | Assumed (typical for fresh) |
| Supplier lead time | 1 day (base); 2 and 3 days in sensitivity | Assumed |
| Forecast horizon | Next day primary (= lead time); 2, 3, 7 days evaluated | Derived from lead time |
| Shelf life | 3 sellable days (base); 1-7 days in sensitivity | Assumed, uniform across SKUs |
| Inventory capacity | 5 x max historical daily demand per SKU | Assumed |
| Ordering constraints | Continuous quantities (units are normalised); pack-size rounding implemented and tested but off | Assumed |
| Service-level requirement | None imposed; τ chosen by profit; 95% service policy shown for comparison; Pareto frontier for constraints | Design choice |
| Prices and costs | Hypothetical EUR scenarios (no prices in data) | Assumed |

## 5. References

- Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.
- Arrow, K. J., Harris, T., & Marschak, J. (1951). Optimal inventory policy. *Econometrica*, 19(3), 250-272.
- Ban, G.-Y., & Rudin, C. (2019). The big data newsvendor: Practical insights from machine learning. *Operations Research*, 67(1), 90-108.
- Bijvank, M., & Vis, I. F. A. (2011). Lost-sales inventory theory: A review. *European Journal of Operational Research*, 215(1), 1-13.
- Broekmeulen, R. A. C. M., & van Donselaar, K. H. (2009). A heuristic to manage perishable inventory with batch ordering, positive lead-times, and time-varying demand. *Computers & Operations Research*, 36(11), 3013-3018.
- Conlon, C. T., & Mortimer, J. H. (2013). Demand estimation under incomplete product availability. *American Economic Journal: Microeconomics*, 5(4), 1-30.
- Croston, J. D. (1972). Forecasting and stock control for intermittent demands. *Operational Research Quarterly*, 23(3), 289-303.
- Fildes, R., Ma, S., & Kolassa, S. (2022). Retail forecasting: Research and practice. *International Journal of Forecasting*, 38(4), 1283-1318.
- Huh, W. T., Janakiraman, G., Muckstadt, J. A., & Rusmevichientong, P. (2009). Asymptotic optimality of order-up-to policies in lost sales inventory systems. *Management Science*, 55(3), 404-420.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
- Januschowski, T., Wang, Y., Torkkola, K., Erkkilä, T., Hasson, H., & Gasthaus, J. (2022). Forecasting with trees. *International Journal of Forecasting*, 38(4), 1473-1481.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting*, 38(4), 1346-1364.
- Nahmias, S. (1982). Perishable inventory theory: A review. *Operations Research*, 30(4), 680-708.
- Nahmias, S. (1994). Demand estimation in lost sales inventory systems. *Naval Research Logistics*, 41(6), 739-757.
- Nahmias, S. (2011). *Perishable Inventory Systems*. Springer.
- Qin, Y., Wang, R., Vakharia, A. J., Chen, Y., & Seref, M. M. H. (2011). The newsvendor problem: Review and directions for future research. *European Journal of Operational Research*, 213(2), 361-374.
- Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2), 303-314.
- Syntetos, A. A., Boylan, J. E., & Croston, J. D. (2005). On the categorization of demand patterns. *Journal of the Operational Research Society*, 56(5), 495-503.
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting*, 16(4), 437-450.
- Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent demand: Linking forecasting to inventory obsolescence. *European Journal of Operational Research*, 214(3), 606-615.
- UNEP (2024). *Food Waste Index Report 2024*. United Nations Environment Programme.
- van Donselaar, K., van Woensel, T., Broekmeulen, R., & Fransoo, J. (2006). Inventory control of perishables in supermarkets. *International Journal of Production Economics*, 104(2), 462-472.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
- Wang, Y., Gu, J., Long, L., Li, X., Shen, L., Fu, Z., Zhou, X., & Jiang, X. (2025). FreshRetailNet-50K: A stockout-annotated censored demand dataset for latent demand recovery and forecasting in fresh retail. arXiv:2505.16319.

*Literature was not re-fetched during the build (the environment blocked most academic
hosts); references are standard works cited from the author's knowledge and should be
spot-checked before publication.*
