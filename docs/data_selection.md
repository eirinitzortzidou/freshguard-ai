# Dataset Selection

FreshGuard needs data that supports three things at once: store-SKU demand forecasting, a
credible inventory simulation for **perishable** products, and an honest treatment of
**censored demand** (zero or low sales caused by empty shelves rather than by low demand).

## Candidates

| Criterion | FreshRetailNet-50K | Corporación Favorita | M5 (Walmart) |
|---|---|---|---|
| Source | Dingdong-Inc via Hugging Face ([dataset](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K), [paper arXiv:2505.16319](https://arxiv.org/abs/2505.16319), [baseline code](https://github.com/Dingdong-Inc/frn-50k-baseline)) | Kaggle competition ([link](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)) | MOFC / Kaggle ([link](https://www.kaggle.com/c/m5-forecasting-accuracy)); public mirror used by Nixtla `datasetsforecast` |
| Public access | Yes (HF Hub, no login) | Kaggle account + competition rules acceptance | Yes (mirror) / Kaggle |
| License | CC BY 4.0 | Kaggle competition rules (non-commercial; no redistribution) | M5 competition terms (research/education); not redistributed here |
| Store ids | 898 stores, 18 cities | 54 stores | 10 stores, 3 states |
| Product ids | 865 products, 50,000 store-SKU series | ~4,100 items | 3,049 items (1,437 FOODS) |
| History | 90 days train + 7 days eval (2024-03-28 .. 2024-07-02), daily **and hourly** | 2013-2017 daily | 2011-2016 daily |
| Perishable coverage | Entire dataset is fresh retail (fresh e-grocery front warehouses) | `perishable` flag + families (PRODUCE, BREAD/BAKERY, MEATS ...) | No perishable flag; FOODS departments anonymised |
| Category labels | Anonymised ids (management group, 3 category levels) | Named families | Anonymised (FOODS_1/2/3) |
| Inventory / stockout info | **Hourly stock status (16 opening hours)** | None | None |
| Prices | Discount ratio only | None (oil price, promotions) | Weekly shelf prices (USD) |
| Promotions | Discount, activity flag | `onpromotion` | Price changes, SNAP days |
| Calendar / external | Holiday flag, weather (precipitation, temperature, humidity, wind) | Holidays, oil | Events, SNAP |
| Demand censoring | Directly observable via stock flags | Not observable | Not observable (must be inferred) |
| Forecasting suitability | Good; short history (weekly seasonality only) | Very good (long history) | Very good |
| Inventory-simulation suitability | **Best**: censoring known, fresh assortment | Medium (perishable flag, no stock data) | Weak (no stock data, no shelf-life context) |
| Compute (laptop, CPU) | 4.85M rows; fits in 16 GB | 125M rows; needs subsetting | 7.4M rows (FOODS, 2015-16) |

## Decision

**Primary dataset: FreshRetailNet-50K.** It is the only public candidate that (a) is entirely
fresh retail and (b) annotates intraday stockouts, so the gap between observed sales and true
demand can be estimated rather than assumed. Its CC BY 4.0 license allows use with
attribution. Its main limitations are the short 97-day history and anonymised categories.

**Fallback / secondary benchmark: M5 FOODS** (`configs/m5.yaml`). Implemented end to end
(loader, zero-run stockout inference, observed USD prices converted to EUR) because, during
the build, the execution environment could not reach Hugging Face. It is weaker for the
business question: no stock data, no shelf-life context and no fresh-product labels.

Favorita was not used: it requires Kaggle authentication and acceptance of competition
rules, has no stock information, and unofficial mirrors would breach its terms.

## Access log (what actually happened)

1. `huggingface.co`, `*.hf.co` and `hf-mirror.com` returned HTTP 403 from the build
   environment's egress proxy; a helper session in a reconfigured environment hit the same
   block.
2. M5 was downloaded from the public Nixtla mirror
   (`raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip`) and ingested.
3. The FreshRetailNet-50K maintainer files `train.parquet` (4,500,000 rows, 106 MB) and
   `eval.parquet` (350,000 rows, 8.4 MB) were then uploaded manually by the project owner.
   SHA-256: train `6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d`,
   eval `1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e`.
   Both match the published schema (validated by `freshguard.data.schema`).

## Manual import procedure (reproducible)

```bash
# Option A: automatic (needs huggingface.co reachable)
make data                                   # calls huggingface_hub.snapshot_download

# Option B: manual
#   download data/train-*.parquet and data/eval-*.parquet from
#   https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/tree/main
mkdir -p data/raw/freshretailnet
cp ~/Downloads/train*.parquet ~/Downloads/eval*.parquet data/raw/freshretailnet/
make data                                   # finds the files and skips the download
```

Files are matched by name (`*train*.parquet`, `*eval*.parquet`), schema-checked, and never
committed (`data/raw/` is git-ignored).

## What is real and what is assumed

| Item | Status |
|---|---|
| Daily and hourly sales, stock status, discount, holiday/activity flags, weather | Real (dataset) |
| Store / product / category identifiers | Real but anonymised: reported as ids only. **No category is labelled fruit, vegetables, meat or bakery.** |
| Units of `sale_amount` | Real but normalised by the publisher (0.1 granularity); treated as "units" |
| Latent demand on stockout days | Estimated (see `docs/methodology.md`), with sensitivity bounds |
| Prices, unit costs, margins, disposal, holding, goodwill costs | **Assumed** (hypothetical EUR scenarios, `configs/base.yaml`) |
| Shelf life, lead time, review period, capacity | **Assumed** operating setting, varied in sensitivity analysis |
| Inventory levels, deliveries, waste | **Simulated**; the dataset contains none of these |
