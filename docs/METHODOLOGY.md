# Methodology & known limitations

## Reproducibility

```bash
pip install -e ".[dev]"
export LIQUIDATION_DATA_ROOT=/path/to/parquet   # needs raw + data/enriched/
python -m cmf_week2.cli.run_all
# or
cmf-run-baseline --all-symbols
pytest
```

Package name: **`cmf_week2`** (repo folder may be `CMF_Week_2`).

## Two metric tables

| File | Split | Content |
|------|-------|---------|
| `baseline_metrics.csv` | **Official calendar**: train Dec–Jan, test Feb | Polars heuristic + logistic baselines |
| `baseline_metrics_full.csv` | **50/50 chronological** (liquidation_task) | LGBM / Kalman PnL from full pipeline |

Do not mix splits when comparing Score.

## Why `n_days` differs (BTC ~10 vs ETH ~46 in full CSV)

`baseline_metrics_full.csv` uses the **second half** of the loaded sample (`train_50_test_50`). Effective test calendar span after markout label drop depends on symbol-specific trade density and how many trailing trades lose τ=300s labels. BTC test window spans ~10 days; ETH ~46 days — not a bug in turnover, but **heterogeneous test windows**. Prefer official Feb validation (`baseline_metrics.csv`) for apples-to-apples comparison.

## Fixes in v0.2

1. **Imports** — installable package `cmf_week2`, no `week_baseline` dependency.
2. **Heuristic leakage** — quantile threshold tuned on **train** only.
3. **Liq features** — 30s rolling on **dense 1s grid** (zero-filled).
4. **Split** — calendar train/test + **300s purge** before split.
5. **Markout** — asof on **quote timestamps**, stale quote >5s → NaN.
6. **Stats** — bootstrap SE, daily score mean/std in `baseline_metrics.csv`.
7. **LGBM bundle** — vendored under `cmf_week2/strategies_bundle/`.
8. **AUC vs hit rate** — README uses **hit_rate_test** for direction quality; **auc_test** is ROC-AUC (see `lgbm_direction_metrics.json`).

## Not yet modeled (quant reviewer)

- Queue priority, fill probability, cancels, funding
- Purged cross-validation beyond single purge
- Full PnL for `lgbm_nk_5s` filter (direction WR higher, trade filter not re-run)
