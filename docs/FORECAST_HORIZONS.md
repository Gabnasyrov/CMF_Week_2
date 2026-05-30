# LGBM direction forecast — forecast horizons

**Target:** `sign(kalman_mu[t+h] − kalman_mu[t])` on 1s Binance panel.  
**Model:** LGBM without Kalman features (`lgbm_no_kalman`), same 21 inputs as @ 30s.  
**Split:** 40% train / 60% test (~90 days). Source: `research/price_forecast/results/tables/qf_results_*.json`.

## Hit rate by forecast horizon h

| h (s) | BTC hit rate | ETH hit rate | vs 30s (BTC) |
|-------|--------------|--------------|--------------|
| 1 | 0.704 | 0.622 | +14.4 pp |
| 3 | 0.665 | 0.603 | +10.5 pp |
| **5** | **0.638** | **0.587** | **+7.8 pp** |
| 30 | 0.560 | 0.544 | — |
| 300 | 0.512 | 0.516 | −4.8 pp |

Bundled ROC/AUC in this repo: **30s only** (`results/lgbm_direction_metrics.json`).

**Note on 10s:** the training grid is `{1, 3, 5, 30, 300}` — there is **no separate 10s model**. The closest trained horizons are **5s** (above) and **30s** (below).

Short horizons (1–5s) often show higher hit rate, partly from **microstructure noise** on 1s bars; for maker markout the production filter uses **30s** (and optionally **300s**) signals.

## Direction quality vs trade-filter PnL

Higher hit rate at 5s **does not automatically** mean better **Score** in the task metrics:

| Layer | What is evaluated | Horizons in deliverable |
|-------|-------------------|-------------------------|
| Direction model | Hit rate / AUC vs Kalman move | 1, 3, 5, 30, 300s (reference table above) |
| Trade filter → PnL | Score, PnL_kept @ markout τ | **lgbm_nk_30s**, **lgbm_nk_300s** only (`results/baseline_metrics_full.csv`) |

Markout horizons τ ∈ {30, 120, 300}s are **independent** of forecast horizon h.

**Why 30s for PnL report:** balances signal stability and alignment with maker markout; 5s filter PnL was not run in the bundled pipeline (would need `research/pnl_evaluation` with `lgbm_nk_5s` strategy).

## Regenerate direction metrics @ 30s

```bash
export LIQUIDATION_DATA_ROOT=/path/to/parquet
python scripts/plot_lgbm_auc.py
```

Reference multi-horizon hit rates: `results/direction_horizons_reference.json`.
