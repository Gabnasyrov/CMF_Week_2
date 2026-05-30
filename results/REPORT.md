# Weekly baseline — report summary

## Goal

Build a **simple baseline** trade filter on Binance maker flow using liquidation context (Binance + Bybit), evaluate **Score**, **PnL_kept**, **PnL_filtered**, **turnover/day**, with **≥500k USD/day** kept turnover.

## Data

- **Period:** ~6 months (train Dec 2025 – Jan 2026, validation Feb 2026)
- **Loader:** Polars `scan_parquet` (`week_baseline/lib/load_data.py`)
- **Symbols:** BTCUSDT, ETHUSDT perpetuals
- **Streams:** Binance trades, BBO, liquidations; Bybit liquidations (`timestamp` μs UTC; Bybit +200 ms before join)
- Parquet **not included in git**

## Classifiers

| ID | Type | Description |
|----|------|-------------|
| `baseline_keep_all` | — | No filter; Score = 0 |
| `heuristic_bybit_flow` | Heuristic | Filter when \|Bybit L_net_30s\| in top quantile |
| `ml_logistic_toxic` | ML | Logistic regression on liq features; threshold tuned for turnover |
| `lgbm_nk_30s` | ML (full) | LightGBM from `tz_assignment` — best production baseline |

## Direction forecast horizons

LGBM no-K is trained at **h ∈ {1, 3, 5, 30, 300}s**. Hit rate @ **5s** exceeds @ **30s**:

| h | BTC hit | ETH hit |
|---|---------|---------|
| **5s** | **0.638** | **0.587** |
| 30s | 0.560 | 0.544 |

There is **no 10s** model (nearest: 5s and 30s). Bundled **PnL Score** uses **30s** filter signal; markout τ is separate. Details: `docs/FORECAST_HORIZONS.md`, `results/direction_horizons_reference.json`.

## Key results (validation, Binance trades, τ=30s)

From `baseline_metrics_full.csv` (PnL in **bps**, turnover in **USD/day**):

| Symbol | Strategy | Score (bps) | PnL_kept (bps) | PnL_filtered (bps) | Turnover kept (USD/day) | OK |
|--------|----------|-------------|----------------|--------------------|-------------------------|-----|
| BTC | lgbm_nk_30s | **+0.065** | −0.108 | −0.236 | **$5.98B** | ✓ |
| BTC | baseline | 0.000 | −0.173 | — | $12.16B | ✓ |
| ETH | lgbm_nk_30s | **+0.218** | +0.238 | −0.213 | **$6.32B** | ✓ |
| ETH | baseline | 0.000 | +0.020 | — | $12.21B | ✓ |

**Turnover is always positive.** Minus in `PnL_filtered` is normal: filtered trades are the toxic ones we remove.

**Turnover constraint (≥$500k/day):** met for all rows (kept turnover is billions, not millions).

## Deliverables

- Notebook: `notebooks/Weekly_Baseline_Report.ipynb`
- Code: `week_baseline/lib/`
- Run: `python scripts/run_baseline.py`

## Next steps

- Extend Polars pipeline to full trade universe (chunked by day)
- Add cross-impact features from `research/cascade_alpha_v2/`
- Tune ML threshold on validation only (avoid test leakage)
- Optional: PnL backtest with `lgbm_nk_5s` filter (higher direction WR, not in current CSV)
