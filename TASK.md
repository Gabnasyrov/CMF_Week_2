# Task description (CMF Week 2)

## Goal

Create a simple baseline for filtering Binance perpetual trades:

1. Load data with **Polars** (6-month archive)
2. Build a trade classifier (ML or heuristics)
3. Report **Score**, **PnL_kept**, **PnL_filtered**, **turnover per day**
4. Ensure **turnover constraint ≥ 500,000 USD/day** on kept trades

## Data

Public dataset: [Google Drive](https://drive.google.com/file/d/1XmxRsElei-vE8Gc5tkKs2wH4FJVRTevS/view)

Streams (parquet, `timestamp` in microseconds UTC):

- Binance: trades, book tickers, liquidations (`btcusdt`, `ethusdt`)
- Bybit: liquidations

## Markout

Horizons τ ∈ {30, 120, 300} seconds.

Maker PnL (bps):

```
pnl_i(τ) = -s_i * (m_i(τ) - p_i) / p_i * 10_000 + 0.5
```

where `s_i = +1` for taker buy (maker sell), `w_i = min(notional, 100k USD)`.

## Filter & metrics

Binary filter `f_i ∈ {0,1}`: `0` = keep, `1` = filter out.

```
PnL_all   = weighted mean markout (all trades)
PnL_kept  = weighted mean on f=0
PnL_filtered = weighted mean on f=1
Score     = PnL_kept - PnL_all
Turnover/day = sum(w * (1-f)) / n_days  >= 500_000 USD
```

## Split

- Train: Dec 2025 – Jan 2026
- Validation: Feb 2026

## Deliverables in this repo

| Artifact | Path |
|----------|------|
| Report notebook | `notebooks/Weekly_Baseline_Report.ipynb` |
| Polars pipeline | `lib/`, `scripts/run_baseline.py` |
| Full metrics CSV | `results/baseline_metrics_full.csv` |
| Summary | `results/REPORT.md`, `README.md` |
