# CMF Week 2 — Liquidation trade filter baseline

**Course task:** build a simple baseline that filters Binance maker trades using liquidation context, evaluate **Score**, **PnL_kept**, **PnL_filtered**, **turnover/day**, with **≥500k USD/day** kept turnover.

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Weekly_Baseline_Report.ipynb
```

The notebook runs **without raw data** using bundled results in `results/`.

## Data (6 months)

Download parquet from Google Drive:  
https://drive.google.com/file/d/1XmxRsElei-vE8Gc5tkKs2wH4FJVRTevS/view

```bash
python scripts/download_data.py --out-dir data/
export LIQUIDATION_DATA_ROOT=$(pwd)/data
python scripts/run_baseline.py --symbol btcusdt --max-trades 500000
```

## Metrics

| Metric | Definition |
|--------|------------|
| **PnL_all(τ)** | Weighted mean maker markout, all trades |
| **PnL_kept(τ)** | Mean on kept trades (`f=0`) |
| **PnL_filtered(τ)** | Mean on filtered trades (`f=1`) |
| **Score(τ)** | `PnL_kept − PnL_all` — maximize |
| **Turnover/day** | `Σ w·(1−f) / days`, `w=min(notional, 100k USD)` |
| **Constraint** | Turnover/day ≥ **500,000 USD** |

Horizons τ ∈ {30, 120, 300} seconds. Bybit liquidations shifted **+200 ms** before feature join.

## Classifiers

| Strategy | Type |
|----------|------|
| `baseline_keep_all` | No filter (Score = 0) |
| `heuristic_bybit_flow` | Filter extreme \|Bybit L_net_30s\| |
| `ml_logistic_toxic` | Logistic regression + turnover-calibrated threshold |
| `lgbm_nk_30s` | LightGBM (full pipeline, bundled in `baseline_metrics_full.csv`) |

## Repository layout

```
CMF_Week_2/
├── README.md
├── config.py
├── lib/                    # Polars loaders, features, markout, filter, metrics
├── scripts/
│   ├── download_data.py
│   └── run_baseline.py
├── notebooks/
│   └── Weekly_Baseline_Report.ipynb   ← main report
├── results/
│   ├── baseline_metrics_full.csv
│   ├── baseline_metrics.csv
│   ├── REPORT.md
│   └── figures/
└── data/                   # not in git — download locally
```

## Key results (validation, τ=30s)

| Symbol | Strategy | Score (bps) | PnL_kept | Turnover/day | Constraint |
|--------|----------|-------------|----------|--------------|------------|
| BTC | lgbm_nk_30s | **+0.065** | −0.108 | 5.98M | ✓ |
| BTC | baseline | 0.000 | −0.173 | 12.2B | ✓ |
| ETH | lgbm_nk_30s | **+0.218** | +0.238 | −0.213 | 6.32M | ✓ |
| ETH | baseline | 0.000 | +0.020 | — | 12.2B | ✓ |

All reported strategies meet the **500k USD/day** turnover constraint.

See `results/REPORT.md` and the notebook for details.
