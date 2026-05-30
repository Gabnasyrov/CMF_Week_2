# CMF Week 2 — Liquidation trade filter baseline

**Course task:** filter Binance maker trades using liquidation context; report **Score**, **PnL**, **turnover/day** (≥500k USD).

## Quick start (fresh clone)

```bash
pip install -e ".[dev]"
export LIQUIDATION_DATA_ROOT=/path/to/parquet   # raw + enriched/ subdirs
python -m cmf_week2.cli.run_baseline --symbol btcusdt --max-trades 50000
pytest -q
jupyter notebook notebooks/Weekly_Baseline_Report.ipynb
```

Entry points: `cmf-run-baseline`, `cmf-run-all`.

**Reviewers:** [`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md) → [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

Notebook runs **without raw data** using bundled `results/`.

## Data (~6 months, not in git)

Parquet streams (`timestamp` μs UTC):

- Binance: trades, book tickers, liquidations — `btcusdt`, `ethusdt`
- Bybit: liquidations (+200 ms before join)
- LGBM panel: `enriched/binance_bbo/`, etc. under same root

## Metrics

| Metric | Definition |
|--------|------------|
| **Score(τ)** | `PnL_kept − PnL_all` (bps) |
| **Turnover/day** | `Σ w·(1−f) / days`, `w=min(notional, 100k USD)` |

Markout τ ∈ {30, 120, 300}s. Direction forecast h ∈ {1,3,5,30,300}s — see [`docs/FORECAST_HORIZONS.md`](docs/FORECAST_HORIZONS.md).

## Key results

**Polars baseline** (official split, Feb test): `results/baseline_metrics.csv` — includes bootstrap SE.

**LGBM PnL** (50/50 split, full pipeline): `results/baseline_metrics_full.csv` — see METHODOLOGY for `n_days` heterogeneity.

**LGBM direction @ 30s** (`lgbm_direction_metrics.json`):

| Symbol | ROC-AUC | Hit rate |
|--------|---------|----------|
| BTC | 0.560 | 0.543 |
| ETH | 0.540 | 0.527 |

Do not confuse **auc_test** with **hit_rate_test**.

## Layout

```
CMF_Week_2/
├── pyproject.toml
├── cmf_week2/           ← installable package
│   ├── lib/             ← Polars pipeline (fixed)
│   ├── cli/             ← run_baseline, run_all, plot_lgbm_auc
│   └── strategies_bundle/
├── tests/
├── docs/
└── results/
```
