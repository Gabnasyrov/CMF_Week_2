# Reviewer guide — sequential walkthrough

Open files **in this order** to verify inputs → model → metrics → conclusions.

---

## Step 1 — Task & data

| # | File | What you see |
|---|------|----------------|
| 1 | [TASK.md](TASK.md) | Goal, metrics formulas, turnover constraint |
| 2 | [README.md](README.md) | Quick start, data download link |
| 3 | `notebooks/Weekly_Baseline_Report.ipynb` §1–2 | Polars inventory, sample trades |

**Input data:** Binance trades + BBO + liquidations, Bybit liquidations (+200ms).  
6-month archive: [Google Drive](https://drive.google.com/file/d/1XmxRsElei-vE8Gc5tkKs2wH4FJVRTevS/view).

---

## Step 2 — Features (model input)

| # | File | What you see |
|---|------|----------------|
| 4 | [docs/FEATURES.md](docs/FEATURES.md) | 21 features for `lgbm_nk_30s` (no Kalman cols) |
| 5 | [docs/LGBM_PARAMS.md](docs/LGBM_PARAMS.md) | LightGBM hyperparameters |

**Training target:** `sign(kalman_mu[t+30s] − kalman_mu[t])` on 1s panel.  
**Normalization:** train-only z-score, clip ±8.

Simple Polars baseline (notebook §4) uses lighter features: Bybit/Binance liq flow 30s, spread — see `lib/filter.py`.

---

## Step 3 — Baseline definition

| # | File | What you see |
|---|------|----------------|
| 6 | [docs/BASELINE.md](docs/BASELINE.md) | What «baseline» means in PnL tables |

**Baseline = no filter:** `f_i = 0` for all trades → Score = 0, full turnover.

---

## Step 4 — LGBM quality (direction model, intermediate)

| # | File | What you see |
|---|------|----------------|
| 7 | [results/lgbm_direction_metrics.json](results/lgbm_direction_metrics.json) | AUC BTC **0.560**, ETH **0.540**; hit rates |
| 8 | [results/figures/lgbm_roc_btcusdt_30s.png](results/figures/lgbm_roc_btcusdt_30s.png) | ROC curve BTC |
| 9 | [results/figures/lgbm_roc_ethusdt_30s.png](results/figures/lgbm_roc_ethusdt_30s.png) | ROC curve ETH |

Regenerate (requires enriched parquet):

```bash
export LIQUIDATION_DATA_ROOT=/path/to/data
python scripts/plot_lgbm_auc.py
```

---

## Step 5 — Trade filter → PnL (final metrics)

| # | File | What you see |
|---|------|----------------|
| 10 | `notebooks/Weekly_Baseline_Report.ipynb` §3 | Metric definitions (Score, PnL_kept, …) |
| 11 | [results/baseline_metrics_full.csv](results/baseline_metrics_full.csv) | Full test PnL for all strategies × τ |
| 12 | [results/REPORT.md](results/REPORT.md) | Summary table @ τ=30s |
| 13 | [results/figures/score_tau30_full.png](results/figures/score_tau30_full.png) | Score bar chart |

**Filter rule (LGBM):** drop trade when forecast direction aligns with taker flow (`signal × taker_side > 0`).

---

## Step 6 — Conclusions

| Finding | Evidence |
|---------|----------|
| LGBM beats no-filter on Score @ 30s | BTC +0.065 bps, ETH +0.218 bps vs baseline 0 |
| Turnover constraint met | All rows ≥ $500k/day kept (billions in practice) |
| PnL_filtered < 0 is expected | Filter removes toxic trades |
| ETH Score > BTC @ 30s | `baseline_metrics_full.csv` |

---

## Pipeline diagram

```
raw parquet (Polars load)
    ↓
1s feature panel (enriched BBO + liq + Hawkes)
    ↓
LGBM no-K train → direction signal @ 30s
    ↓
asof join signal → each Binance trade
    ↓
adverse filter f_i  (baseline: f_i=0)
    ↓
maker markout @ τ → Score, PnL_kept, PnL_filtered, turnover/day
```

---

## File map

```
CMF_Week_2/
├── REVIEWER_GUIDE.md          ← you are here
├── TASK.md
├── README.md
├── docs/                      ← features, params, baseline
├── notebooks/                 ← interactive report
├── lib/                       ← Polars baseline code
├── scripts/
│   ├── download_data.py
│   ├── run_baseline.py
│   └── plot_lgbm_auc.py
└── results/                   ← CSV, JSON, figures
```
