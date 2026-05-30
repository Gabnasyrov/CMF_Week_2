# LightGBM hyperparameters (`lgbm_nk_30s`)

From `research/strategies_bundle/config/{symbol}.json`:

| Parameter | BTC | ETH |
|-----------|-----|-----|
| `objective` | binary | binary |
| `metric` | auc | auc |
| `max_depth` | 10 | 10 |
| `num_leaves` | 31 | 31 |
| `learning_rate` | 0.02 | 0.02 |
| `n_estimators` | 1200 | 1200 |
| `feature_fraction` | 0.8 (default) | 0.8 |
| `min_child_samples` | 50 (default) | 50 |
| early stopping | off (`no_early_stopping: true`) | off |

Train split: first **40%** of 1s panel (chronological).  
PnL evaluation split: **50% train / 50% test** on trade events (separate from direction train).
