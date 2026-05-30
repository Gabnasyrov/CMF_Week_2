# LGBM input features (`lgbm_nk_30s`)

Model **`lgbm_nk_30s`** uses **z-scored** features (`{name}_z`) from train split only.

## Feature set: `FEATURES_NO_KALMAN` (21 features)

Kalman columns are **excluded** to avoid target leakage (`target = sign(Δ kalman_mu)`).

| Group | Features |
|-------|----------|
| **Hawkes** | `hawkes_lambda`, `hawkes_branching` |
| **Bayes trend** | `trend_regime_5s`, `trend_slope_5s`, `cp_cred_5s` |
| **Micro / book** | `microprice_g`, `book_imbalance`, `ofi_increment`, `spread_bps`, `misprice_bid_bps`, `misprice_ask_bps`, `bid_depth_usd`, `ask_depth_usd`, `signed_depth_l1` |
| **Flow / vol** | `signed_vol_1s`, `signed_vol_5s`, `trades_per_sec`, `vol_30s`, `vov_30s` |
| **Liquidations** | `liq_bid_5s`, `liq_ask_5s` |

Source: 1s Binance panel from enriched BBO + trades + liquidations (Binance + Bybit liq, Bybit +200ms).

## Target (training label)

```
y_mu_sign_30s = sign(kalman_mu[t+30s] - kalman_mu[t])
```

Binary LGBM: predict P(up move) → signal ∈ {-1, +1}.

## Normalization

Train-only z-score per feature, clip to ±8 (`ZScoreNormalizer`).
