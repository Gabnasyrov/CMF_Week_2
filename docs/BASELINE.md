# What is «baseline»?

In PnL tables, **`baseline`** = **no trade filter**.

| Property | Value |
|----------|-------|
| Signal | always `0` |
| Filter `f_i` | always `0` (keep every trade) |
| Score | **0 by definition** (`PnL_kept = PnL_all`) |
| Turnover kept | 100% of clipped notional |
| Purpose | reference to measure filter uplift |

## How LGBM strategy differs

1. **Train** LGBM no-K on 1s panel → predict direction of Kalman level move @ 30s.
2. **Map** signal to each Binance trade (backward asof join on timestamp).
3. **Filter** (`filter_adverse_trades`): drop trade when `signal × taker_side > 0`  
   (signal agrees with taker flow → adverse for passive maker).
4. **Markout** on kept trades @ τ ∈ {30, 120, 300}s → Score, PnL_kept, turnover.

Other strategies in full pipeline: `kalman_30s`, `lgbm_full_30s`, `ensemble_30s` — see `research/pnl_evaluation/config.py`.
