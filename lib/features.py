"""Trade-level features via Polars (causal joins)."""

from __future__ import annotations

import polars as pl


def attach_liq_features(trades: pl.DataFrame, liq_bn: pl.DataFrame, liq_bb: pl.DataFrame) -> pl.DataFrame:
    """Rolling 30s liquidation flow known at trade second."""
    for name, liq in (("bn", liq_bn), ("bb", liq_bb)):
        liq = liq.sort("sec").with_columns(
            pl.col("liq_signed").rolling_sum(window_size=30, min_samples=1).alias(f"{name}_l_net_30s"),
            pl.col("liq_notional").rolling_sum(window_size=30, min_samples=1).alias(f"{name}_liq_notional_30s"),
            pl.col("liq_count").rolling_sum(window_size=30, min_samples=1).alias(f"{name}_liq_count_30s"),
        )
        trades = trades.join(liq.select("sec", f"{name}_l_net_30s", f"{name}_liq_notional_30s", f"{name}_liq_count_30s"), on="sec", how="left")

    trades = trades.with_columns(
        pl.col("bn_l_net_30s").fill_null(0),
        pl.col("bb_l_net_30s").fill_null(0),
        pl.col("bn_liq_notional_30s").fill_null(0),
        pl.col("bb_liq_notional_30s").fill_null(0),
        (pl.col("timestamp") // 3_600_000_000 % 24).alias("hour_utc"),
        pl.when(pl.col("side") == "buy").then(1).otherwise(-1).alias("taker_sign"),
    )
    return trades


def attach_bbo_features(trades: pl.DataFrame, bbo_1s: pl.DataFrame) -> pl.DataFrame:
    bbo = bbo_1s.sort("sec").select("sec", "mid", "spread_bps")
    return trades.join(bbo, on="sec", how="left")
