"""Trade-level features via Polars (causal joins on dense 1s grid)."""

from __future__ import annotations

import polars as pl


def _dense_liq_30s(liq: pl.DataFrame, prefix: str, sec_lo: int, sec_hi: int) -> pl.DataFrame:
    """Rolling 30s liquidation flow on dense seconds [sec_lo, sec_hi] (zeros where no liq)."""
    if sec_hi < sec_lo:
        sec_hi = sec_lo
    liq = liq.filter((pl.col("sec") >= sec_lo - 29) & (pl.col("sec") <= sec_hi))
    dense = pl.DataFrame({"sec": pl.arange(sec_lo, sec_hi + 1, eager=True, dtype=pl.Int64)})
    dense = dense.join(liq, on="sec", how="left").with_columns(
        pl.col("liq_signed").fill_null(0.0),
        pl.col("liq_notional").fill_null(0.0),
        pl.col("liq_count").fill_null(0),
    )
    return dense.with_columns(
        pl.col("liq_signed").rolling_sum(window_size=30, min_samples=1).alias(f"{prefix}_l_net_30s"),
        pl.col("liq_notional").rolling_sum(window_size=30, min_samples=1).alias(f"{prefix}_liq_notional_30s"),
        pl.col("liq_count").rolling_sum(window_size=30, min_samples=1).alias(f"{prefix}_liq_count_30s"),
    ).select("sec", f"{prefix}_l_net_30s", f"{prefix}_liq_notional_30s", f"{prefix}_liq_count_30s")


def attach_liq_features(trades: pl.DataFrame, liq_bn: pl.DataFrame, liq_bb: pl.DataFrame) -> pl.DataFrame:
    sec_lo = int(trades["sec"].min())
    sec_hi = int(trades["sec"].max())
    for prefix, liq in (("bn", liq_bn), ("bb", liq_bb)):
        feat = _dense_liq_30s(liq, prefix, sec_lo, sec_hi)
        trades = trades.join(feat, on="sec", how="left")

    trades = trades.with_columns(
        pl.col("bn_l_net_30s").fill_null(0),
        pl.col("bb_l_net_30s").fill_null(0),
        pl.col("bn_liq_notional_30s").fill_null(0),
        pl.col("bb_liq_notional_30s").fill_null(0),
        pl.col("bn_liq_count_30s").fill_null(0).cast(pl.Int64),
        pl.col("bb_liq_count_30s").fill_null(0).cast(pl.Int64),
        (pl.col("timestamp") // 3_600_000_000 % 24).alias("hour_utc"),
        pl.when(pl.col("side") == "buy").then(1).otherwise(-1).alias("taker_sign"),
    )
    return trades


def attach_bbo_features(trades: pl.DataFrame, bbo_1s: pl.DataFrame) -> pl.DataFrame:
    sec_lo = int(trades["sec"].min())
    sec_hi = int(trades["sec"].max())
    bbo = bbo_1s.filter((pl.col("sec") >= sec_lo) & (pl.col("sec") <= sec_hi)).select("sec", "mid", "spread_bps")
    return trades.join(bbo, on="sec", how="left")
