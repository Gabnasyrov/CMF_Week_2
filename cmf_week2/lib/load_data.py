"""Load parquet streams with Polars."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from cmf_week2.config import BYBIT_LATENCY_US, dt_to_us, paths_for_symbol


def _scan_parquet(path, t0_us: int | None, t1_us: int | None) -> pl.LazyFrame:
    lf = pl.scan_parquet(str(path))
    if t0_us is not None:
        lf = lf.filter(pl.col("timestamp") >= t0_us)
    if t1_us is not None:
        lf = lf.filter(pl.col("timestamp") <= t1_us)
    return lf


def load_trades(
    sym: str,
    t0: datetime | None = None,
    t1: datetime | None = None,
    max_rows: int | None = None,
    tail: bool = False,
) -> pl.DataFrame:
    p = paths_for_symbol(sym)["trades"]
    t0_us = dt_to_us(t0) if t0 else None
    t1_us = dt_to_us(t1) if t1 else None
    lf = _scan_parquet(p, t0_us, t1_us).select("timestamp", "side", "price", "amount")
    if max_rows is not None and tail:
        lf = lf.sort("timestamp", descending=True).head(max_rows).sort("timestamp")
    elif max_rows is not None:
        lf = lf.head(max_rows)
    return lf.collect().with_columns(
        (pl.col("price") * pl.col("amount")).alias("notional"),
        (pl.col("timestamp") // 1_000_000).alias("sec"),
    )


def load_bbo_quotes(
    sym: str,
    t0: datetime | None = None,
    t1: datetime | None = None,
    pad_us: int = 400_000_000,
    t0_us_override: int | None = None,
    t1_us_override: int | None = None,
) -> pl.DataFrame:
    """Raw BBO quotes for asof markout (sorted by timestamp)."""
    p = paths_for_symbol(sym)["bbo"]
    if t0_us_override is not None:
        t0_us = t0_us_override - pad_us
    else:
        t0_us = dt_to_us(t0) - pad_us if t0 else None
    if t1_us_override is not None:
        t1_us = t1_us_override + pad_us
    else:
        t1_us = dt_to_us(t1) + pad_us if t1 else None
    return (
        _scan_parquet(p, t0_us, t1_us)
        .select("timestamp", "bid_price", "ask_price")
        .with_columns(
            ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid"),
            ((pl.col("ask_price") - pl.col("bid_price")) / ((pl.col("bid_price") + pl.col("ask_price")) / 2) * 10_000).alias(
                "spread_bps"
            ),
            (pl.col("timestamp") // 1_000_000).alias("sec"),
        )
        .sort("timestamp")
        .collect()
    )


def load_bbo_mid_1s(sym: str, t0: datetime | None = None, t1: datetime | None = None) -> pl.DataFrame:
    """1-second last mid from Binance BBO (for features)."""
    return (
        load_bbo_quotes(sym, t0, t1)
        .group_by("sec")
        .agg(pl.col("mid").last(), pl.col("spread_bps").last())
        .sort("sec")
    )


def load_liq_1s(
    sym: str,
    venue: str,
    t0: datetime | None = None,
    t1: datetime | None = None,
) -> pl.DataFrame:
    paths = paths_for_symbol(sym)
    p = paths["liq_bybit"] if venue == "bybit" else paths["liq_binance"]
    t0_us = dt_to_us(t0) if t0 else None
    t1_us = dt_to_us(t1) if t1 else None
    lf = _scan_parquet(p, t0_us, t1_us).select("timestamp", "side", "price", "amount")
    df = lf.collect()
    if venue == "bybit":
        df = df.with_columns((pl.col("timestamp") + BYBIT_LATENCY_US).alias("timestamp"))
    signed = pl.when(pl.col("side") == "buy").then(1.0).otherwise(-1.0)
    df = df.with_columns(
        (pl.col("price") * pl.col("amount")).alias("notional"),
        (signed * pl.col("price") * pl.col("amount")).alias("liq_signed"),
        (pl.col("timestamp") // 1_000_000).alias("sec"),
    )
    return (
        df.group_by("sec")
        .agg(
            pl.col("notional").sum().alias("liq_notional"),
            pl.col("liq_signed").sum().alias("liq_signed"),
            pl.len().alias("liq_count"),
        )
        .sort("sec")
    )


def data_inventory(symbols: tuple[str, ...]) -> pl.DataFrame:
    rows = []
    for sym in symbols:
        for name, path in paths_for_symbol(sym).items():
            exists = path.is_file()
            rows.append(
                {
                    "symbol": sym,
                    "stream": name,
                    "path": str(path),
                    "exists": exists,
                    "size_mb": round(path.stat().st_size / 1e6, 1) if exists else None,
                }
            )
    return pl.DataFrame(rows)
