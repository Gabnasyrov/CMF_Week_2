"""Load parquet streams with Polars."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from ..config import BYBIT_LATENCY_US, dt_to_us, paths_for_symbol


def _scan_parquet(path: Path, t0_us: int | None, t1_us: int | None) -> pl.LazyFrame:
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
) -> pl.DataFrame:
    p = paths_for_symbol(sym)["trades"]
    t0_us = dt_to_us(t0) if t0 else None
    t1_us = dt_to_us(t1) if t1 else None
    lf = _scan_parquet(p, t0_us, t1_us).select(
        "timestamp",
        "side",
        "price",
        "amount",
    )
    if max_rows is not None:
        lf = lf.head(max_rows)
    df = lf.collect()
    return df.with_columns(
        (pl.col("price") * pl.col("amount")).alias("notional"),
        (pl.col("timestamp") // 1_000_000).alias("sec"),
    )


def load_bbo_mid_1s(
    sym: str,
    t0: datetime | None = None,
    t1: datetime | None = None,
) -> pl.DataFrame:
    """1-second last mid from Binance BBO."""
    p = paths_for_symbol(sym)["bbo"]
    t0_us = dt_to_us(t0) if t0 else None
    t1_us = dt_to_us(t1) if t1 else None
    pad = 400_000_000  # 400s pad for markout horizons
    if t0_us is not None:
        t0_us -= pad
    if t1_us is not None:
        t1_us += pad
    df = (
        _scan_parquet(p, t0_us, t1_us)
        .select(
            "timestamp",
            "bid_price",
            "ask_price",
        )
        .with_columns(
            ((pl.col("bid_price") + pl.col("ask_price")) / 2).alias("mid"),
            ((pl.col("ask_price") - pl.col("bid_price")) / ((pl.col("bid_price") + pl.col("ask_price")) / 2) * 10_000).alias(
                "spread_bps"
            ),
            (pl.col("timestamp") // 1_000_000).alias("sec"),
        )
        .group_by("sec")
        .agg(pl.col("mid").last(), pl.col("spread_bps").last())
        .sort("sec")
        .collect()
    )
    return df


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
        (signed * pl.col("price") * pl.col("amount")).alias("signed_usd"),
        (pl.col("timestamp") // 1_000_000).alias("sec"),
    )
    return (
        df.group_by("sec")
        .agg(
            pl.col("notional").sum().alias("liq_notional"),
            pl.col("signed_usd").sum().alias("liq_signed"),
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
