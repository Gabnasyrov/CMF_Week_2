"""Calendar split with purge for markout horizons."""

from __future__ import annotations

import polars as pl

from cmf_week2.config import PURGE_SEC, TRAIN_END_US, VAL_END_US, VAL_START_US


def official_train_test(trades: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Train: timestamp <= TRAIN_END (Dec 2025 – Jan 2026).
    Test:  timestamp >= VAL_START (Feb 2026 validation).
    Purge last PURGE_SEC from train rows (no label overlap into test).
    """
    purge_us = PURGE_SEC * 1_000_000
    train = trades.filter(pl.col("timestamp") <= TRAIN_END_US - purge_us)
    test = trades.filter((pl.col("timestamp") >= VAL_START_US) & (pl.col("timestamp") <= VAL_END_US))
    return train, test
