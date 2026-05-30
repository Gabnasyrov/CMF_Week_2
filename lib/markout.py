"""Maker markout labels (task spec)."""

from __future__ import annotations

import numpy as np
import polars as pl

from config import MAKER_REBATE_BPS, WEIGHT_CAP_USD, TAUS_SEC


def mid_at_sec(mid_sec: np.ndarray, mid_val: np.ndarray, query_sec: np.ndarray) -> np.ndarray:
    out = np.full(len(query_sec), np.nan)
    for i, s in enumerate(query_sec):
        j = int(np.searchsorted(mid_sec, s, side="right") - 1)
        if j >= 0:
            out[i] = mid_val[j]
    return out


def add_markouts(trades: pl.DataFrame, bbo_1s: pl.DataFrame, taus: tuple[int, ...] = TAUS_SEC) -> pl.DataFrame:
    mid_sec = bbo_1s["sec"].to_numpy().astype(np.int64)
    mid_val = bbo_1s["mid"].to_numpy().astype(float)
    ts = trades["timestamp"].to_numpy().astype(np.int64)
    price = trades["price"].to_numpy().astype(float)
    side = trades["side"].to_numpy()
    taker = np.where(side == "buy", 1.0, -1.0)

    w = np.minimum(trades["notional"].to_numpy().astype(float), WEIGHT_CAP_USD)
    out = trades.with_columns(pl.Series("w", w))

    for tau in taus:
        tau_us = tau * 1_000_000
        target_sec = (ts + tau_us) // 1_000_000
        mid_tau = mid_at_sec(mid_sec, mid_val, target_sec)
        pnl = np.full(len(ts), np.nan)
        m = np.isfinite(mid_tau) & np.isfinite(price) & (price > 0)
        pnl[m] = -taker[m] * (mid_tau[m] - price[m]) / price[m] * 10_000 + MAKER_REBATE_BPS
        out = out.with_columns(pl.Series(f"pnl_{tau}s_bps", pnl))
    return out
