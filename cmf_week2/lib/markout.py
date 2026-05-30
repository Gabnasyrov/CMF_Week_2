"""Maker markout labels — asof on quote timestamps with stale-quote filter."""

from __future__ import annotations

import numpy as np
import polars as pl

from cmf_week2.config import MAKER_REBATE_BPS, MAX_QUOTE_STALE_US, TAUS_SEC, WEIGHT_CAP_USD


def mid_at_horizon_asof(
    trade_ts: np.ndarray,
    bbo_ts: np.ndarray,
    bbo_mid: np.ndarray,
    tau_us: int,
    max_stale_us: int = MAX_QUOTE_STALE_US,
) -> np.ndarray:
    """Last mid at or before trade_ts + tau; NaN if quote is stale."""
    q = trade_ts.astype(np.int64) + tau_us
    j = np.searchsorted(bbo_ts, q, side="right") - 1
    out = np.full(len(trade_ts), np.nan, dtype=float)
    ok = j >= 0
    if not ok.any():
        return out
    age = q[ok] - bbo_ts[j[ok]]
    fresh = age <= max_stale_us
    idx = np.flatnonzero(ok)
    out[idx[fresh]] = bbo_mid[j[ok][fresh]]
    return out


def add_markouts(
    trades: pl.DataFrame,
    bbo: pl.DataFrame,
    taus: tuple[int, ...] = TAUS_SEC,
) -> pl.DataFrame:
    bbo_ts = bbo["timestamp"].to_numpy().astype(np.int64)
    bbo_mid = bbo["mid"].to_numpy().astype(float)
    ts = trades["timestamp"].to_numpy().astype(np.int64)
    price = trades["price"].to_numpy().astype(float)
    side = trades["side"].to_numpy()
    taker = np.where(side == "buy", 1.0, -1.0)
    w = np.minimum(trades["notional"].to_numpy().astype(float), WEIGHT_CAP_USD)

    out = trades.with_columns(pl.Series("w", w))
    for tau in taus:
        tau_us = tau * 1_000_000
        mid_tau = mid_at_horizon_asof(ts, bbo_ts, bbo_mid, tau_us)
        pnl = np.full(len(ts), np.nan)
        m = np.isfinite(mid_tau) & np.isfinite(price) & (price > 0)
        pnl[m] = -taker[m] * (mid_tau[m] - price[m]) / price[m] * 10_000 + MAKER_REBATE_BPS
        out = out.with_columns(pl.Series(f"pnl_{tau}s_bps", pnl))
    return out
