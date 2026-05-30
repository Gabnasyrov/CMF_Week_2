"""Score, PnL_kept, PnL_filtered, turnover (description.md)."""

from __future__ import annotations

import numpy as np

from ..config import MIN_TURNOVER_PER_DAY


def weighted_mean(x: np.ndarray, w: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = np.isfinite(x) & (w > 0)
    if mask is not None:
        m &= mask
    if not m.any():
        return float("nan")
    return float(np.average(x[m], weights=w[m]))


def evaluate_filter(
    pnl: np.ndarray,
    w: np.ndarray,
    f: np.ndarray,
    ts_us: np.ndarray,
) -> dict:
    """
    f=1 filter out (drop), f=0 keep.
    Returns Score, PnL_kept, PnL_filtered, turnover/day, constraint flag.
    """
    m = np.isfinite(pnl) & (w > 0)
    if not m.any():
        return {"n_trades": 0}

    days = max((ts_us[m].max() - ts_us[m].min()) / 86_400_000_000, 1.0)
    kept = m & (f == 0)
    filt = m & (f == 1)

    pnl_all = weighted_mean(pnl, w, m)
    pnl_kept = weighted_mean(pnl, w, kept)
    pnl_filtered = weighted_mean(pnl, w, filt)
    turnover_kept = float(w[kept].sum() / days) if kept.any() else 0.0

    return {
        "pnl_all_bps": pnl_all,
        "pnl_kept_bps": pnl_kept,
        "pnl_filtered_bps": pnl_filtered,
        "score_bps": pnl_kept - pnl_all if np.isfinite(pnl_kept) and np.isfinite(pnl_all) else float("nan"),
        "turnover_kept_usd_day": turnover_kept,
        "turnover_all_usd_day": float(w[m].sum() / days),
        "n_trades": int(m.sum()),
        "n_kept": int(kept.sum()),
        "n_filtered": int(filt.sum()),
        "kept_frac": float(kept.sum() / max(m.sum(), 1)),
        "winrate_kept": float((pnl[kept] > 0).mean()) if kept.any() else float("nan"),
        "meets_turnover_constraint": turnover_kept >= MIN_TURNOVER_PER_DAY,
        "n_days": days,
    }
