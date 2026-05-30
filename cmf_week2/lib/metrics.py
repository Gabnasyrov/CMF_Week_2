"""Score, PnL, turnover, daily attribution, bootstrap SE."""

from __future__ import annotations

import numpy as np

from cmf_week2.config import DAY_US, MIN_TURNOVER_PER_DAY


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
    """f=1 filter out, f=0 keep."""
    m = np.isfinite(pnl) & (w > 0)
    if not m.any():
        return {"n_trades": 0}

    days = max((ts_us[m].max() - ts_us[m].min()) / DAY_US, 1.0)
    kept = m & (f == 0)
    filt = m & (f == 1)

    pnl_all = weighted_mean(pnl, w, m)
    pnl_kept = weighted_mean(pnl, w, kept)
    pnl_filtered = weighted_mean(pnl, w, filt)
    turnover_kept = float(w[kept].sum() / days) if kept.any() else 0.0

    daily = daily_pnl_stats(pnl, w, f, ts_us)
    boot = bootstrap_score(pnl, w, f, ts_us, n_boot=500, seed=42)

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
        "daily_mean_score_bps": daily.get("mean_daily_score_bps"),
        "daily_std_score_bps": daily.get("std_daily_score_bps"),
        "daily_hit_frac": daily.get("hit_frac"),
        "score_se_bps": boot.get("score_se_bps"),
        "score_ci95_low_bps": boot.get("ci95_low"),
        "score_ci95_high_bps": boot.get("ci95_high"),
    }


def daily_pnl_stats(pnl: np.ndarray, w: np.ndarray, f: np.ndarray, ts_us: np.ndarray) -> dict:
    m = np.isfinite(pnl) & (w > 0)
    if not m.any():
        return {}
    day = ts_us // DAY_US
    scores = []
    for d in np.unique(day[m]):
        dm = m & (day == d)
        dk = dm & (f == 0)
        if not dk.any():
            continue
        all_d = weighted_mean(pnl, w, dm)
        kept_d = weighted_mean(pnl, w, dk)
        if np.isfinite(all_d) and np.isfinite(kept_d):
            scores.append(kept_d - all_d)
    if not scores:
        return {}
    arr = np.asarray(scores, dtype=float)
    return {
        "mean_daily_score_bps": float(arr.mean()),
        "std_daily_score_bps": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "hit_frac": float((arr > 0).mean()),
        "n_calendar_days": len(arr),
    }


def bootstrap_score(
    pnl: np.ndarray,
    w: np.ndarray,
    f: np.ndarray,
    ts_us: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    m = np.isfinite(pnl) & (w > 0)
    if m.sum() < 10:
        return {}
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(m)
    scores = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        scores.append(
            weighted_mean(pnl[pick], w[pick], f[pick] == 0) - weighted_mean(pnl[pick], w[pick], None)
        )
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        return {}
    return {
        "score_se_bps": float(arr.std(ddof=1)),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
    }
