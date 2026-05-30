"""Score, PnL, turnover, daily attribution, bootstrap SE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cmf_week2.config import DAY_US, MIN_TURNOVER_PER_DAY


def weighted_mean(x: np.ndarray, w: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = np.isfinite(x) & (w > 0)
    if mask is not None:
        m &= mask
    if not m.any():
        return float("nan")
    return float(np.average(x[m], weights=w[m]))


@dataclass
class MetricAccumulator:
    sum_w_all: float = 0.0
    sum_wp_all: float = 0.0
    sum_w_kept: float = 0.0
    sum_wp_kept: float = 0.0
    sum_w_filt: float = 0.0
    sum_wp_filt: float = 0.0
    n_trades: int = 0
    n_kept: int = 0
    n_filtered: int = 0
    n_win_kept: int = 0
    ts_min: int | None = None
    ts_max: int | None = None
    daily_scores: list[float] = field(default_factory=list)

    def add(self, pnl: np.ndarray, w: np.ndarray, f: np.ndarray, ts_us: np.ndarray) -> None:
        m = np.isfinite(pnl) & (w > 0)
        if not m.any():
            return
        kept = m & (f == 0)
        filt = m & (f == 1)
        self.sum_w_all += float(w[m].sum())
        self.sum_wp_all += float((w[m] * pnl[m]).sum())
        if kept.any():
            self.sum_w_kept += float(w[kept].sum())
            self.sum_wp_kept += float((w[kept] * pnl[kept]).sum())
            self.n_win_kept += int((pnl[kept] > 0).sum())
        if filt.any():
            self.sum_w_filt += float(w[filt].sum())
            self.sum_wp_filt += float((w[filt] * pnl[filt]).sum())
        self.n_trades += int(m.sum())
        self.n_kept += int(kept.sum())
        self.n_filtered += int(filt.sum())
        tmin = int(ts_us[m].min())
        tmax = int(ts_us[m].max())
        self.ts_min = tmin if self.ts_min is None else min(self.ts_min, tmin)
        self.ts_max = tmax if self.ts_max is None else max(self.ts_max, tmax)

        day = ts_us // DAY_US
        for d in np.unique(day[m]):
            dm = m & (day == d)
            dk = dm & (f == 0)
            if not dk.any():
                continue
            all_d = weighted_mean(pnl, w, dm)
            kept_d = weighted_mean(pnl, w, dk)
            if np.isfinite(all_d) and np.isfinite(kept_d):
                self.daily_scores.append(kept_d - all_d)

    def finalize(self, bootstrap: bool = True) -> dict:
        if self.n_trades == 0 or self.sum_w_all <= 0:
            return {"n_trades": 0}
        days = max((self.ts_max - self.ts_min) / DAY_US, 1.0) if self.ts_min and self.ts_max else 1.0
        pnl_all = self.sum_wp_all / self.sum_w_all
        pnl_kept = self.sum_wp_kept / self.sum_w_kept if self.sum_w_kept > 0 else float("nan")
        pnl_filtered = self.sum_wp_filt / self.sum_w_filt if self.sum_w_filt > 0 else float("nan")
        turnover_kept = self.sum_w_kept / days if self.sum_w_kept > 0 else 0.0
        out = {
            "pnl_all_bps": pnl_all,
            "pnl_kept_bps": pnl_kept,
            "pnl_filtered_bps": pnl_filtered,
            "score_bps": pnl_kept - pnl_all if np.isfinite(pnl_kept) else float("nan"),
            "turnover_kept_usd_day": turnover_kept,
            "turnover_all_usd_day": self.sum_w_all / days,
            "n_trades": self.n_trades,
            "n_kept": self.n_kept,
            "n_filtered": self.n_filtered,
            "kept_frac": self.n_kept / max(self.n_trades, 1),
            "winrate_kept": self.n_win_kept / max(self.n_kept, 1) if self.n_kept else float("nan"),
            "meets_turnover_constraint": turnover_kept >= MIN_TURNOVER_PER_DAY,
            "n_days": days,
        }
        if self.daily_scores:
            arr = np.asarray(self.daily_scores, dtype=float)
            out["daily_mean_score_bps"] = float(arr.mean())
            out["daily_std_score_bps"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            out["daily_hit_frac"] = float((arr > 0).mean())
            out["n_calendar_days"] = len(arr)
        return out


def evaluate_filter(
    pnl: np.ndarray,
    w: np.ndarray,
    f: np.ndarray,
    ts_us: np.ndarray,
) -> dict:
    """f=1 filter out, f=0 keep."""
    acc = MetricAccumulator()
    acc.add(pnl, w, f, ts_us)
    out = acc.finalize(bootstrap=False)
    if out.get("n_trades", 0) == 0:
        return out

    boot = bootstrap_score(pnl, w, f, ts_us, n_boot=500, seed=42)
    out.update(boot)
    daily = daily_pnl_stats(pnl, w, f, ts_us)
    out.update({k: v for k, v in daily.items() if k not in out})
    return out


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
        "score_ci95_low_bps": float(np.quantile(arr, 0.025)),
        "score_ci95_high_bps": float(np.quantile(arr, 0.975)),
    }
