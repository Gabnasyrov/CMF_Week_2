"""Day-chunked full-history evaluation (train tuning, metrics on all trades)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from cmf_week2.config import DAY_US, PURGE_SEC, TRAIN_END, TRAIN_END_US, TRAIN_START, TAUS_SEC, dt_to_us
from cmf_week2.lib.features import attach_bbo_features, attach_liq_features
from cmf_week2.lib.filter import FilterFit, apply_filter, fit_filters
from cmf_week2.lib.load_data import data_range_us, load_bbo_mid_1s, load_bbo_quotes, load_liq_1s, load_trades
from cmf_week2.lib.markout import add_markouts
from cmf_week2.lib.metrics import MetricAccumulator


def _day_bounds(day: datetime) -> tuple[datetime, datetime]:
    t0 = day.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=1) - timedelta(microseconds=1)
    return t0, t1


def _iter_days(t0: datetime, t1: datetime):
    d = t0.date()
    end = t1.date()
    while d <= end:
        yield datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        d += timedelta(days=1)


def _prepare_train_sample(
    sym: str,
    max_train_trades: int,
    liq_bn: pl.DataFrame,
    liq_bb: pl.DataFrame,
) -> pl.DataFrame:
    purge_us = PURGE_SEC * 1_000_000
    train_end_us = TRAIN_END_US - purge_us
    train = load_trades(
        sym,
        TRAIN_START,
        TRAIN_END,
        max_rows=max_train_trades,
        tail=True,
    ).filter(pl.col("timestamp") <= train_end_us)
    if len(train) == 0:
        raise ValueError("Empty train sample for filter tuning")
    t0_us = int(train["timestamp"].min())
    t1_us = int(train["timestamp"].max())
    bbo_q = load_bbo_quotes(sym, t0_us_override=t0_us, t1_us_override=t1_us)
    bbo_1s = load_bbo_mid_1s(sym, TRAIN_START, TRAIN_END).filter(
        (pl.col("sec") >= t0_us // 1_000_000) & (pl.col("sec") <= t1_us // 1_000_000 + 400)
    )
    train = attach_liq_features(train, liq_bn, liq_bb)
    train = attach_bbo_features(train, bbo_1s)
    train = add_markouts(train, bbo_q)
    return train.filter(pl.col("pnl_30s_bps").is_not_null())


def run_full_history_symbol(
    sym: str,
    max_train_trades: int = 500_000,
) -> list[dict]:
    t0_us, t1_us = data_range_us(sym)
    t0 = datetime.fromtimestamp(t0_us / 1e6, tz=timezone.utc)
    t1 = datetime.fromtimestamp(t1_us / 1e6, tz=timezone.utc)
    print(f"[{sym}] full history {t0.date()} .. {t1.date()}", flush=True)

    liq_bn = load_liq_1s(sym, "binance", t0, t1)
    liq_bb = load_liq_1s(sym, "bybit", t0, t1)
    train = _prepare_train_sample(sym, max_train_trades, liq_bn, liq_bb)
    print(f"  train sample for tuning: {len(train):,} trades", flush=True)

    fits: dict[int, FilterFit] = {tau: fit_filters(train, tau) for tau in TAUS_SEC}
    acc: dict[tuple[int, str], MetricAccumulator] = {
        (tau, name): MetricAccumulator()
        for tau in TAUS_SEC
        for name in ("baseline_keep_all", "heuristic_bybit_flow", "ml_logistic_toxic")
    }

    n_days = 0
    n_trades_total = 0
    for day in _iter_days(t0, t1):
        d0, d1 = _day_bounds(day)
        chunk = load_trades(sym, d0, d1)
        if len(chunk) == 0:
            continue
        t0c = int(chunk["timestamp"].min())
        t1c = int(chunk["timestamp"].max())
        bbo_q = load_bbo_quotes(sym, t0_us_override=t0c, t1_us_override=t1c)
        bbo_1s = (
            load_bbo_quotes(sym, t0_us_override=t0c, t1_us_override=t1c)
            .group_by("sec")
            .agg(pl.col("mid").last(), pl.col("spread_bps").last())
            .sort("sec")
        )
        chunk = attach_liq_features(chunk, liq_bn, liq_bb)
        chunk = attach_bbo_features(chunk, bbo_1s)
        chunk = add_markouts(chunk, bbo_q).filter(pl.col("pnl_30s_bps").is_not_null())
        if len(chunk) == 0:
            continue
        n_days += 1
        n_trades_total += len(chunk)
        if n_days % 10 == 0:
            print(f"  day {day.date()} cumulative trades={n_trades_total:,}", flush=True)

        for tau in TAUS_SEC:
            pnl_col = f"pnl_{tau}s_bps"
            pnl = chunk[pnl_col].to_numpy()
            w = chunk["w"].to_numpy()
            ts = chunk["timestamp"].to_numpy()
            fit = fits[tau]
            for name in acc:
                if name[0] != tau:
                    continue
                f = apply_filter(chunk, fit, name[1])
                acc[(tau, name[1])].add(pnl, w, f, ts)

    print(f"  done: {n_days} days, {n_trades_total:,} trades with markout", flush=True)
    rows = []
    for tau in TAUS_SEC:
        for name in ("baseline_keep_all", "heuristic_bybit_flow", "ml_logistic_toxic"):
            fit = fits[tau]
            thr = None
            if name == "heuristic_bybit_flow":
                thr = fit.heuristic_thr
            elif name == "ml_logistic_toxic":
                thr = fit.ml_thr
            m = acc[(tau, name)].finalize()
            rows.append(
                {
                    "symbol": sym,
                    "split": "full_history",
                    "tau_sec": tau,
                    "strategy": name,
                    "threshold": thr,
                    **m,
                }
            )
    return rows
