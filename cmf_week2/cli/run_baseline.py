"""Run Polars baseline — test split or full history."""

from __future__ import annotations

import argparse
import sys

import polars as pl

from cmf_week2.config import RESULTS, SYMBOLS, TAUS_SEC, TRAIN_END, TRAIN_START, VAL_END, VAL_START, ensure_dirs, has_raw_data
from cmf_week2.lib.features import attach_bbo_features, attach_liq_features
from cmf_week2.lib.filter import classify_trades
from cmf_week2.lib.full_history import run_full_history_symbol
from cmf_week2.lib.load_data import load_bbo_mid_1s, load_bbo_quotes, load_liq_1s, load_trades
from cmf_week2.lib.markout import add_markouts


def _build_trades(sym: str, t0, t1, max_rows: int | None, tail: bool = False) -> pl.DataFrame:
    trades = load_trades(sym, t0, t1, max_rows=max_rows, tail=tail)
    if len(trades) == 0:
        return trades
    t0_us = int(trades["timestamp"].min())
    t1_us = int(trades["timestamp"].max())
    bbo_q = load_bbo_quotes(sym, t0_us_override=t0_us, t1_us_override=t1_us)
    bbo_1s = load_bbo_mid_1s(sym, t0, t1).filter(
        (pl.col("sec") >= t0_us // 1_000_000) & (pl.col("sec") <= t1_us // 1_000_000 + 400)
    )
    liq_bn = load_liq_1s(sym, "binance", t0, t1)
    liq_bb = load_liq_1s(sym, "bybit", t0, t1)
    trades = attach_liq_features(trades, liq_bn, liq_bb)
    trades = attach_bbo_features(trades, bbo_1s)
    trades = add_markouts(trades, bbo_q)
    return trades.filter(pl.col("pnl_30s_bps").is_not_null())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="btcusdt", choices=SYMBOLS)
    p.add_argument(
        "--eval-scope",
        choices=("test", "full"),
        default="full",
        help="full=all parquet history (default); test=Feb validation only",
    )
    p.add_argument("--max-trades", type=int, default=None, help="Cap test trades (test scope only)")
    p.add_argument("--max-train-trades", type=int, default=500_000, help="Train sample for threshold tuning")
    p.add_argument("--all-symbols", action="store_true")
    args = p.parse_args(argv)

    symbols = SYMBOLS if args.all_symbols else (args.symbol,)
    for sym in symbols:
        if not has_raw_data((sym,)):
            print(f"ERROR: missing parquet for {sym} — set LIQUIDATION_DATA_ROOT", file=sys.stderr)
            return 1

    ensure_dirs()
    all_rows: list[dict] = []

    if args.eval_scope == "full":
        for sym in symbols:
            all_rows.extend(run_full_history_symbol(sym, max_train_trades=args.max_train_trades))
    else:
        for sym in symbols:
            train_s = _build_trades(sym, TRAIN_START, TRAIN_END, args.max_train_trades, tail=True)
            test_s = _build_trades(sym, VAL_START, VAL_END, args.max_trades, tail=False)
            trades = pl.concat([train_s, test_s], how="vertical").sort("timestamp")
            for tau in TAUS_SEC:
                _, results = classify_trades(trades, tau_sec=tau)
                for r in results:
                    row = {
                        "symbol": sym,
                        "split": "official_train_dec_jan_test_feb",
                        "tau_sec": tau,
                        "strategy": r.name,
                        **r.metrics,
                    }
                    if r.threshold is not None:
                        row["threshold"] = r.threshold
                    all_rows.append(row)

    out_df = pl.DataFrame(all_rows)
    csv_path = RESULTS / "baseline_metrics.csv"
    out_df.write_csv(csv_path)
    show = out_df.filter(pl.col("tau_sec") == 30).select(
        "symbol",
        "split",
        "strategy",
        "score_bps",
        "pnl_kept_bps",
        "pnl_all_bps",
        "n_trades",
        "n_days",
        "n_calendar_days",
        "turnover_kept_usd_day",
        "meets_turnover_constraint",
    )
    print(show, flush=True)
    print(f"\nWrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
