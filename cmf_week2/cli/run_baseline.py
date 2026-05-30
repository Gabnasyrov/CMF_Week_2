"""Run Polars baseline on official calendar split."""

from __future__ import annotations

import argparse
import sys

import polars as pl

from cmf_week2.config import RESULTS, SYMBOLS, TAUS_SEC, TRAIN_END, TRAIN_START, VAL_END, VAL_START, ensure_dirs, has_raw_data
from cmf_week2.lib.features import attach_bbo_features, attach_liq_features
from cmf_week2.lib.filter import classify_trades
from cmf_week2.lib.load_data import load_bbo_mid_1s, load_bbo_quotes, load_liq_1s, load_trades
from cmf_week2.lib.markout import add_markouts


def run_symbol(sym: str, max_test_trades: int | None, max_train_trades: int | None) -> pl.DataFrame:
    print(f"[{sym}] loading train {TRAIN_START.date()}..{TRAIN_END.date()}", flush=True)
    train = load_trades(sym, TRAIN_START, TRAIN_END, max_rows=max_train_trades, tail=True)
    print(f"  train trades={len(train):,}", flush=True)
    test = load_trades(sym, VAL_START, VAL_END, max_rows=max_test_trades)
    print(f"  test trades={len(test):,}", flush=True)
    trades = pl.concat([train, test], how="vertical").sort("timestamp")
    t0_us = int(trades["timestamp"].min())
    t1_us = int(trades["timestamp"].max())

    bbo_q = load_bbo_quotes(sym, t0_us_override=t0_us, t1_us_override=t1_us)
    bbo_1s = load_bbo_mid_1s(sym, TRAIN_START, VAL_END).filter(
        (pl.col("sec") >= int(t0_us // 1_000_000)) & (pl.col("sec") <= int(t1_us // 1_000_000) + 400)
    )
    liq_bn = load_liq_1s(sym, "binance", TRAIN_START, VAL_END)
    liq_bb = load_liq_1s(sym, "bybit", TRAIN_START, VAL_END)

    trades = attach_liq_features(trades, liq_bn, liq_bb)
    trades = attach_bbo_features(trades, bbo_1s)
    trades = add_markouts(trades, bbo_q)
    return trades.filter(pl.col("pnl_30s_bps").is_not_null())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="btcusdt", choices=SYMBOLS)
    p.add_argument("--max-trades", type=int, default=None, help="Cap **test** trades only (Feb)")
    p.add_argument("--max-train-trades", type=int, default=500_000, help="Cap train rows (most recent within train window)")
    p.add_argument("--all-symbols", action="store_true")
    args = p.parse_args(argv)

    symbols = SYMBOLS if args.all_symbols else (args.symbol,)
    for sym in symbols:
        if not has_raw_data((sym,)):
            print(f"ERROR: missing parquet for {sym} — set LIQUIDATION_DATA_ROOT", file=sys.stderr)
            return 1

    ensure_dirs()
    all_rows = []

    for sym in symbols:
        trades = run_symbol(sym, args.max_trades, args.max_train_trades)
        for tau in TAUS_SEC:
            _, results = classify_trades(trades, tau_sec=tau)
            for r in results:
                row = {
                    "symbol": sym,
                    "split": "official_train_dec_jan_test_feb",
                    "tau_sec": tau,
                    "strategy": r.name,
                    **{k: v for k, v in r.metrics.items() if k not in ("n_trades",) or True},
                }
                if r.threshold is not None:
                    row["threshold"] = r.threshold
                all_rows.append(row)

    out_df = pl.DataFrame(all_rows)
    csv_path = RESULTS / "baseline_metrics.csv"
    out_df.write_csv(csv_path)
    show = out_df.filter(pl.col("tau_sec") == 30).select(
        "symbol",
        "strategy",
        "score_bps",
        "score_se_bps",
        "pnl_kept_bps",
        "n_days",
        "turnover_kept_usd_day",
        "meets_turnover_constraint",
    )
    print(show, flush=True)
    print(f"\nWrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
