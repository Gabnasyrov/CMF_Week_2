#!/usr/bin/env python3
"""Run Polars baseline pipeline and write results CSV + markdown report."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from week_baseline.config import (  # noqa: E402
    RESULTS,
    SYMBOLS,
    TAUS_SEC,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    ensure_dirs,
    has_raw_data,
)
from week_baseline.lib.features import attach_bbo_features, attach_liq_features  # noqa: E402
from week_baseline.lib.filter import classify_trades  # noqa: E402
from week_baseline.lib.load_data import load_bbo_mid_1s, load_liq_1s, load_trades  # noqa: E402
from week_baseline.lib.markout import add_markouts  # noqa: E402


def run_symbol(sym: str, max_trades: int | None, t0: datetime, t1: datetime) -> pl.DataFrame:
    print(f"[{sym}] loading trades...", flush=True)
    trades = load_trades(sym, t0, t1, max_rows=max_trades)
    print(f"  trades={len(trades):,}", flush=True)
    bbo = load_bbo_mid_1s(sym, t0, t1)
    liq_bn = load_liq_1s(sym, "binance", t0, t1)
    liq_bb = load_liq_1s(sym, "bybit", t0, t1)
    trades = attach_liq_features(trades, liq_bn, liq_bb)
    trades = attach_bbo_features(trades, bbo)
    trades = add_markouts(trades, bbo)
    trades = trades.filter(pl.col("pnl_30s_bps").is_not_null())
    return trades


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="btcusdt")
    p.add_argument("--max-trades", type=int, default=500_000, help="Cap trades for notebook-speed runs")
    p.add_argument("--split", choices=("chronological", "official"), default="official")
    args = p.parse_args()

    if not has_raw_data((args.symbol,)):
        print("ERROR: missing parquet — run scripts/download_data.py or set LIQUIDATION_DATA_ROOT", file=sys.stderr)
        return 1

    ensure_dirs()
    if args.split == "official":
        t0, t1 = TRAIN_START, VAL_END
    else:
        t0, t1 = TRAIN_START, VAL_END

    trades = run_symbol(args.symbol, args.max_trades, t0, t1)
    rows = []
    for tau in TAUS_SEC:
        _, results = classify_trades(trades, train_frac=2 / 3 if args.split == "official" else 0.5, tau_sec=tau)
        for r in results:
            row = {"symbol": args.symbol, "tau_sec": tau, "strategy": r.name, **r.metrics}
            if r.threshold is not None:
                row["threshold"] = r.threshold
            rows.append(row)

    out_df = pl.DataFrame(rows)
    csv_path = RESULTS / "baseline_metrics.csv"
    out_df.write_csv(csv_path)
    print(out_df.select(["strategy", "tau_sec", "score_bps", "pnl_kept_bps", "pnl_filtered_bps", "turnover_kept_usd_day", "meets_turnover_constraint"]))
    print(f"\nWrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
