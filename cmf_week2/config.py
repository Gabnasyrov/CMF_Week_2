"""Paths, splits, task constants."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent

_local_data = (REPO_ROOT / "data").resolve()
if os.environ.get("LIQUIDATION_DATA_ROOT"):
    DATA = Path(os.environ["LIQUIDATION_DATA_ROOT"]).resolve()
else:
    DATA = _local_data

RESULTS = REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
NOTEBOOKS = REPO_ROOT / "notebooks"
STRATEGIES_BUNDLE = PKG_ROOT / "strategies_bundle"

SYMBOLS = ("btcusdt", "ethusdt")
TAUS_SEC = (30, 120, 300)

MAKER_REBATE_BPS = 0.5
WEIGHT_CAP_USD = 100_000
BYBIT_LATENCY_US = 200_000
MIN_TURNOVER_PER_DAY = 500_000
MAX_QUOTE_STALE_US = 5_000_000  # drop markout if BBO older than 5s at horizon

# Official calendar split (task spec)
TRAIN_START = datetime(2025, 12, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
VAL_START = datetime(2026, 2, 1, tzinfo=timezone.utc)
VAL_END = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)

DAY_US = 86_400_000_000
PURGE_SEC = max(TAUS_SEC)  # purge around split for markout labels


def dt_to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


TRAIN_END_US = dt_to_us(TRAIN_END)
VAL_START_US = dt_to_us(VAL_START)
VAL_END_US = dt_to_us(VAL_END)


def paths_for_symbol(sym: str) -> dict[str, Path]:
    return {
        "trades": DATA / "binance_trades" / f"perp_{sym}.parquet",
        "bbo": DATA / "binance_booktickers" / f"perp_{sym}.parquet",
        "liq_binance": DATA / "binance_liquidations" / f"perp_{sym}.parquet",
        "liq_bybit": DATA / "bybit_liquidations" / f"{sym}.parquet",
    }


def has_raw_data(symbols: tuple[str, ...] = SYMBOLS) -> bool:
    for sym in symbols:
        for path in paths_for_symbol(sym).values():
            if not path.is_file():
                return False
    return True


def liquidation_task_root() -> Path | None:
    env = os.environ.get("LIQUIDATION_TASK_ROOT")
    if env:
        p = Path(env).resolve()
        return p if p.is_dir() else None
    sibling = REPO_ROOT.parent / "liquidation_task"
    return sibling if sibling.is_dir() else None


def ensure_dirs() -> None:
    for d in (RESULTS, FIGURES, TABLES, NOTEBOOKS, DATA):
        d.mkdir(parents=True, exist_ok=True)
