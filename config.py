"""Week baseline: paths, splits, task constants."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
_local_data = (PKG_ROOT / "data").resolve()
if os.environ.get("LIQUIDATION_DATA_ROOT"):
    DATA = Path(os.environ["LIQUIDATION_DATA_ROOT"]).resolve()
else:
    DATA = _local_data

RESULTS = PKG_ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
NOTEBOOKS = PKG_ROOT / "notebooks"

SYMBOLS = ("btcusdt", "ethusdt")
TAUS_SEC = (30, 120, 300)

MAKER_REBATE_BPS = 0.5
WEIGHT_CAP_USD = 100_000
BYBIT_LATENCY_US = 200_000
MIN_TURNOVER_PER_DAY = 500_000

# Official public split (3 months in task spec)
TRAIN_START = datetime(2025, 12, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
VAL_START = datetime(2026, 2, 1, tzinfo=timezone.utc)
VAL_END = datetime(2026, 2, 28, 23, 59, 59, tzinfo=timezone.utc)

# Google Drive archive may contain ~6 months; filter after load if needed
DATA_START = datetime(2025, 12, 1, tzinfo=timezone.utc)
DATA_END = datetime(2026, 5, 31, 23, 59, 59, tzinfo=timezone.utc)

GOOGLE_DRIVE_FILE_ID = "1XmxRsElei-vE8Gc5tkKs2wH4FJVRTevS"

DAY_US = 86_400_000_000


def dt_to_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


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


def ensure_dirs() -> None:
    for d in (RESULTS, FIGURES, TABLES, NOTEBOOKS, DATA):
        d.mkdir(parents=True, exist_ok=True)
