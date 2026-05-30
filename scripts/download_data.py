#!/usr/bin/env python3
"""Extract 6-month parquet archive to data/ (optional; archive not bundled in repo)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FILE_ID = "1XmxRsElei-vE8Gc5tkKs2wH4FJVRTevS"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DATA)
    p.add_argument("--file-id", default=FILE_ID)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "liquidation_data.zip"

    try:
        import gdown  # noqa: F401
    except ImportError:
        print("Install: pip install gdown", file=sys.stderr)
        return 1

    url = f"https://drive.google.com/uc?id={args.file_id}"
    print(f"Downloading {url} → {archive}", flush=True)
    import gdown

    gdown.download(url, str(archive), quiet=False, fuzzy=True)

    print("Extracting...", flush=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(out)

    print(f"Done. Place/check parquet under {out}/binance_trades/ etc.", flush=True)
    print("Set: export LIQUIDATION_DATA_ROOT=" + str(out.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
