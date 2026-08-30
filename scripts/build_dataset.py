#!/usr/bin/env python
"""Stream Nasdaq's published ITCH days and keep the panel's messages.

    python scripts/build_dataset.py                 # all seven days, twelve symbols
    python scripts/build_dataset.py --quick         # one day, four symbols
    python scripts/build_dataset.py --dates 2019-01-30 2020-01-30

About 31 GB is transferred and 90 GB inflated; roughly 1.5 GB of parquet is
written.  Nothing else touches disk -- the raw files are consumed from the
socket.  Re-running skips days whose parquet is already complete.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hfx.itch.source import DAYS
from hfx.pipeline.build import build_day
from hfx.pipeline.panel import QUICK_SYMBOLS, SYMBOLS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="*", default=None, help="ISO dates; default all")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true", help="one day, four symbols")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument(
        "--raw-dir",
        default=None,
        help="directory of already-downloaded .gz files, used instead of the network",
    )
    args = ap.parse_args()

    dates = args.dates or (["2019-01-30"] if args.quick else sorted(DAYS))
    symbols = args.symbols or (QUICK_SYMBOLS if args.quick else SYMBOLS)

    os.makedirs(args.outdir, exist_ok=True)
    manifest_path = os.path.join(args.outdir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)

    for date in dates:
        print(f"[{date}] {len(symbols)} symbols", flush=True)
        info = build_day(date, symbols, args.outdir, raw_dir=args.raw_dir)
        if info.get("skipped"):
            print(f"[{date}] already extracted", flush=True)
            continue
        info["system_events"] = [[ts, code] for ts, code in info["system_events"]]
        manifest[date] = info
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=1)
        print(
            f"[{date}] {info['n_messages'] / 1e6:.1f}M messages, "
            f"{info['n_kept'] / 1e6:.2f}M kept, {info['minutes']:.1f} min",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
