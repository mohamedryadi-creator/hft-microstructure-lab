#!/usr/bin/env python
"""Turn the extracted messages into the measurements committed in ``results/``.

    python scripts/run_study.py              # every symbol-day found in data/raw
    python scripts/run_study.py --symbols SIRI INTC --dates 2019-01-30

Writes ``results/panel.csv`` (one row per symbol-day) and ``results/curves.npz``
(keyed ``SYMBOL|DATE|name``).  Both are small enough to commit, which is what
lets every notebook run offline.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hfx.pipeline.study import study_symbol_day  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_events(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return {c: table.column(c).to_numpy() for c in table.column_names}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="redo symbol-days already present")
    args = ap.parse_args()

    os.makedirs(args.results, exist_ok=True)
    panel_path = os.path.join(args.results, "panel.csv")
    curves_path = os.path.join(args.results, "curves.npz")

    done = set()
    rows = []
    if os.path.exists(panel_path) and not args.force:
        existing = pd.read_csv(panel_path)
        rows = existing.to_dict("records")
        done = {(r["symbol"], r["date"]) for r in rows}
    curves = {}
    if os.path.exists(curves_path) and not args.force:
        with np.load(curves_path) as fh:
            curves = dict(fh)

    files = sorted(glob.glob(os.path.join(args.raw, "*_events.parquet")))
    for path in files:
        name = os.path.basename(path)
        symbol, date = name.split("_")[0], name.split("_")[1]
        if args.symbols and symbol not in args.symbols:
            continue
        if args.dates and date not in args.dates:
            continue
        if (symbol, date) in done:
            continue
        t0 = time.time()
        events = load_events(path)
        row, day_curves = study_symbol_day(events, symbol, date)
        rows.append(row)
        for key, value in day_curves.items():
            curves[f"{symbol}|{date}|{key}"] = np.asarray(value)
        print(
            f"{symbol:5s} {date}  {row.get('n_trades', 0):7d} trades  "
            f"spread {row.get('mean_spread_ticks', float('nan')):5.2f} ticks  "
            f"n={row.get('hawkes_branching', float('nan')):.3f}  "
            f"eta={row.get('eta', float('nan')):.3f}  "
            f"vol={row.get('vol_preavg_pct', float('nan')):.2f}%  "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(panel_path, index=False)
        np.savez_compressed(curves_path, **curves)
    print(f"{len(rows)} symbol-days in {panel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
