#!/usr/bin/env python
"""Chapter 08: what a market maker earns in a book that reacts, and what sight is worth.

    python scripts/run_agents.py
    python scripts/run_agents.py --symbols INTC SIRI --deep

For each symbol the queue-reactive tables are pooled over the training days, an
environment is built from them, and two families of quoting policy are searched
over the same simulated price paths:

* **blind**   -- quote unless the inventory says otherwise.  This is the
  information the closed form of chapter 05 uses.
* **sighted** -- the same, plus a rule that refuses to bid into a thin bid queue.

The difference is the value of seeing the book.  Writes ``results/agents.csv``
and ``results/agents.npz``.
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

from hfx.book.replay import replay  # noqa: E402
from hfx.mm import queue_agent as qa  # noqa: E402
from hfx.mm.queue_env import QueueBookEnv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEST_DAYS = 2
INVENTORY_GRID = (1, 2, 3, 6)
IMBALANCE_GRID = (-1.1, -0.6, -0.4, -0.2, 0.0, 0.2)
REBATES = np.linspace(0.0, 0.005, 51)


def load_events(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return {c: table.column(c).to_numpy() for c in table.column_names}


def pooled_tables(paths, symbol):
    events = time_ns = sizes = regen = None
    for path in paths:
        out = replay(load_events(path), symbol, os.path.basename(path).split("_")[1])
        e = out.qr_events[0, 0] + out.qr_events[0, 1]
        t = out.qr_time[0, 0] + out.qr_time[0, 1]
        r = (out.regen_hist[0] + out.regen_hist[1]).astype(float)
        events = e if events is None else events + e
        time_ns = t if time_ns is None else time_ns + t
        sizes = out.size_hist if sizes is None else sizes + out.size_hist
        regen = r if regen is None else regen + r
    return events, time_ns, sizes, regen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--deep", action="store_true", help="also train the torch agent")
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)

    by_symbol: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(args.raw, "*_events.parquet"))):
        symbol = os.path.basename(path).split("_")[0]
        if args.symbols and symbol not in args.symbols:
            continue
        by_symbol.setdefault(symbol, []).append(path)

    rows, curves = [], {}
    for symbol, paths in sorted(by_symbol.items()):
        t0 = time.time()
        events, time_ns, sizes, regen = pooled_tables(sorted(paths)[:-TEST_DAYS], symbol)

        def factory(seed, symbol=symbol, events=events, time_ns=time_ns,
                    sizes=sizes, regen=regen):
            return QueueBookEnv.from_tables(
                events, time_ns, sizes, regen, dt=0.02, batch=args.batch,
                rng=np.random.default_rng(seed),
            )

        best, table = qa.policy_search(
            factory, INVENTORY_GRID, IMBALANCE_GRID, seeds=(11, 12),
            n_steps=args.steps, warmup=100,
        )
        frame = pd.DataFrame(table)
        blind = frame[~frame.sighted]
        sighted = frame[frame.sighted]
        best_blind = blind.loc[blind.reward_per_second.idxmax()]
        best_sighted = sighted.loc[sighted.reward_per_second.idxmax()]

        naive = qa.evaluate(factory(11), qa.always_at_touch, n_steps=args.steps, warmup=100)
        frontier = qa.rebate_frontier(table, REBATES)

        record = {
            "symbol": symbol,
            "q_max": float(factory(0).q_max),
            "naive_reward": naive["reward_per_second"],
            "naive_fills": naive["fills_per_second"],
            "blind_reward": float(best_blind.reward_per_second),
            "blind_se": float(best_blind.reward_se),
            "blind_inventory_max": int(best_blind.inventory_max),
            "blind_fills": float(best_blind.fills_per_second),
            "sighted_reward": float(best_sighted.reward_per_second),
            "sighted_se": float(best_sighted.reward_se),
            "sighted_inventory_max": int(best_sighted.inventory_max),
            "sighted_imbalance_min": float(best_sighted.imbalance_min),
            "sighted_fills": float(best_sighted.fills_per_second),
            "value_of_sight": float(best_sighted.reward_per_second - best_blind.reward_per_second),
            "break_even_rebate_blind": frontier["blind"]["break_even"],
            "break_even_rebate_sighted": frontier["sighted"]["break_even"],
        }

        if args.deep:
            from hfx.mm import deep

            if deep.available():
                # A smaller batch and many more gradient steps: the deep agent
                # is short of *updates*, not of transitions.
                small = QueueBookEnv.from_tables(events, time_ns, sizes, regen,
                                                 dt=0.02, batch=2048,
                                                 rng=np.random.default_rng(1))
                net, losses = deep.train(small, n_steps=6000, seed=0)
                metrics = qa.evaluate(factory(11), deep.policy_from(net),
                                      n_steps=args.steps, warmup=100)
                record["deep_reward"] = metrics["reward_per_second"]
                record["deep_fills"] = metrics["fills_per_second"]
                curves[f"{symbol}|deep_loss"] = losses

        rows.append(record)
        curves[f"{symbol}|search"] = frame[
            ["inventory_max", "imbalance_min", "reward_per_second", "reward_se",
             "fills_per_second", "inventory_abs"]
        ].to_numpy()
        for family in ("blind", "sighted"):
            curves[f"{symbol}|frontier_{family}"] = np.column_stack(
                [frontier[family]["rebates"], frontier[family]["reward"]]
            )
        print(
            f"{symbol:5s} blind {record['blind_reward'] * 1e4:+7.3f}e-4 "
            f"sighted {record['sighted_reward'] * 1e4:+7.3f}e-4 "
            f"(imb>={record['sighted_imbalance_min']:+.2f})  "
            f"sight worth {record['value_of_sight'] * 1e4:+7.3f}e-4/s  "
            f"break-even rebate blind {record['break_even_rebate_blind'] * 100:.2f} ticks "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(os.path.join(args.results, "agents.csv"), index=False)
    np.savez_compressed(os.path.join(args.results, "agents.npz"), **curves)
    print(f"{len(rows)} symbols -> {os.path.join(args.results, 'agents.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
