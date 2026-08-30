#!/usr/bin/env python
"""Chapter 07: fit and evaluate the predictability of the book, out of sample.

    python scripts/run_learning.py                # every symbol
    python scripts/run_learning.py --symbols INTC SIRI

Trains on the first five ITCH days and tests on the last two.  The split is
**by day**: snapshots half a second apart are near-duplicates of each other, so
a split inside a day measures memorisation rather than prediction.

Writes ``results/learning.csv`` (one row per symbol and model) and
``results/learning.npz`` (surfaces and calibration curves), both small enough to
commit, which is what lets notebook 07 run offline.
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

from hfx.book.replay import NQ2, replay, size_bucket_centres  # noqa: E402
from hfx.predict import features as F  # noqa: E402
from hfx.predict import firstpassage as fp  # noqa: E402
from hfx.predict import models as M  # noqa: E402
from hfx.queue import reactive as qr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEST_DAYS = 2


def load_events(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return {c: table.column(c).to_numpy() for c in table.column_names}


def analytic_surface(events2, time2, events1, time1, size_hist):
    """The queue-reactive model's P(up), from pooled training-day tables only."""
    lam, _ = qr.intensities(events1, time1, min_seconds=2.0)
    probs, steps = fp.jump_distribution(size_hist, size_bucket_centres())
    seconds2 = time2 / 1e9
    with np.errstate(divide="ignore", invalid="ignore"):
        lam2 = np.where(seconds2[None, None, :, :] > 0,
                        events2 / np.maximum(seconds2, 1e-9)[None, None, :, :], np.nan)
    return fp.absorption_probability(lam, probs, steps, grid=NQ2, lam2=lam2,
                                     time2=seconds2, min_seconds=30.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=os.path.join(ROOT, "data", "raw"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(args.results, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.raw, "*_events.parquet")))
    by_symbol: dict[str, list[str]] = {}
    for path in files:
        symbol, date = os.path.basename(path).split("_")[:2]
        if args.symbols and symbol not in args.symbols:
            continue
        by_symbol.setdefault(symbol, []).append(path)

    rows, curves = [], {}
    for symbol, paths in sorted(by_symbol.items()):
        t0 = time.time()
        paths.sort()
        train_paths, test_paths = paths[:-TEST_DAYS], paths[-TEST_DAYS:]
        if not train_paths or not test_paths:
            continue

        pooled_e2 = np.zeros((3, 2, NQ2, NQ2))
        pooled_t2 = np.zeros((NQ2, NQ2))
        pooled_e1 = np.zeros((3, 64))
        pooled_t1 = np.zeros(64)
        pooled_sizes = None
        train_X, train_y, test_X, test_y, test_extra = [], [], [], [], []

        for path in paths:
            out = replay(load_events(path), symbol, os.path.basename(path).split("_")[1])
            X, y, extra = F.build(out.states, out.quotes, out.stats["aes"])
            if path in train_paths:
                train_X.append(X)
                train_y.append(y)
                pooled_e2 += out.qr_events2
                pooled_t2 += out.qr_time2
                pooled_e1 += out.qr_events[0, 0] + out.qr_events[0, 1]
                pooled_t1 += out.qr_time[0, 0] + out.qr_time[0, 1]
                pooled_sizes = out.size_hist if pooled_sizes is None else pooled_sizes + out.size_hist
            else:
                test_X.append(X)
                test_y.append(y)
                test_extra.append(extra)

        Xtr = np.concatenate(train_X)
        ytr = np.concatenate(train_y)
        Xte = np.concatenate(test_X)
        yte = np.concatenate(test_y)
        extra = {k: np.concatenate([e[k] for e in test_extra]) for k in test_extra[0]}

        surface, fallback = analytic_surface(pooled_e2, pooled_t2, pooled_e1, pooled_t1, pooled_sizes)
        score, inside = F.analytic_score(surface, extra)

        imbalance = [F.FEATURES.index("imbalance")]
        model_cols = [F.FEATURES.index(n) for n in F.MODEL_FEATURES]
        fitted = {
            "logistic_imbalance": M.Logistic().fit(Xtr[:, imbalance], ytr),
            "logistic_queues": M.Logistic().fit(Xtr[:, model_cols], ytr),
            "logistic_all": M.Logistic().fit(Xtr, ytr),
        }
        boosted = M.fit_boosted(Xtr, ytr, early_stopping=True, validation_fraction=0.15)
        boosted_queues = M.fit_boosted(Xtr[:, model_cols], ytr, early_stopping=True,
                                       validation_fraction=0.15)

        predictions = {
            "analytic": score,
            "logistic_imbalance": fitted["logistic_imbalance"].predict_proba(Xte[:, imbalance]),
            "logistic_queues": fitted["logistic_queues"].predict_proba(Xte[:, model_cols]),
            "logistic_all": fitted["logistic_all"].predict_proba(Xte),
            "boosted_queues": boosted_queues.predict_proba(Xte[:, model_cols])[:, 1],
            "boosted_all": boosted.predict_proba(Xte)[:, 1],
        }

        # Everything is scored twice: on the whole test set, and on the subset
        # where the analytic surface is defined.  Only the second is a fair
        # head-to-head, because the model has nothing to say off its own grid.
        for name, p in predictions.items():
            for scope, mask in (("all", np.ones(yte.size, bool)), ("common", inside)):
                if name == "analytic" and scope == "all":
                    continue
                metrics = M.evaluate(yte[mask], p[mask])
                rows.append({
                    "symbol": symbol, "model": name, "scope": scope,
                    "n_train": int(ytr.size), "fallback_share": float(fallback),
                    **{k: v for k, v in metrics.items() if not k.startswith("calibration")},
                })
                if scope == "common":
                    curves[f"{symbol}|{name}|calibration"] = np.column_stack([
                        metrics["calibration_predicted"], metrics["calibration_observed"],
                        metrics["calibration_counts"],
                    ])

        fitted_surface, counts = M.surface_from_scores(
            extra["bucket_bid"], extra["bucket_ask"], predictions["logistic_all"], NQ2
        )
        empirical_surface, _ = M.surface_from_scores(
            extra["bucket_bid"], extra["bucket_ask"], yte.astype(float), NQ2
        )
        curves[f"{symbol}|surface_analytic"] = surface
        curves[f"{symbol}|surface_fitted"] = fitted_surface
        curves[f"{symbol}|surface_empirical"] = empirical_surface
        curves[f"{symbol}|surface_counts"] = counts
        curves[f"{symbol}|coefficients"] = np.array(
            [fitted["logistic_all"].coef_[F.FEATURES.index(n)] for n in F.FEATURES]
        )

        common = [r for r in rows if r["symbol"] == symbol and r["scope"] == "common"]
        best = {r["model"]: r["auc"] for r in common}
        print(
            f"{symbol:5s} train {ytr.size:7d} test {yte.size:7d} common {int(inside.sum()):6d} "
            f"| AUC analytic {best.get('analytic', float('nan')):.3f} "
            f"logit-imb {best.get('logistic_imbalance', float('nan')):.3f} "
            f"boosted {best.get('boosted_all', float('nan')):.3f} "
            f"| fallback {fallback:.0%} [{time.time() - t0:.0f}s]",
            flush=True,
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(args.results, "learning.csv"), index=False)
    np.savez_compressed(os.path.join(args.results, "learning.npz"), **curves)
    np.save(os.path.join(args.results, "feature_names.npy"), np.array(F.FEATURES))
    print(f"{frame.symbol.nunique()} symbols -> {os.path.join(args.results, 'learning.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
