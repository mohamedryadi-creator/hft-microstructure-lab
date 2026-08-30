"""Load the committed measurements.  Nothing here touches the network."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS = os.path.join(ROOT, "results")


def panel(path: str | None = None) -> pd.DataFrame:
    """One row per symbol-day, sorted by price -- which is the tick axis."""
    df = pd.read_csv(path or os.path.join(RESULTS, "panel.csv"))
    return df.sort_values(["price", "date"]).reset_index(drop=True)


def curves(path: str | None = None) -> dict:
    with np.load(path or os.path.join(RESULTS, "curves.npz")) as fh:
        return dict(fh)


def curve(store: dict, symbol: str, date: str, name: str):
    return store[f"{symbol}|{date}|{name}"]


def symbol_days(store: dict, name: str):
    """``[(symbol, date), ...]`` for which ``name`` was computed."""
    out = []
    for key in store:
        sym, date, field = key.split("|")
        if field == name:
            out.append((sym, date))
    return sorted(set(out))
