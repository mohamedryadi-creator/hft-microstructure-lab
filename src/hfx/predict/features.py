r"""The state of the book, and what happened next.

One row every half second of the session, from the snapshots
:mod:`hfx.book.replay` emits on a fixed clock, labelled with the **direction of
the next best-price change** -- an event strictly in the future of every feature
in the row.  There is no window, no smoothing across the label, and no
resampling that could let the answer leak backwards.

The features are the ones the queue-reactive model of chapter 04 has an opinion
about, plus two it does not:

===================  =========================================================
``imbalance``        :math:`(q_b-q_a)/(q_b+q_a)` at the best -- the one number
                     every practitioner plots
``log_q_bid/ask``    queue sizes in average event sizes, on a log scale
``imbalance2``       the same at one tick behind the best
``spread_ticks``     how many ticks wide the book is
``flow_fast/slow``   signed traded size with one- and ten-second memories, the
                     exponential kernels of chapter 02 used as features
``log_since``        seconds since the last best-price change
===================  =========================================================

The model sees only the first two groups.  Whether the last two add anything is
one of the things chapter 07 measures.
"""

from __future__ import annotations

import numpy as np

from ..book.replay import NQ2, TICK

FEATURES = (
    "imbalance",
    "log_q_bid",
    "log_q_ask",
    "imbalance2",
    "spread_ticks",
    "flow_fast",
    "flow_slow",
    "log_since",
)

#: Features the queue-reactive model could in principle reproduce: it knows the
#: two queues at the best and nothing else.
MODEL_FEATURES = ("imbalance", "log_q_bid", "log_q_ask")


def label_next_move(states, quotes):
    """``(keep, y)`` -- which snapshots have a next price change, and its sign.

    ``y`` is 1 when the next change takes the mid up and 0 when it takes it
    down.  Snapshots whose next change leaves the mid where it was -- a spread
    that widened symmetrically -- carry no direction and are dropped.
    """
    ts = np.asarray(states["ts"])
    q_ts = np.asarray(quotes["ts"])
    mid = (np.asarray(quotes["bid"]) + np.asarray(quotes["ask"])) / 2.0
    nxt = np.searchsorted(q_ts, ts, side="right")
    valid = (nxt >= 1) & (nxt < mid.size)
    move = np.zeros(ts.size)
    move[valid] = mid[nxt[valid]] - mid[nxt[valid] - 1]
    keep = valid & (move != 0)
    return keep, (move > 0).astype(np.int8)


def build(states, quotes, aes: float):
    """Assemble ``(X, y, extra)`` for one symbol-day.

    ``extra`` carries the queue-bucket indices on the joint grid, so the fitted
    surface can be laid next to the analytic one cell by cell.
    """
    keep, y = label_next_move(states, quotes)
    take = lambda name: np.asarray(states[name])[keep].astype(np.float64)

    q_bid, q_ask = take("q_bid") / aes, take("q_ask") / aes
    q_bid2, q_ask2 = take("q_bid2") / aes, take("q_ask2") / aes
    total = q_bid + q_ask
    total2 = q_bid2 + q_ask2
    seconds = take("ns_since_change") / 1e9

    columns = {
        "imbalance": np.where(total > 0, (q_bid - q_ask) / np.where(total > 0, total, 1), 0.0),
        "log_q_bid": np.log1p(q_bid),
        "log_q_ask": np.log1p(q_ask),
        "imbalance2": np.where(total2 > 0, (q_bid2 - q_ask2) / np.where(total2 > 0, total2, 1), 0.0),
        "spread_ticks": (take("ask") - take("bid")) / TICK,
        "flow_fast": take("flow_fast"),
        "flow_slow": take("flow_slow"),
        "log_since": np.log1p(seconds),
    }
    X = np.column_stack([columns[name] for name in FEATURES]).astype(np.float32)
    extra = {
        "bucket_bid": np.clip(q_bid.astype(int), 0, NQ2 - 1).astype(np.int16),
        "bucket_ask": np.clip(q_ask.astype(int), 0, NQ2 - 1).astype(np.int16),
        "ts": np.asarray(states["ts"])[keep],
    }
    return X, y[keep].astype(np.int8), extra


def analytic_score(surface, extra):
    """Read the model's ``P(up)`` off its surface at each observed state.

    The overflow row and column of the joint grid are excluded: they pool every
    queue above the grid and the first-passage problem was not solved on them.
    """
    b, a = extra["bucket_bid"], extra["bucket_ask"]
    inside = (b < surface.shape[0] - 1) & (a < surface.shape[1] - 1) & (b > 0) & (a > 0)
    score = np.full(b.size, np.nan)
    score[inside] = surface[b[inside], a[inside]]
    return score, inside
