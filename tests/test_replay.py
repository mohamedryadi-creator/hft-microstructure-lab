"""Replay against a book small enough to check by hand.

The scenario is built with the synthetic ITCH encoder and decoded by the real
decoder, so the test covers the whole path -- bytes to trades -- rather than the
replay in isolation.  Every expected number below was worked out from the eight
messages, not read off a first run.
"""

import numpy as np

from hfx.book.replay import replay
from hfx.itch import synth
from hfx.itch.reader import ItchExtractor
from hfx.itch.spec import OPEN_NS

S = 1_000_000_000  # one second in nanoseconds
BID, ASK = 1_500_000, 1_500_200  # $150.00 and $150.02, wire units of 1/10 000


def scenario():
    m = [
        synth.stock_directory(7, "TEST"),
        # Pre-open: the book exists before the session and must be carried in.
        synth.add_order(7, OPEN_NS - 60 * S, 3, -1, 300, ASK),
        synth.add_order(7, OPEN_NS - 60 * S, 1, +1, 100, BID),
        # In session.
        synth.add_order(7, OPEN_NS + 1 * S, 2, +1, 200, BID),
        synth.execute(7, OPEN_NS + 2 * S, 3, 150, match=1),
        synth.cancel(7, OPEN_NS + 3 * S, 1, 100),
        synth.replace(7, OPEN_NS + 4 * S, 2, 4, 200, BID + 100),
        synth.hidden_trade(7, OPEN_NS + 5 * S, 0, +1, 400, BID + 150, match=2),
        synth.delete(7, OPEN_NS + 6 * S, 4),
        synth.delete(7, OPEN_NS + 7 * S, 3),
    ]
    ext = ItchExtractor(["TEST"])
    ext.feed(synth.frame(m))
    return {k: np.asarray(v) for k, v in ext.buffers["TEST"].as_dict().items()}


def test_trades_carry_the_true_aggressor_side():
    out = replay(scenario(), "TEST", "2019-01-30")
    tr = out.trades
    assert len(tr["ts"]) == 2

    # A market buy lifting the displayed offer.  The resting order was a sell,
    # so the aggressor is a buyer: +1.
    assert tr["ts"][0] == OPEN_NS + 2 * S
    assert (tr["price"][0], tr["size"][0], tr["side"][0]) == (ASK, 150, +1)
    assert tr["hidden_size"][0] == 0
    # State *before* the trade, which is what impact and effective spread need.
    assert (tr["bid"][0], tr["ask"][0]) == (BID, ASK)
    assert (tr["bid_size"][0], tr["ask_size"][0]) == (300, 300)

    # A hidden execution.  ITCH gives the side of the non-displayed order that
    # was matched -- a buy -- so the aggressor is a seller: -1.
    assert tr["ts"][1] == OPEN_NS + 5 * S
    assert (tr["price"][1], tr["size"][1], tr["side"][1]) == (BID + 150, 400, -1)
    assert tr["hidden_size"][1] == 400
    assert (tr["bid"][1], tr["ask"][1]) == (BID + 100, ASK)

    assert out.stats["n_unknown_ref"] == 0


def test_quote_path_records_best_price_changes():
    out = replay(scenario(), "TEST", "2019-01-30")
    q = out.quotes
    # One row at the first in-session event, one when the replace improves the
    # bid by a tick.  Neither of the two deletes at the end qualifies: each
    # empties a side, and a one-sided book has no spread.
    assert list(q["ts"]) == [OPEN_NS + 1 * S, OPEN_NS + 4 * S]
    assert list(q["bid"]) == [BID, BID + 100]
    assert list(q["ask"]) == [ASK, ASK]
    assert list(q["bid_size"]) == [300, 200]
    assert list(q["ask_size"]) == [300, 150]


def test_time_weighted_spread_matches_the_intervals_by_hand():
    out = replay(scenario(), "TEST", "2019-01-30")
    # Three seconds at a two-tick spread, then two at one tick.  The last
    # second has no bid at all and is not counted anywhere.
    assert out.spread_time[2] == 3 * S
    assert out.spread_time[1] == 2 * S
    assert out.spread_time.sum() == 5 * S


def test_queue_reactive_counts_land_in_the_right_cells():
    out = replay(scenario(), "TEST", "2019-01-30")
    aes = out.stats["aes"]
    assert aes == (300 + 100 + 200) / 3  # mean displayed add size

    limit, cancel, market = 0, 1, 2
    bid_side, ask_side = 0, 1
    ev = out.qr_events
    # The in-session add of 200 shares joins a best bid holding 100: bucket 0.
    assert ev[0, bid_side, limit, 0] == 1
    # The execution consumes the best offer, which was holding 300: bucket 1.
    assert ev[0, ask_side, market, 1] == 1
    # Three cancels land on the best bid with 300, 200 and 200 resting -- the
    # explicit cancel, the cancelled leg of the replace, and the delete of the
    # replacement, which by then is itself the best bid.  All in bucket 1.
    assert ev[0, bid_side, cancel, 1] == 3
    assert ev[0, bid_side, cancel, :].sum() == 3
    # The delete that empties the offer sees 150 resting: bucket 0.
    assert ev[0, ask_side, cancel, 0] == 1
    # The replace re-posts a tick inside the old best.  That price is not one of
    # the levels being tracked at the time, so it is not counted as a limit
    # order anywhere -- price-improving orders sit outside the model.
    assert ev[0, bid_side, limit, :].sum() == 1


def test_fill_outcomes_follow_orders_that_joined_the_touch():
    out = replay(scenario(), "TEST", "2019-01-30")
    filled, part_cancel, cancel_unfilled = 0, 1, 2
    joined, improved = 0, 1
    # Order 2 joined the best bid behind 100 shares and was cancel-replaced away
    # without ever trading.
    assert out.fill_counts[joined, 0, cancel_unfilled] == 1
    # Its replacement improved the best bid, and was deleted unfilled.
    assert out.fill_counts[improved, 0, cancel_unfilled] == 1
    assert out.fill_counts[:, :, filled].sum() == 0
    assert out.fill_counts[:, :, part_cancel].sum() == 0
    assert out.stats["n_watched_unresolved"] == 0


def test_minute_counts_bin_the_session():
    out = replay(scenario(), "TEST", "2019-01-30")
    add, cancel, trade, replace_ = 0, 1, 2, 3
    assert out.minute_counts[0, add] == 1
    assert out.minute_counts[0, cancel] == 3      # one X and two D
    assert out.minute_counts[0, trade] == 1       # the hidden print is not a book event
    assert out.minute_counts[0, replace_] == 1
    assert out.minute_counts[1:].sum() == 0
