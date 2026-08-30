"""Replay a day of messages through the book and collect what the study needs.

One pass, because the pass is the expensive part.  Everything downstream --
trades with their true aggressor side, the quote path, the time the book spends
in each queue state, the intensities of the queue-reactive model, the fate of
orders that joined the touch -- is accumulated here while the book is being
maintained, and comes out small enough to cache and to commit.

Two conventions are worth stating because they set the sign of every impact and
imbalance number in the repository.

* **Aggressor side.**  ``E``/``C`` executions consume a *resting* order whose
  side the book knows, so the aggressor is its opposite.  No Lee-Ready
  inference, no tick test, no attenuation from misclassification.
* **Hidden trades.**  For ``P`` messages Nasdaq documents the buy/sell
  indicator as the side of the *non-displayed order being matched*, so the
  aggressor is again the opposite.  Chapter 01 checks this empirically: with the
  convention right, 99.3% of *displayed* trades print at or through the touch,
  as an aggressive order must, while 95-99% of hidden ones print strictly
  inside the spread, as non-displayed liquidity does.

Trades are the *aggressive orders*, not the prints: several prints at one
nanosecond on one side are one market order, and ``n_prints`` says how many
resting orders it consumed.  ``hidden_size`` is the part of it that traded
against non-displayed liquidity.

Queue sizes are reported in units of the *average event size* (AES) for the
symbol-day, following Huang-Lehalle-Rosenbaum: a queue of 30 000 shares in SIRI
and one of 300 shares in AMZN are the same object once measured in the size of
the orders that actually arrive.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field

import numpy as np

from ..itch.spec import CLOSE_NS, OPEN_NS

TICK = 100                 # one cent, in the wire unit of 1/10 000 dollar
NQ = 64                    # queue-size buckets, in units of the average event size
NSZ = 48                   # event-size buckets, log-spaced in units of the AES
SIZE_LO, SIZE_HI = 0.02, 50.0
MAX_SPREAD_TICKS = 200
N_LEVELS = 2               # best, and one tick behind it
KIND_LIMIT, KIND_CANCEL, KIND_MARKET = 0, 1, 2

_A, _F, _E, _C, _X, _D, _U, _P, _Q, _H = (ord(c) for c in "AFECXDUPQH")


@dataclass
class ReplayOutput:
    """Everything one symbol-day contributes to the study."""

    symbol: str
    date: str
    trades: dict[str, np.ndarray] = field(default_factory=dict)
    quotes: dict[str, np.ndarray] = field(default_factory=dict)
    spread_time: np.ndarray = None          # ns spent at each spread in ticks
    queue_time: np.ndarray = None           # [side, bucket] ns at each best-queue size
    imbalance_time: np.ndarray = None       # ns by signed imbalance bucket
    qr_events: np.ndarray = None            # [level, side, kind, bucket] event counts
    qr_time: np.ndarray = None              # [level, side, bucket] ns in state
    qr_events2: np.ndarray = None           # [kind, side, bucket_bid, bucket_ask]
    qr_time2: np.ndarray = None             # [bucket_bid, bucket_ask]
    qr_shares: np.ndarray = None            # [level, side, kind, bucket] total shares
    size_hist: np.ndarray = None            # [kind, NSZ] event sizes in AES units
    regen_hist: np.ndarray = None           # [side, bucket] queue right after a price change
    fill_counts: np.ndarray = None          # [joined?, ahead bucket, outcome]
    minute_counts: np.ndarray = None        # [minute of session, kind] event counts
    stats: dict = field(default_factory=dict)


def _buckets(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int64)


def replay(events, symbol: str, date: str, aes: float | None = None) -> ReplayOutput:
    """Drive the book with one symbol-day of decoded messages.

    ``events`` is a mapping of the columns written by :mod:`hfx.pipeline.build`
    (``ts``, ``etype``, ``ref``, ``aux``, ``side``, ``shares``, ``price``,
    ``flag``), each a sequence indexable in order.
    """
    from .book import OrderBook

    ts_a = np.asarray(events["ts"], dtype=np.int64)
    et_a = np.asarray(events["etype"], dtype=np.uint8)
    ref_a = np.asarray(events["ref"], dtype=np.uint64)
    aux_a = np.asarray(events["aux"], dtype=np.uint64)
    sd_a = np.asarray(events["side"], dtype=np.int8)
    sh_a = np.asarray(events["shares"], dtype=np.int64)
    px_a = np.asarray(events["price"], dtype=np.int64)
    fl_a = np.asarray(events["flag"], dtype=np.uint8)

    # Average event size: the unit in which queues are measured.  Taken from the
    # displayed limit orders of the day, so it is fixed before the loop and the
    # same number is used for estimation and for simulation.
    if aes is None:
        is_add = (et_a == _A) | (et_a == _F)
        aes = float(sh_a[is_add].mean()) if is_add.any() else 100.0
    aes = max(aes, 1.0)

    book = OrderBook()
    out = ReplayOutput(symbol=symbol, date=date)

    spread_time = _buckets(MAX_SPREAD_TICKS + 2)
    queue_time = np.zeros((2, NQ), dtype=np.int64)
    imbalance_time = _buckets(21)
    qr_events = np.zeros((N_LEVELS, 2, 3, NQ), dtype=np.int64)
    qr_time = np.zeros((N_LEVELS, 2, NQ), dtype=np.int64)
    qr_events2 = np.zeros((3, 2, NQ, NQ), dtype=np.int64)
    qr_time2 = np.zeros((NQ, NQ), dtype=np.int64)
    qr_shares = np.zeros((N_LEVELS, 2, 3, NQ), dtype=np.int64)
    size_hist = np.zeros((3, NSZ), dtype=np.int64)
    regen_hist = np.zeros((2, NQ), dtype=np.int64)
    fill_counts = np.zeros((2, NQ, 3), dtype=np.int64)   # joined/improved x ahead x outcome
    minute_counts = np.zeros((391, 4), dtype=np.int64)   # add / cancel / trade / replace

    t_ts, t_px, t_sz, t_sd = array("q"), array("q"), array("q"), array("b")
    t_bid, t_ask, t_bsz, t_asz = array("q"), array("q"), array("q"), array("q")
    t_hidden, t_nprint = array("q"), array("i")
    q_ts, q_bid, q_ask, q_bsz, q_asz = (array("q") for _ in range(5))

    # order ref -> [at_best_kind, ahead_bucket, filled_any]  for orders that
    # joined or improved the touch during the session
    watched: dict[int, list] = {}

    n_unknown = 0
    n_cross = 0
    halted = False
    halt_ns = 0
    halt_start = 0

    # cached top-of-book state, valid between events
    prev_ts = 0
    bid = ask = 0
    bsz = asz = 0
    qb = [0] * N_LEVELS
    qa = [0] * N_LEVELS
    have_state = False
    quoted = False

    # One aggressive order shows up as several prints.  Nasdaq gives every print
    # its own match number -- checked: 89 735 executions, 89 735 distinct match
    # numbers on AAPL -- so the match number does *not* group a sweep.  What does
    # is the timestamp: all the executions an incoming order causes carry the
    # same nanosecond.  Prints are therefore grouped by (timestamp, aggressor
    # side), which reconstructs the market order that caused them.
    m_ts = -1
    m_side = 0
    m_px = m_sz = m_hidden_sz = m_n = 0
    m_bid = m_ask = m_bsz = m_asz = 0

    def flush_trade():
        nonlocal m_ts, m_px, m_sz, m_n, m_hidden_sz
        if m_n and m_sz:
            t_ts.append(m_ts)
            t_px.append(m_px // m_sz)
            t_sz.append(m_sz)
            t_sd.append(m_side)
            t_bid.append(m_bid)
            t_ask.append(m_ask)
            t_bsz.append(m_bsz)
            t_asz.append(m_asz)
            t_hidden.append(m_hidden_sz)
            t_nprint.append(m_n)
        m_ts = -1
        m_px = m_sz = m_n = m_hidden_sz = 0

    # Iterating a numpy array element by element costs more in scalar unboxing
    # than the whole book update does.  Convert to Python lists in bounded
    # chunks instead: list iteration hands back real ints, and the chunking
    # keeps peak memory flat on a five-million-message symbol-day.
    n = len(ts_a)
    CHUNK = 500_000
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        block = zip(
            ts_a[start:stop].tolist(),
            et_a[start:stop].tolist(),
            ref_a[start:stop].tolist(),
            aux_a[start:stop].tolist(),
            sd_a[start:stop].tolist(),
            sh_a[start:stop].tolist(),
            px_a[start:stop].tolist(),
            fl_a[start:stop].tolist(),
        )
        for ts, code, ev_ref, ev_aux, ev_side, ev_shares, ev_price, ev_flag in block:
            in_session = OPEN_NS <= ts <= CLOSE_NS

            # --- time-weighted accumulation over the interval just ended --------
            if have_state and in_session and not halted and prev_ts >= OPEN_NS:
                dt = ts - prev_ts
                if dt > 0:
                    if bid and ask:
                        s = (ask - bid) // TICK
                        spread_time[s if s <= MAX_SPREAD_TICKS else MAX_SPREAD_TICKS + 1] += dt
                        b0 = int(qb[0] / aes)
                        a0 = int(qa[0] / aes)
                        b0 = b0 if b0 < NQ else NQ - 1
                        a0 = a0 if a0 < NQ else NQ - 1
                        queue_time[0, b0] += dt
                        queue_time[1, a0] += dt
                        qr_time2[b0, a0] += dt
                        tot = qb[0] + qa[0]
                        if tot:
                            k = int((qb[0] - qa[0]) / tot * 10 + 10.5)
                            imbalance_time[k if 0 <= k <= 20 else (0 if k < 0 else 20)] += dt
                        for lv in range(N_LEVELS):
                            bl = int(qb[lv] / aes)
                            al = int(qa[lv] / aes)
                            qr_time[lv, 0, bl if bl < NQ else NQ - 1] += dt
                            qr_time[lv, 1, al if al < NQ else NQ - 1] += dt

            # --- apply the message ---------------------------------------------
            if code == _A or code == _F:
                price = ev_price
                side = ev_side
                shares = ev_shares
                best = bid if side > 0 else ask
                ahead = book.add(ev_ref, side, price, shares, ts)
                if in_session:
                    _count_event(qr_events, qr_events2, qr_shares, size_hist,
                                 KIND_LIMIT, side, price, shares, bid, ask, qb, qa, aes)
                    minute_counts[_minute(ts), 0] += 1
                    if best and (price == best or (side > 0 and price > best)
                                 or (side < 0 and price < best)):
                        joined = 0 if price == best else 1
                        ab = int(ahead / aes)
                        watched[ev_ref] = [joined, ab if ab < NQ else NQ - 1, 0]

            elif code == _X or code == _D:
                r = ev_ref
                info = book.orders.get(r)
                if info is None:
                    n_unknown += 1
                else:
                    side, price = info[0], info[1]
                    shares = ev_shares if code == _X else info[2]
                    book.reduce(r, shares)
                    if in_session:
                        _count_event(qr_events, qr_events2, qr_shares, size_hist,
                                     KIND_CANCEL, side, price, shares, bid, ask, qb, qa, aes)
                        minute_counts[_minute(ts), 1] += 1
                    if r not in book.orders:
                        w = watched.pop(r, None)
                        if w is not None and in_session:
                            fill_counts[w[0], w[1], 1 if w[2] else 2] += 1

            elif code == _U:
                old = ev_ref
                info = book.orders.get(old)
                if info is None:
                    n_unknown += 1
                else:
                    side, old_price = info[0], info[1]
                    price = ev_price
                    res = book.replace(old, ev_aux, ev_shares, price, ts)
                    if in_session:
                        _count_event(qr_events, qr_events2, qr_shares, size_hist,
                                     KIND_CANCEL, side, old_price, info[2], bid, ask, qb, qa, aes)
                        _count_event(qr_events, qr_events2, qr_shares, size_hist,
                                     KIND_LIMIT, side, price, ev_shares, bid, ask, qb, qa, aes)
                        minute_counts[_minute(ts), 3] += 1
                    w = watched.pop(old, None)
                    if w is not None and in_session:
                        fill_counts[w[0], w[1], 1 if w[2] else 2] += 1
                    if res is not None:
                        best = bid if side > 0 else ask
                        if in_session and best and (
                            price == best
                            or (side > 0 and price > best)
                            or (side < 0 and price < best)
                        ):
                            ab = int(res[1] / aes)
                            watched[ev_aux] = [
                                0 if price == best else 1, ab if ab < NQ else NQ - 1, 0
                            ]

            elif code == _E or code == _C:
                r = ev_ref
                info = book.orders.get(r)
                if info is None:
                    n_unknown += 1
                else:
                    resting_side, resting_price = info[0], info[1]
                    shares = ev_shares
                    price = ev_price if code == _C else resting_price
                    printable = (code == _E) or (ev_flag == ord("Y"))
                    aggressor = -resting_side
                    if ts != m_ts or aggressor != m_side:
                        flush_trade()
                        m_ts, m_side = ts, aggressor
                        m_bid, m_ask, m_bsz, m_asz = bid, ask, bsz, asz
                    if printable:
                        m_px += price * shares
                        m_sz += shares
                        m_n += 1
                    book.reduce(r, shares)
                    if in_session:
                        _count_event(qr_events, qr_events2, qr_shares, size_hist,
                                     KIND_MARKET, resting_side, resting_price, shares,
                                     bid, ask, qb, qa, aes)
                        minute_counts[_minute(ts), 2] += 1
                    w = watched.get(r)
                    if w is not None:
                        w[2] = 1
                        if r not in book.orders:
                            watched.pop(r, None)
                            if in_session:
                                fill_counts[w[0], w[1], 0] += 1

            elif code == _P:
                aggressor = -ev_side
                if ts != m_ts or aggressor != m_side:
                    flush_trade()
                    m_ts, m_side = ts, aggressor
                    m_bid, m_ask, m_bsz, m_asz = bid, ask, bsz, asz
                m_px += ev_price * ev_shares
                m_sz += ev_shares
                m_hidden_sz += ev_shares
                m_n += 1

            elif code == _Q:
                flush_trade()
                n_cross += 1

            elif code == _H:
                flush_trade()
                state = chr(ev_flag)
                if state in ("H", "P") and not halted:
                    halted, halt_start = True, ts
                elif state == "T" and halted:
                    halted = False
                    halt_ns += ts - halt_start

            # --- refresh the cached state --------------------------------------
            new_bid, new_bsz, new_ask, new_asz = book.top()
            if in_session:
                # The queue that greets the market on the far side of a price
                # change.  The queue-reactive model has to start its next
                # excursion somewhere, and this is where, measured rather than
                # assumed.
                if new_bid != bid and new_bid and new_bsz:
                    rb = int(new_bsz / aes)
                    regen_hist[0, rb if rb < NQ else NQ - 1] += 1
                if new_ask != ask and new_ask and new_asz:
                    ra = int(new_asz / aes)
                    regen_hist[1, ra if ra < NQ else NQ - 1] += 1
            if new_bid != bid or new_ask != ask or not quoted:
                if in_session and new_bid and new_ask:
                    quoted = True
                    q_ts.append(ts)
                    q_bid.append(new_bid)
                    q_ask.append(new_ask)
                    q_bsz.append(new_bsz)
                    q_asz.append(new_asz)
            bid, bsz, ask, asz = new_bid, new_bsz, new_ask, new_asz
            levels_bid, levels_ask = book.levels[0], book.levels[1]
            for lv in range(N_LEVELS):
                qb[lv] = levels_bid.get(bid - lv * TICK, 0) if bid else 0
                qa[lv] = levels_ask.get(ask + lv * TICK, 0) if ask else 0
            have_state = True
            prev_ts = ts

    flush_trade()

    out.trades = {
        "ts": np.frombuffer(t_ts, dtype=np.int64).copy(),
        "price": np.frombuffer(t_px, dtype=np.int64).copy(),
        "size": np.frombuffer(t_sz, dtype=np.int64).copy(),
        "side": np.frombuffer(t_sd, dtype=np.int8).copy(),
        "bid": np.frombuffer(t_bid, dtype=np.int64).copy(),
        "ask": np.frombuffer(t_ask, dtype=np.int64).copy(),
        "bid_size": np.frombuffer(t_bsz, dtype=np.int64).copy(),
        "ask_size": np.frombuffer(t_asz, dtype=np.int64).copy(),
        "hidden_size": np.frombuffer(t_hidden, dtype=np.int64).copy(),
        "n_prints": np.frombuffer(t_nprint, dtype=np.int32).copy(),
    }
    out.quotes = {
        "ts": np.frombuffer(q_ts, dtype=np.int64).copy(),
        "bid": np.frombuffer(q_bid, dtype=np.int64).copy(),
        "ask": np.frombuffer(q_ask, dtype=np.int64).copy(),
        "bid_size": np.frombuffer(q_bsz, dtype=np.int64).copy(),
        "ask_size": np.frombuffer(q_asz, dtype=np.int64).copy(),
    }
    out.spread_time = spread_time
    out.queue_time = queue_time
    out.imbalance_time = imbalance_time
    out.qr_events = qr_events
    out.qr_time = qr_time
    out.qr_events2 = qr_events2
    out.qr_time2 = qr_time2
    out.qr_shares = qr_shares
    out.size_hist = size_hist
    out.regen_hist = regen_hist
    out.fill_counts = fill_counts
    out.minute_counts = minute_counts
    out.stats = {
        "aes": aes,
        "n_events": int(n),
        "n_unknown_ref": int(n_unknown),
        "n_cross": int(n_cross),
        "halt_ns": int(halt_ns),
        "n_open_orders_at_close": len(book.orders),
        "n_watched_unresolved": len(watched),
    }
    return out


def _minute(ts: int) -> int:
    m = (ts - OPEN_NS) // 60_000_000_000
    return int(m) if 0 <= m <= 390 else 390


def size_bucket_edges():
    """Log-spaced edges for the event-size histogram, in units of the AES."""
    return np.geomspace(SIZE_LO, SIZE_HI, NSZ + 1)


def size_bucket_centres():
    edges = size_bucket_edges()
    return np.sqrt(edges[:-1] * edges[1:])


_SIZE_EDGES = np.geomspace(SIZE_LO, SIZE_HI, NSZ + 1)


def _count_event(qr_events, qr_events2, qr_shares, size_hist,
                 kind, side, price, shares, bid, ask, qb, qa, aes):
    """Attribute an event to a level, and record it against the queue state."""
    if side > 0:
        if not bid:
            return
        lv = (bid - price) // TICK
    else:
        if not ask:
            return
        lv = (price - ask) // TICK
    if lv < 0 or lv >= N_LEVELS:
        return
    s = 0 if side > 0 else 1
    q = qb[lv] if side > 0 else qa[lv]
    b = int(q / aes)
    b = b if b < NQ else NQ - 1
    qr_events[lv, s, kind, b] += 1
    qr_shares[lv, s, kind, b] += shares
    if lv == 0:
        u = shares / aes
        sb = int(np.searchsorted(_SIZE_EDGES, u, side="right")) - 1
        if 0 <= sb < NSZ:
            size_hist[kind, sb] += 1
    if lv == 0:
        b0 = int(qb[0] / aes)
        a0 = int(qa[0] / aes)
        qr_events2[kind, s, b0 if b0 < NQ else NQ - 1, a0 if a0 < NQ else NQ - 1] += 1
