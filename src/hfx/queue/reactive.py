r"""The queue-reactive model: prices formed by queues emptying.

Huang, Lehalle and Rosenbaum's model contains no exogenous price process at all.
The book is a Markov jump process whose event rates depend on the queue sizes,

.. math::
    \lambda^{L}_i(q),\quad \lambda^{C}_i(q),\quad \lambda^{M}_i(q),

for limit orders, cancellations and market orders at limit :math:`i`, and the
*price* is a consequence: when a queue at the best empties, the best price on
that side moves a tick.  Volatility is then an output of the local mechanics of
the book, not an input -- which is the sense in which the model explains price
formation rather than describing it.

Estimation is the maximum likelihood estimator for a Markov jump process and
needs no numerical optimisation at all,

.. math:: \hat\lambda_e(q) = \frac{N_e(q)}{T(q)},

the number of type-:math:`e` events observed while the queue held :math:`q`,
over the time it held :math:`q`.  :mod:`hfx.book.replay` accumulates both
counters in the same pass that reconstructs the book, in units of the average
event size so that a 30 000-share queue in SIRI and a 300-share queue in AMZN
are comparable objects.

What the model is then asked to reproduce, on data it was not fitted to:

* the stationary distribution of the queue sizes,
* the daily volatility, against the realized volatility measured in chapter 03.
"""

from __future__ import annotations

import numpy as np

KIND_LIMIT, KIND_CANCEL, KIND_MARKET = 0, 1, 2


def intensities(events, time_ns, min_seconds: float = 1.0):
    r"""``lambda_e(q) = N_e(q) / T(q)`` in events per second, and its support.

    Parameters
    ----------
    events : (3, NQ) counts by kind and queue bucket.
    time_ns : (NQ,) nanoseconds spent in each bucket.

    Returns
    -------
    lam : (3, NQ) intensities, ``nan`` where the state was barely visited.
    valid : (NQ,) boolean, the buckets with enough time to estimate anything.
    """
    events = np.asarray(events, dtype=float)
    seconds = np.asarray(time_ns, dtype=float) / 1e9
    valid = seconds >= min_seconds
    lam = np.full(events.shape, np.nan)
    lam[:, valid] = events[:, valid] / seconds[valid]
    return lam, valid


def intensity_standard_error(events, time_ns):
    """Poisson standard error of each intensity, ``sqrt(N)/T``."""
    events = np.asarray(events, dtype=float)
    seconds = np.asarray(time_ns, dtype=float) / 1e9
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(events) / seconds


def birth_death_invariant(lam_up, lam_down):
    r"""Invariant law of a birth-death queue, :math:`\pi(q)\propto\prod_{k\le q}
    \frac{\lambda^{\uparrow}(k-1)}{\lambda^{\downarrow}(k)}`.

    States with no estimate are truncated away: the product is taken up to the
    last bucket where both rates are finite and the downward one is positive.
    """
    up = np.asarray(lam_up, dtype=float)
    down = np.asarray(lam_down, dtype=float)
    n = up.size
    ratios = np.ones(n)
    last = 0
    for q in range(1, n):
        if not np.isfinite(up[q - 1]) or not np.isfinite(down[q]) or down[q] <= 0:
            break
        ratios[q] = ratios[q - 1] * up[q - 1] / down[q]
        last = q
    pi = np.zeros(n)
    pi[: last + 1] = ratios[: last + 1]
    total = pi.sum()
    return pi / total if total > 0 else pi


class SizeSampler:
    """Draw event sizes, in units of the average event size, from a histogram.

    The textbook queue-reactive model moves the queue by exactly one average
    event.  Real order sizes are far from constant -- a hundred-share round lot
    next to a five-thousand-share block -- and the difference is not cosmetic:
    a queue that empties in one large cancellation empties much sooner than a
    diffusive count of average events suggests.
    """

    def __init__(self, counts, centres):
        counts = np.asarray(counts, dtype=float)
        total = counts.sum()
        self.centres = np.asarray(centres, dtype=float)
        self.probs = counts / total if total > 0 else np.ones(counts.size) / counts.size
        self._cdf = np.cumsum(self.probs)

    @property
    def mean(self) -> float:
        return float(self.probs @ self.centres)

    def draw(self, u):
        return self.centres[np.searchsorted(self._cdf, u)]


def simulate(lam, samplers, regen, n_moves: int, rng=None, block: int = 65_536,
             max_steps: int = 8_000_000):
    r"""Run the two-queue model until ``n_moves`` price changes have happened.

    Each side is a Markov jump process: at queue size :math:`q` (in units of the
    average event size) limit orders arrive at :math:`\lambda^L(q)`,
    cancellations at :math:`\lambda^C(q)` and market orders at
    :math:`\lambda^M(q)`, and each event changes the queue by a size drawn from
    that event type's measured distribution.  The price moves a tick when a
    queue empties, and both queues are then redrawn from ``regen`` -- the
    distribution of the best queue *measured* immediately after a real price
    change, so the model is closed with data rather than with an assumption.

    Parameters
    ----------
    lam : (3, NQ) intensities per second by kind and queue bucket.
    samplers : three :class:`SizeSampler`, one per kind.
    regen : (NQ,) probabilities for the queue size after a price change.

    ``max_steps`` bounds the work: a queue whose estimated intensities almost
    balance can take tens of millions of events to empty, and a study that hangs
    on one symbol is worse than one that reports fewer moves for it.  The
    returned arrays are truncated to the moves actually completed.

    Returns
    -------
    directions, holding_times, visited : the sign of each price move, the
    seconds between moves, and a sample of the queue sizes visited.
    """
    rng = np.random.default_rng() if rng is None else rng
    lam = np.nan_to_num(np.asarray(lam, dtype=float), nan=0.0)
    nq = lam.shape[1]
    regen = np.asarray(regen, dtype=float)
    regen = regen / regen.sum()
    regen_cdf = np.cumsum(regen)
    centres = np.arange(nq) + 0.5

    def draw_queue(u):
        return float(centres[int(np.searchsorted(regen_cdf, u))])

    directions = np.empty(n_moves, dtype=np.int8)
    holding = np.empty(n_moves)
    visited = []

    qb = draw_queue(rng.random())
    qa = draw_queue(rng.random())
    t = 0.0
    move = 0
    steps = 0
    pool_u = rng.random(block)
    pool_e = rng.exponential(size=block)
    pool_s = rng.random(block)
    cursor = block
    while move < n_moves and steps < max_steps:
        steps += 1
        if cursor >= block:
            pool_u = rng.random(block)
            pool_e = rng.exponential(size=block)
            pool_s = rng.random(block)
            cursor = 0
        bb = int(qb) if qb < nq else nq - 1
        ba = int(qa) if qa < nq else nq - 1
        rb, ra = lam[:, bb], lam[:, ba]
        total = rb[0] + rb[1] + rb[2] + ra[0] + ra[1] + ra[2]
        if total <= 0:
            qb, qa = draw_queue(rng.random()), draw_queue(rng.random())
            continue
        t += pool_e[cursor] / total
        u = pool_u[cursor] * total
        s = pool_s[cursor]
        cursor += 1
        if u < rb[0] + rb[1] + rb[2]:
            side, rates, q = 0, rb, qb
        else:
            side, rates, q = 1, ra, qa
            u -= rb[0] + rb[1] + rb[2]
        if u < rates[0]:
            q += samplers[0].draw(s)
        elif u < rates[0] + rates[1]:
            q -= samplers[1].draw(s)
        else:
            q -= samplers[2].draw(s)
        if side == 0:
            qb = q
            if len(visited) < 500_000:
                visited.append(qb)
        else:
            qa = q
        if qb <= 0 or qa <= 0:
            directions[move] = -1 if qb <= 0 else +1
            holding[move] = t
            t = 0.0
            move += 1
            qb = draw_queue(rng.random())
            qa = draw_queue(rng.random())
    return directions[:move], holding[:move], np.asarray(visited)


def implied_volatility(holding_times, tick: float, price: float, seconds: float = 23_400.0):
    r"""Daily volatility implied by the rate at which the price changes.

    Successive price moves are :math:`\pm\alpha` with i.i.d. holding times, so
    the variance accumulated over a session of ``seconds`` is
    :math:`\alpha^2\,\text{seconds}/\mathbb{E}[T]`, and the volatility in
    returns divides by the price.  No price dynamics were assumed anywhere: the
    number came out of the queues.
    """
    mean_holding = float(np.mean(holding_times))
    variance = tick**2 * seconds / mean_holding
    return float(np.sqrt(variance) / price)
