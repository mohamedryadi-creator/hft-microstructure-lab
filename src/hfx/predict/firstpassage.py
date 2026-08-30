r"""How informative the book is, according to the queue-reactive model.

The model of chapter 04 says nothing directly about which way the price will go.
But it says everything about the queues, and the price moves when a queue
empties, so the two questions are the same one: starting from
:math:`(q_{\text{bid}}, q_{\text{ask}})`, **which queue empties first?**

That is a first-passage problem on a two-dimensional Markov chain, and it is
exact.  Write :math:`u(b,a)` for the probability that the ask side empties first
-- the price goes up.  Every transient state satisfies

.. math::
    u(b,a) = \sum_{(b',a')} p\big((b,a)\to(b',a')\big)\, u(b',a'),

with :math:`u = 1` on the ask-absorbed boundary and :math:`0` on the
bid-absorbed one.  The transition probabilities are the estimated intensities
divided by their sum, and the jump sizes are drawn from the measured order-size
distribution -- the same correction that chapter 04 needed to reproduce the rate
of price changes at all.

The result is a **prediction about price direction that was never fitted to a
price**.  Chapter 07 puts it next to what a classifier trained on the realised
outcomes actually finds, and the gap between them is the finding.
"""

from __future__ import annotations

import numpy as np


def jump_distribution(size_hist, centres, max_step: int = 12):
    """Order sizes, in whole queue buckets, as ``(probabilities, steps)``.

    A bucket is one average event size, so an order of 1.4 average sizes moves
    the queue by one bucket and an order of 4.6 by five.  Sizes below half a
    bucket would leave the state unchanged and stall the chain, so they are
    rounded up to one -- the alternative is a chain that never absorbs.
    """
    hist = np.asarray(size_hist, dtype=float)
    if hist.ndim > 1:
        hist = hist.sum(axis=0)
    steps = np.clip(np.round(np.asarray(centres, dtype=float)), 1, max_step).astype(int)
    weights = np.bincount(steps, weights=hist, minlength=max_step + 1)
    total = weights.sum()
    if total <= 0:
        return np.array([1.0]), np.array([1])
    probs = weights / total
    keep = probs > 1e-4
    idx = np.flatnonzero(keep)
    return probs[idx] / probs[idx].sum(), idx


def _rates(lam, lam2, b, a, side, min_seconds, time2):
    """Event rates on ``side`` in state ``(b, a)``: joint where the data
    supports it, own-queue otherwise.

    The joint tables are the model as Huang, Lehalle and Rosenbaum write it --
    the intensity at one limit depends on *both* queues -- but the far corners
    of a 24 by 24 grid are visited for seconds a day, and a rate estimated over
    seconds is noise.  Where the residence time is too short the own-queue rate
    stands in; the fraction of states that falls back is reported.
    """
    if lam2 is not None and time2 is not None and time2[b, a] >= min_seconds:
        return lam2[:, side, b, a], True
    own = b if side == 0 else a
    return lam[:, min(own, lam.shape[1] - 1)], False


def absorption_probability(
    lam,
    size_probs,
    size_steps,
    grid: int = 24,
    lam2=None,
    time2=None,
    min_seconds: float = 30.0,
):
    r"""``u[b, a]`` = P(the ask queue empties before the bid queue).

    Parameters
    ----------
    lam : (3, NQ) intensities by kind and own queue size, from
        :func:`hfx.queue.reactive.intensities`.
    size_probs, size_steps : the jump distribution from :func:`jump_distribution`.
    lam2, time2 : optional joint tables ``(3, 2, G, G)`` and ``(G, G)`` giving
        rates that depend on both queues.

    Returns
    -------
    u : (grid, grid) with ``u[0, :] = 0`` (bid gone, price down) and
        ``u[:, 0] = 1`` (ask gone, price up).
    fallback : the share of interior states that had to use the own-queue rates.
    """
    lam = np.nan_to_num(np.asarray(lam, dtype=float))
    g = int(grid)
    n = g * g
    matrix = np.zeros((n, n))
    rhs = np.zeros(n)
    fallback = 0
    interior = 0

    for b in range(1, g):
        for a in range(1, g):
            row = b * g + a
            rb, joint_b = _rates(lam, lam2, b, a, 0, min_seconds, time2)
            ra, joint_a = _rates(lam, lam2, b, a, 1, min_seconds, time2)
            interior += 1
            fallback += 0 if (joint_b and joint_a) else 1
            up_b, down_b = rb[0], rb[1] + rb[2]
            up_a, down_a = ra[0], ra[1] + ra[2]
            total = up_b + down_b + up_a + down_a
            matrix[row, row] = 1.0
            if total <= 0:
                rhs[row] = 0.5          # a state nothing ever leaves; call it even
                continue
            for prob, step in zip(size_probs, size_steps):
                # bid side
                for rate, sign in ((up_b, +1), (down_b, -1)):
                    if rate <= 0:
                        continue
                    nb = b + sign * step
                    weight = prob * rate / total
                    if nb <= 0:
                        continue         # the bid emptied: u = 0, nothing to add
                    matrix[row, min(nb, g - 1) * g + a] -= weight
                # ask side
                for rate, sign in ((up_a, +1), (down_a, -1)):
                    if rate <= 0:
                        continue
                    na = a + sign * step
                    weight = prob * rate / total
                    if na <= 0:
                        rhs[row] += weight   # the ask emptied: u = 1
                        continue
                    matrix[row, b * g + min(na, g - 1)] -= weight

    for k in range(g):
        bid_gone, ask_gone = k, k * g
        matrix[bid_gone, :] = 0.0
        matrix[bid_gone, bid_gone] = 1.0
        rhs[bid_gone] = 0.0
        matrix[ask_gone, :] = 0.0
        matrix[ask_gone, ask_gone] = 1.0
        rhs[ask_gone] = 1.0
    matrix[0, :] = 0.0
    matrix[0, 0] = 1.0
    rhs[0] = 0.5                          # both gone at once

    u = np.linalg.solve(matrix, rhs).reshape(g, g)
    return u, (fallback / interior if interior else 1.0)


def simulate_absorption(lam, size_probs, size_steps, start, n_paths: int, rng=None,
                        grid: int = 24, max_steps: int = 20_000):
    """Monte Carlo of the same chain, for checking :func:`absorption_probability`.

    Deliberately written as a plain loop over paths: it shares no code with the
    linear solve, so agreement between the two means something.
    """
    rng = np.random.default_rng() if rng is None else rng
    lam = np.nan_to_num(np.asarray(lam, dtype=float))
    size_probs = np.asarray(size_probs, dtype=float)
    size_steps = np.asarray(size_steps, dtype=int)
    cdf = np.cumsum(size_probs)
    up_count = 0
    for _ in range(n_paths):
        b, a = start
        for _step in range(max_steps):
            rb = lam[:, min(b, lam.shape[1] - 1)]
            ra = lam[:, min(a, lam.shape[1] - 1)]
            rates = np.array([rb[0], rb[1] + rb[2], ra[0], ra[1] + ra[2]])
            total = rates.sum()
            if total <= 0:
                break
            which = int(np.searchsorted(np.cumsum(rates), rng.random() * total))
            step = int(size_steps[np.searchsorted(cdf, rng.random())])
            if which == 0:
                b = min(b + step, grid - 1)
            elif which == 1:
                b -= step
            elif which == 2:
                a = min(a + step, grid - 1)
            else:
                a -= step
            if b <= 0 or a <= 0:
                up_count += int(a <= 0 and b > 0) + 0.5 * int(a <= 0 and b <= 0)
                break
    return up_count / n_paths


def imbalance_profile(u, buckets=None):
    """Collapse the surface onto queue imbalance, the one-number summary.

    ``I = (b - a) / (b + a)``, which is what every practitioner plots and what
    chapter 07's logistic baseline uses.
    """
    g = u.shape[0]
    b = np.arange(g)[:, None] * np.ones((1, g))
    a = np.ones((g, 1)) * np.arange(g)[None, :]
    total = b + a
    with np.errstate(invalid="ignore", divide="ignore"):
        imbalance = np.where(total > 0, (b - a) / np.where(total > 0, total, 1), 0.0)
    if buckets is None:
        buckets = np.linspace(-1.0, 1.0, 11)
    idx = np.clip(np.searchsorted(buckets, imbalance) - 1, 0, len(buckets) - 2)
    out = np.full(len(buckets) - 1, np.nan)
    for k in range(len(buckets) - 1):
        mask = (idx == k) & (total > 0)
        if mask.any():
            out[k] = float(u[mask].mean())
    return 0.5 * (buckets[:-1] + buckets[1:]), out
