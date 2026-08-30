r"""Tabular Q-learning for the market-making MDP, with a benchmark it must pass.

The point of this module is not that reinforcement learning can make markets.
It is that an agent which cannot recover a policy we can write down has not
learned anything, and no result it produces on data where we *cannot* write the
answer down should be believed.  So the learner is given the environment of
:mod:`hfx.mm.mdp`, whose optimum is computed exactly by dynamic programming, and
is required to find it.

Learning is batched: many independent copies of the environment share one table.
That is still Q-learning -- each transition is a genuine sample of the
environment and the update is the standard one -- but it lets the table see
millions of transitions in seconds of numpy instead of minutes of Python.
"""

from __future__ import annotations

import numpy as np

from .mdp import action_grid, fill_probability


def q_learning(
    A, k, phi, Q: int, dt: float,
    deltas=None,
    beta: float = 0.99,
    n_iter: int = 80_000,
    batch: int = 256,
    lr0: float = 1.0,
    epsilon0: float = 0.6,
    rng=None,
):
    """Learn the quoting policy from simulated experience.

    Returns ``(delta_bid, delta_ask, table)`` with ``table`` of shape
    ``(2Q+1, m, m)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    deltas = action_grid(k) if deltas is None else np.asarray(deltas, float)
    m = deltas.size
    n = 2 * Q + 1
    table = np.zeros((n, m, m))
    counts = np.zeros((n, m, m))
    p_fill = fill_probability(deltas, A, k, dt)
    inventory = np.arange(-Q, Q + 1)

    state = np.full(batch, Q, dtype=np.int64)
    for it in range(n_iter):
        frac = it / n_iter
        eps = epsilon0 * (1.0 - 0.98 * frac)

        flat_best = table.reshape(n, m * m).argmax(axis=1)
        best_b, best_a = np.divmod(flat_best, m)
        ab = np.where(rng.random(batch) < eps, rng.integers(0, m, batch), best_b[state])
        aa = np.where(rng.random(batch) < eps, rng.integers(0, m, batch), best_a[state])

        q = inventory[state]
        can_buy = q < Q
        can_sell = q > -Q
        buy = (rng.random(batch) < p_fill[ab]) & can_buy
        sell = (rng.random(batch) < p_fill[aa]) & can_sell
        reward = buy * deltas[ab] + sell * deltas[aa] - phi * q * q * dt
        nxt = state + buy.astype(np.int64) - sell.astype(np.int64)

        target = reward + beta * table.reshape(n, m * m)[nxt].max(axis=1)
        current = table[state, ab, aa]
        # A per-cell Robbins-Monro step, 1/(1+visits)^0.7.  A single global
        # schedule either crawls in the states the policy visits constantly or
        # never settles in the ones it visits rarely, and market making has
        # both: the inventory sits near zero and the interesting quotes are at
        # the edges.
        np.add.at(counts, (state, ab, aa), 1.0)
        lr = lr0 / (1.0 + counts[state, ab, aa]) ** 0.7
        np.add.at(table, (state, ab, aa), lr * (target - current))
        state = nxt
        # Restart the occasional path so the tails of the inventory range keep
        # being visited even once the policy has learned to avoid them.
        reset = rng.random(batch) < 0.01
        if reset.any():
            state[reset] = rng.integers(0, n, int(reset.sum()))

    flat_best = table.reshape(n, m * m).argmax(axis=1)
    best_b, best_a = np.divmod(flat_best, m)
    delta_bid = np.where(inventory < Q, deltas[best_b], np.inf)
    delta_ask = np.where(inventory > -Q, deltas[best_a], np.inf)
    return delta_bid, delta_ask, table
