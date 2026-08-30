r"""The same market-making problem as a discrete-time Markov decision process.

Two reasons for building it, and neither is that dynamic programming is more
convenient than the closed form.

1. It is the **environment** the reinforcement learner acts in.  Comparing a
   learned policy with a closed form solved in continuous time would confound
   two different approximations; comparing it with the exact optimum *of its own
   environment* isolates the learning.
2. It converges to the closed form as the step shrinks, which checks the
   continuous-time algebra from the other side.

Over a step of length ``dt`` the maker quoting :math:`\delta` is filled with
probability :math:`1-e^{-Ae^{-k\delta}dt}` -- the exact probability of at least
one Poisson arrival, not its linearisation -- and both sides can fill in the
same step, which is why the two quotes do not separate exactly.
"""

from __future__ import annotations

import numpy as np


def action_grid(k: float, n: int = 25, lo: float = -2.0, hi: float = 4.0):
    """Half-spreads to choose from, in units of ``1/k``.

    The range reaches below zero on purpose: a maker holding a large short
    position bids *through* the mid to get flat, and a grid that starts at zero
    silently truncates the optimal policy at exactly the states where inventory
    risk matters most.
    """
    return np.linspace(lo, hi, n) / k


def fill_probability(delta, A: float, k: float, dt: float):
    return 1.0 - np.exp(-A * np.exp(-k * np.asarray(delta, dtype=float)) * dt)


def _backup(V, deltas, A, k, phi, dt, Q, beta):
    """One Bellman sweep; returns the new values and the greedy action indices."""
    n = 2 * Q + 1
    inventory = np.arange(-Q, Q + 1)
    p = fill_probability(deltas, A, k, dt)
    new = np.empty(n)
    best_b = np.zeros(n, dtype=np.int64)
    best_a = np.zeros(n, dtype=np.int64)
    for i, q in enumerate(inventory):
        pb = p if q < Q else np.zeros_like(p)
        pa = p if q > -Q else np.zeros_like(p)
        db = deltas if q < Q else np.zeros_like(deltas)
        da = deltas if q > -Q else np.zeros_like(deltas)
        PB, PA = pb[:, None], pa[None, :]
        reward = PB * db[:, None] + PA * da[None, :] - phi * q * q * dt
        up = V[i + 1] if q < Q else V[i]
        dn = V[i - 1] if q > -Q else V[i]
        ev = (
            (1 - PB) * (1 - PA) * V[i]
            + PB * (1 - PA) * up
            + (1 - PB) * PA * dn
            + PB * PA * V[i]
        )
        total = reward + beta * ev
        flat = int(np.argmax(total))
        best_b[i], best_a[i] = divmod(flat, total.shape[1])
        new[i] = total.flat[flat]
    return new, best_b, best_a


def solve(A, k, phi, Q: int, dt: float, deltas=None, beta: float = 1.0,
          tol: float = 1e-12, max_iter: int = 20_000):
    r"""Optimal policy of the discrete-time problem.

    ``beta = 1`` runs relative value iteration and returns the average-reward
    (ergodic) solution -- the one the continuous-time closed form describes.
    ``beta < 1`` runs ordinary discounted value iteration.

    Returns ``(delta_bid, delta_ask, values, gain)``.
    """
    deltas = action_grid(k) if deltas is None else np.asarray(deltas, float)
    n = 2 * Q + 1
    V = np.zeros(n)
    gain = 0.0
    for _ in range(max_iter):
        new, bb, ba = _backup(V, deltas, A, k, phi, dt, Q, beta)
        if beta >= 1.0:
            gain = new[Q]
            new = new - gain
        if np.max(np.abs(new - V)) < tol:
            V = new
            break
        V = new
    _new, bb, ba = _backup(V, deltas, A, k, phi, dt, Q, beta)
    delta_bid = np.where(np.arange(-Q, Q + 1) < Q, deltas[bb], np.inf)
    delta_ask = np.where(np.arange(-Q, Q + 1) > -Q, deltas[ba], np.inf)
    return delta_bid, delta_ask, V, gain / dt


def simulate_policy(delta_bid, delta_ask, A, k, phi, Q, dt, n_steps, rng=None):
    """Run a quoting policy and return its realised reward and inventory path."""
    rng = np.random.default_rng() if rng is None else rng
    i = Q
    inventory = np.empty(n_steps, dtype=np.int64)
    reward = 0.0
    for t in range(n_steps):
        q = i - Q
        pb = fill_probability(delta_bid[i], A, k, dt) if q < Q else 0.0
        pa = fill_probability(delta_ask[i], A, k, dt) if q > -Q else 0.0
        buy = rng.random() < pb
        sell = rng.random() < pa
        reward += (delta_bid[i] if buy else 0.0) + (delta_ask[i] if sell else 0.0)
        reward -= phi * q * q * dt
        i += int(buy) - int(sell)
        inventory[t] = i - Q
    return reward / (n_steps * dt), inventory
