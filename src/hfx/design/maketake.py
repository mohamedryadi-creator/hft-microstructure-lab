r"""Make-take fees as a Stackelberg game between an exchange and its maker.

The exchange (the *principal*) cannot quote.  It can only change what quoting
pays, and then live with whatever the market maker (the *agent*) does next.  So
the design problem is genuinely two-level: the exchange picks a contract, the
maker re-optimises, and the exchange's revenue is whatever the maker's new
behaviour produces.  This is the structure of El Euch, Mastrolia, Rosenbaum and
Tan's *Optimal make-take fees for market making regulation*, in the tractable
case where the contract is a per-fill rebate.

**The agent.**  With a rebate :math:`z` per fill the maker captures
:math:`\delta+z` and its Hamilton-Jacobi-Bellman equation is the one from
chapter 05 with the inner maximisation

.. math::
    \sup_\delta Ae^{-k\delta}(\delta+z+\Delta)
    \;\Longrightarrow\;
    \delta^\star = \tfrac1k - z - \Delta,
    \qquad \text{value } \tfrac{A}{ek}e^{k(z+\Delta)} ,

so a rebate moves the myopic part of every quote inward by :math:`z` and scales
the linearised system's coupling by :math:`e^{kz}`, which moves the inventory
skew :math:`\Delta` as well.  Individual quotes therefore shift by amounts that
vary across the inventory range -- the schedule tilts rather than translates --
but the **round trip** is exact:

.. math::
    \delta^b(q)+\delta^a(q+1) = \tfrac2k - 2z \quad\text{for every } q,

since the inventory terms cancel between a buy at :math:`q` and the sale that
undoes it.  The whole rebate reaches the price of liquidity; what the skew
decides is *when* the maker earns it.  The maker's ergodic gain is
:math:`\lambda_{\max}(M)/k`, the principal eigenvalue of the same tridiagonal
generator, which :mod:`hfx.mm.mdp` reproduces by value iteration.

**The principal.**  The exchange charges takers :math:`c` per fill and pays the
maker :math:`z`, so it earns :math:`(c-z)` on every trade and trades happen at
the rate the maker's new quotes produce.  Since the fill rate at the optimum is
:math:`\Lambda(\delta^\star)=\tfrac{A}{e}e^{k(z+\Delta)}`, the revenue behaves
like :math:`(c-z)e^{kz}` and the first-order condition gives a rebate that does
not depend on the maker's risk aversion, its inventory limit, or the volatility
at all:

.. math:: z^\star = c - \frac1k .

Everything else -- the inventory distribution, the participation constraint, a
regulator's taste for a tight spread -- perturbs that number, and the module
computes the perturbed optimum numerically so the two can be compared.
"""

from __future__ import annotations

import numpy as np

from ..mm.glft import parameters, system_matrix


class MakerSolution:
    """What the maker does under a given contract, and what it is worth."""

    def __init__(self, rebate, bid, ask, values, gain, stationary, fill_rate, spread):
        self.rebate = float(rebate)
        self.bid = bid
        self.ask = ask
        self.values = values
        self.gain = float(gain)
        self.stationary = stationary
        self.fill_rate = float(fill_rate)
        self.spread = float(spread)

    def __repr__(self) -> str:
        return (
            f"MakerSolution(z={self.rebate:.4f}, spread={self.spread:.4f}, "
            f"fills/s={self.fill_rate:.4f}, gain={self.gain:.4f})"
        )


def maker_solution(A, k, phi, Q: int, rebate: float = 0.0, gamma: float = 0.0,
                   sigma: float = 0.0) -> MakerSolution:
    r"""Solve the maker's ergodic problem under a per-fill rebate ``z``.

    ``rebate = 0`` reproduces chapter 05 exactly -- same eigen-problem, same
    quotes -- which is how this module is checked.
    """
    alpha, eta, floor = parameters(A, k, gamma, sigma, phi)
    eta = eta * np.exp(k * rebate)
    M = system_matrix(Q, alpha, eta)
    values, vectors = np.linalg.eig(M)
    j = int(np.argmax(values.real))
    lam = float(values[j].real)
    v = np.abs(vectors[:, j].real)
    v = v / v.max()

    n = 2 * Q + 1
    bid = np.full(n, np.inf)
    ask = np.full(n, np.inf)
    bid[:-1] = floor - rebate + np.log(v[:-1] / v[1:]) / k
    ask[1:] = floor - rebate + np.log(v[1:] / v[:-1]) / k

    # Inventory is a birth-death chain: up at the bid fill rate, down at the ask.
    up = np.where(np.isfinite(bid), A * np.exp(-k * np.where(np.isfinite(bid), bid, 0.0)), 0.0)
    down = np.where(np.isfinite(ask), A * np.exp(-k * np.where(np.isfinite(ask), ask, 0.0)), 0.0)
    pi = np.ones(n)
    for i in range(1, n):
        pi[i] = pi[i - 1] * up[i - 1] / down[i]
    pi /= pi.sum()

    fill_rate = float(np.sum(pi * (up + down)))
    spread = float(np.sum(pi[:-1] * bid[:-1]) + np.sum(pi[1:] * ask[1:]))
    return MakerSolution(rebate, bid, ask, v, lam / k, pi, fill_rate, spread)


def exchange_gain(solution: MakerSolution, taker_fee: float,
                  spread_weight: float = 0.0) -> float:
    r"""Revenue per unit time, optionally net of a regulator's spread penalty.

    ``spread_weight`` is the weight a *regulator* would put on market quality:
    with it at zero the exchange is a pure profit maximiser, and raising it
    pushes the optimal rebate up -- the exchange is paid to buy a tighter
    spread for the takers.
    """
    return (taker_fee - solution.rebate) * solution.fill_rate - spread_weight * solution.spread


def optimal_rebate(A, k, phi, Q: int, taker_fee: float, spread_weight: float = 0.0,
                   participation: float | None = None, grid=None):
    r"""Maximise the exchange's objective over the rebate.

    ``participation`` is the maker's outside option: contracts whose ergodic
    gain falls below it are unavailable, because the maker would walk away.

    Returns ``(z_star, solution, grid, objective)``.
    """
    grid = np.linspace(-1.0 / k, taker_fee, 401) if grid is None else np.asarray(grid, float)
    objective = np.full(grid.size, -np.inf)
    solutions = []
    for i, z in enumerate(grid):
        sol = maker_solution(A, k, phi, Q, rebate=float(z))
        solutions.append(sol)
        if participation is not None and sol.gain < participation:
            continue
        objective[i] = exchange_gain(sol, taker_fee, spread_weight)
    best = int(np.argmax(objective))
    return float(grid[best]), solutions[best], grid, objective
