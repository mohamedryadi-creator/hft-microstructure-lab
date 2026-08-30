r"""The uncertainty-zones model of a price living on a tick grid.

Robert and Rosenbaum's model says something sharper than "the observed price is
the efficient price plus noise".  It says *where the efficient price is* when the
observed one moves.

The traded price sits on the grid :math:`\alpha\mathbb{Z}`.  Around each
mid-tick value :math:`\alpha(k+\tfrac12)` there is an *uncertainty zone* of
half-width :math:`\eta\alpha`, and the traded price only moves from
:math:`\alpha k` to :math:`\alpha(k+1)` once the efficient price :math:`X` has
crossed the far edge of that zone, at :math:`\alpha(k+\tfrac12+\eta)`.  So
:math:`\eta` measures the market's reluctance to change the printed price:
:math:`\eta=0` is plain rounding to the nearest tick, and at :math:`\eta=1/2` a
whole tick has to be traversed.

Two consequences follow, and both are testable.

**Estimating** :math:`\eta`.  Just after an upward move to :math:`\alpha(k+1)`,
:math:`X` sits exactly on the barrier :math:`\alpha(k+\tfrac12+\eta)`.  The next
move up needs :math:`X` to travel :math:`\alpha`; the next move down needs it to
travel :math:`2\eta\alpha`.  For a continuous martingale between two barriers the
probability of reaching one first is the *other* one's distance over the total,
so

.. math::
    \frac{\mathbb{P}(\text{continuation})}{\mathbb{P}(\text{alternation})}
    = \frac{2\eta\alpha}{\alpha} = 2\eta
    \quad\Longrightarrow\quad \hat\eta = \frac{N_c}{2N_a}.

**Recovering the efficient price.**  At every price change the efficient price
is known *exactly*: :math:`X = P - d\,\alpha(\tfrac12-\eta)` where :math:`d=\pm1`
is the direction of the change.  So the integrated variance can be estimated by
the plain realized variance of that reconstructed series -- the microstructure
noise is not averaged away, it is *removed*, because the model says what it was.

The same identity gives the implicit spread :math:`\alpha(1-2\eta)`: the round
trip a taker pays against the efficient price, which for a large-tick asset is
strictly smaller than the quoted one tick.
"""

from __future__ import annotations

import numpy as np


def simulate(efficient_path, tick: float, eta: float, start_level: int | None = None):
    r"""Traded price on the tick grid implied by a path of the efficient price.

    Returns ``(levels, changed)``: the integer tick level at each point of the
    input path, and a boolean mask of the points where it changed.
    """
    x = np.asarray(efficient_path, dtype=float)
    if not 0.0 <= eta < 1.0:
        raise ValueError("eta must lie in [0, 1)")
    k = int(np.round(x[0] / tick)) if start_level is None else int(start_level)
    levels = np.empty(x.size, dtype=np.int64)
    changed = np.zeros(x.size, dtype=bool)
    half = 0.5 + eta
    for i, xi in enumerate(x):
        moved = False
        while xi >= (k + half) * tick:
            k += 1
            moved = True
        while xi <= (k - half) * tick:
            k -= 1
            moved = True
        levels[i] = k
        changed[i] = moved
    return levels, changed


def estimate_eta(levels) -> tuple[float, int, int]:
    r"""``(eta_hat, n_continuations, n_alternations)`` from a traded-price path.

    Only consecutive changes of exactly one tick are used: the derivation is a
    two-barrier argument about one-tick moves, and a multi-tick jump means the
    price was not observed while it crossed.
    """
    lv = np.asarray(levels, dtype=np.int64)
    d = np.diff(lv)
    d = d[d != 0]
    if d.size < 2:
        return float("nan"), 0, 0
    a, b = d[:-1], d[1:]
    one_tick = (np.abs(a) == 1) & (np.abs(b) == 1)
    n_cont = int(np.sum(one_tick & (a == b)))
    n_alt = int(np.sum(one_tick & (a == -b)))
    if n_alt == 0:
        return float("nan"), n_cont, n_alt
    return n_cont / (2.0 * n_alt), n_cont, n_alt


def efficient_price_at_changes(levels, tick: float, eta: float):
    r"""``(index, x_hat)`` -- the efficient price at each price change.

    :math:`\hat X = \alpha\big(k - d(\tfrac12-\eta)\big)` with :math:`d` the
    direction of the change.  Exact under the model, for a change of any size:
    the last barrier crossed is always the one just behind the new level.
    """
    lv = np.asarray(levels, dtype=np.int64)
    idx = np.flatnonzero(np.diff(lv)) + 1
    if idx.size == 0:
        return idx, np.empty(0)
    direction = np.sign(np.diff(lv)[idx - 1]).astype(float)
    return idx, tick * (lv[idx] - direction * (0.5 - eta))


def integrated_variance(levels, tick: float, eta: float | None = None) -> float:
    """Integrated variance from the reconstructed efficient price."""
    if eta is None:
        eta = estimate_eta(levels)[0]
    _idx, x_hat = efficient_price_at_changes(levels, tick, eta)
    if x_hat.size < 2:
        return float("nan")
    return float(np.sum(np.diff(x_hat) ** 2))


def variance_inflation(eta: float) -> float:
    r"""How much the realized variance of the *grid* price overstates :math:`IV`.

    Between two price changes the efficient price is a martingale started on a
    barrier, absorbed at :math:`+\alpha` (continue) or :math:`-2\eta\alpha`
    (reverse).  Optional stopping gives
    :math:`\mathbb{E}[\langle X\rangle_\tau]=\mathbb{E}[X_\tau^2]=2\eta\alpha^2`,
    so a day carrying :math:`IV` shows about :math:`IV/(2\eta\alpha^2)` one-tick
    changes and a naive realized variance of :math:`\alpha^2` per change:

    .. math:: \frac{RV_{\text{grid}}}{IV} \simeq \frac{1}{2\eta}.

    At :math:`\eta=1/2` the grid price is an unbiased random walk and the ratio
    is one; at :math:`\eta=0.1` the tick alone inflates the variance fivefold.
    It also gives a free consistency check on real data, where both sides are
    measurable.
    """
    return float("inf") if eta <= 0 else 1.0 / (2.0 * eta)


def implicit_spread(tick: float, eta: float) -> float:
    r"""The round-trip cost against the efficient price, :math:`\alpha(1-2\eta)`.

    A buy that moves the price up prints at :math:`X+\alpha(\tfrac12-\eta)`; a
    sell that moves it down prints at :math:`X-\alpha(\tfrac12-\eta)`.  For a
    large-tick asset whose quoted spread is pinned at one tick, this says the
    quoted spread overstates what a taker actually pays, by a factor
    :math:`1/(1-2\eta)`.
    """
    return tick * (1.0 - 2.0 * eta)
