r"""Optimal market making: the closed form, derived rather than quoted.

A market maker posts a bid :math:`\delta^b` and an ask :math:`\delta^a` away
from the mid, is filled at intensity :math:`\Lambda(\delta)=Ae^{-k\delta}`, and
carries the inventory :math:`q` it accumulates while the mid diffuses with
volatility :math:`\sigma`.  Two standard objectives, and both reduce to the same
linear system.

**Risk-neutral with a running inventory penalty** :math:`\phi q^2` (Cartea-
Jaimungal, Guéant).  Writing the value as :math:`x+qS+\theta_q`, the
Hamilton-Jacobi-Bellman equation is

.. math::
    \theta_q' - \phi q^2 + \sup_\delta Ae^{-k\delta}(\delta+\theta_{q+1}-\theta_q)
    + \sup_\delta Ae^{-k\delta}(\delta+\theta_{q-1}-\theta_q) = 0 .

The inner maximisation is explicit, :math:`\delta^\star = 1/k -
(\theta_{q\pm1}-\theta_q)`, with value :math:`\frac{A}{ek}e^{k\Delta}`, so the
substitution :math:`v_q = e^{k\theta_q}` **linearises** the equation:

.. math:: v_q' = k\phi q^2 v_q - \tfrac{A}{e}\,(v_{q+1}+v_{q-1}) .

**Exponential utility** :math:`-e^{-\gamma(X_T+q S_T)}` (Avellaneda-Stoikov,
Guéant-Lehalle-Fernandez-Tapia).  The same computation gives
:math:`\delta^\star = \tfrac1\gamma\ln(1+\tfrac\gamma k) -
(\theta_{q\pm1}-\theta_q)` and the same linear system with

.. math::
    \alpha = \tfrac{k\gamma\sigma^2}{2}, \qquad
    \eta = A\Big(1+\tfrac{\gamma}{k}\Big)^{-(1+k/\gamma)} .

Note that the risk-neutral case is the limit :math:`\gamma\to0` with
:math:`\phi = \gamma\sigma^2/2` held fixed: :math:`\eta\to A/e` and
:math:`\tfrac1\gamma\ln(1+\gamma/k)\to 1/k`.

In the long-horizon limit the quotes stop depending on time and are read off the
**principal eigenvector** of the tridiagonal generator, which is what
:func:`stationary_quotes` returns.  Nothing here is asymptotic in the fills or
the inventory: the only approximation is that :math:`T` is far away.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def system_matrix(Q: int, alpha: float, eta: float) -> np.ndarray:
    r"""The generator of :math:`\dot v = -\alpha q^2 v + \eta(v_{q+1}+v_{q-1})`.

    Inventory runs over :math:`-Q..Q`; the maker cannot buy at :math:`+Q` or
    sell at :math:`-Q`, which is imposed by the missing off-diagonal entries.
    """
    q = np.arange(-Q, Q + 1)
    M = np.diag(-alpha * q.astype(float) ** 2)
    idx = np.arange(2 * Q)
    M[idx, idx + 1] = eta
    M[idx + 1, idx] = eta
    return M


def parameters(A: float, k: float, gamma: float, sigma: float, phi: float | None = None):
    r"""``(alpha, eta, half_spread_floor)`` for either objective.

    ``gamma > 0`` selects exponential utility; ``gamma = 0`` selects the
    risk-neutral problem with running penalty ``phi``.
    """
    if gamma > 0:
        alpha = k * gamma * sigma**2 / 2.0
        eta = A * (1.0 + gamma / k) ** (-(1.0 + k / gamma))
        floor = np.log1p(gamma / k) / gamma
    else:
        if phi is None:
            raise ValueError("the risk-neutral problem needs a penalty phi")
        alpha = k * phi
        eta = A / np.e
        floor = 1.0 / k
    return alpha, eta, floor


def stationary_quotes(A, k, gamma, sigma, Q: int, phi: float | None = None):
    r"""Long-horizon optimal half-spreads, ``(delta_bid, delta_ask, v)``.

    ``delta_bid[i]`` is the distance below the mid to quote when the inventory is
    ``i - Q``; it is ``inf`` at ``q = Q``, where the maker is full and stops
    bidding.  The eigenvector is positive by Perron-Frobenius, so the logarithms
    are well defined.
    """
    alpha, eta, floor = parameters(A, k, gamma, sigma, phi)
    M = system_matrix(Q, alpha, eta)
    values, vectors = np.linalg.eig(M)
    principal = int(np.argmax(values.real))
    v = np.abs(vectors[:, principal].real)
    v = v / v.max()
    n = 2 * Q + 1
    delta_bid = np.full(n, np.inf)
    delta_ask = np.full(n, np.inf)
    delta_bid[:-1] = floor + np.log(v[:-1] / v[1:]) / k
    delta_ask[1:] = floor + np.log(v[1:] / v[:-1]) / k
    return delta_bid, delta_ask, v


def finite_horizon_quotes(A, k, gamma, sigma, Q: int, tau, phi: float | None = None):
    r"""Quotes ``tau`` before the end, from :math:`v(\tau)=e^{M\tau}\mathbf{1}`.

    The terminal condition :math:`\theta_q(T)=0` means the maker is not
    penalised for the inventory it is left holding, so the quotes widen towards
    the stationary ones as ``tau`` grows and collapse to the myopic
    :math:`1/k` (or :math:`\gamma^{-1}\ln(1+\gamma/k)`) as ``tau`` goes to zero.
    """
    alpha, eta, floor = parameters(A, k, gamma, sigma, phi)
    M = system_matrix(Q, alpha, eta)
    v = expm(M * float(tau)) @ np.ones(2 * Q + 1)
    n = 2 * Q + 1
    delta_bid = np.full(n, np.inf)
    delta_ask = np.full(n, np.inf)
    delta_bid[:-1] = floor + np.log(v[:-1] / v[1:]) / k
    delta_ask[1:] = floor + np.log(v[1:] / v[:-1]) / k
    return delta_bid, delta_ask, v


def fill_intensity_curve(distances, counts, seconds: float):
    r"""Fit :math:`\Lambda(\delta)=Ae^{-k\delta}` to a measured fill curve.

    ``counts[i]`` is the number of market orders that reached at least
    ``distances[i]`` from the mid over ``seconds`` of trading, so
    ``counts / seconds`` is the intensity a limit order posted there would face.
    The fit is a least-squares line in log space, which is the maximum
    likelihood fit for multiplicative errors and is what the exponential model
    deserves.
    """
    d = np.asarray(distances, dtype=float)
    lam = np.asarray(counts, dtype=float) / seconds
    ok = lam > 0
    if ok.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(d[ok], np.log(lam[ok]), 1)
    fitted = intercept + slope * d[ok]
    resid = np.log(lam[ok]) - fitted
    ss = 1.0 - resid.var() / np.log(lam[ok]).var()
    return float(np.exp(intercept)), float(-slope), float(ss)
