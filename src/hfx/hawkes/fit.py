r"""Maximum likelihood for Hawkes processes with an exponential kernel bank.

The model fitted here is

.. math::
    \lambda_i(t) = \mu_i + \sum_j \int_0^t \varphi_{ij}(t-s)\,dN_j(s),
    \qquad \varphi_{ij}(t) = \sum_{r=1}^{R} \alpha_{ijr} e^{-\beta_r t},

with the decay rates :math:`\beta_r` **fixed** on a geometric grid and only the
non-negative amplitudes estimated.  Two reasons, and they are the design of the
whole chapter.

1. With :math:`\beta` fixed the log-likelihood is concave in
   :math:`(\mu,\alpha)`, so the estimate is a global maximum reached by a convex
   solver -- not a point some optimiser happened to stop at.
2. A single exponential is a bad model of order flow and the goodness-of-fit
   test says so loudly.  A bank of exponentials spanning six decades can
   represent a slowly decaying, effectively power-law kernel, which is what the
   data asks for.

The branching matrix is :math:`\int\varphi_{ij} = \sum_r \alpha_{ijr}/\beta_r`,
and its spectral radius is the endogeneity ratio.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .features import excitation_features, tail_integral


def log_grid(low: float, high: float, n: int) -> np.ndarray:
    """Geometric grid of decay rates, in units of inverse time."""
    return np.geomspace(low, high, n)


class HawkesExpFit:
    """Result of a fit: parameters, diagnostics, and the fitted kernel."""

    def __init__(self, mu, alpha, betas, loglik, n_events, T, converged, message):
        self.mu = np.asarray(mu, float)
        self.alpha = np.asarray(alpha, float)        # (d, d, R)
        self.betas = np.asarray(betas, float)
        self.loglik = float(loglik)
        self.n_events = int(n_events)
        self.T = float(T)
        self.converged = bool(converged)
        self.message = message

    @property
    def branching_matrix(self) -> np.ndarray:
        r""":math:`\int \varphi_{ij}`, the mean number of type-``i`` children."""
        return (self.alpha / self.betas).sum(axis=2)

    @property
    def branching_ratio(self) -> float:
        """Spectral radius of the branching matrix: the endogeneity ratio."""
        return float(np.max(np.abs(np.linalg.eigvals(self.branching_matrix))))

    def kernel(self, t) -> np.ndarray:
        r""":math:`\varphi_{ij}(t)` evaluated on a grid, shape ``(d, d, len(t))``."""
        t = np.atleast_1d(np.asarray(t, float))
        return np.einsum("ijr,rt->ijt", self.alpha, np.exp(-np.outer(self.betas, t)))

    def mean_intensity(self) -> np.ndarray:
        d = self.mu.size
        return np.linalg.solve(np.eye(d) - self.branching_matrix, self.mu)

    def __repr__(self) -> str:
        return (
            f"HawkesExpFit(n={self.n_events}, branching_ratio="
            f"{self.branching_ratio:.3f}, loglik={self.loglik:.1f})"
        )


def _pack(d: int, R: int):
    """Index helpers for the flat parameter vector ``[mu (d), alpha (d*d*R)]``."""
    return d, d + d * d * R


def fit_exp_bank(
    times,
    marks=None,
    betas=None,
    T: float | None = None,
    d: int | None = None,
    symmetric: bool = False,
    tol: float = 1e-9,
):
    r"""Fit :math:`(\mu, \alpha)` by maximum likelihood on a fixed decay grid.

    ``symmetric`` ties the parameters as
    :math:`\mu_1=\mu_2`, :math:`\alpha_{11}=\alpha_{22}` (self-excitation) and
    :math:`\alpha_{12}=\alpha_{21}` (cross-excitation), which is the natural
    model for a buy/sell order flow with no directional preference and the one
    whose signature plot has a closed form.
    """
    times = np.asarray(times, dtype=float)
    marks = np.zeros(times.size, dtype=np.int64) if marks is None else np.asarray(marks, np.int64)
    d = (int(marks.max()) + 1 if marks.size else 1) if d is None else d
    if symmetric and d != 2:
        raise ValueError("symmetric parameterisation is only defined for d = 2")
    betas = log_grid(1e-2, 1e3, 8) if betas is None else np.asarray(betas, float)
    T = float(times[-1] - times[0]) if T is None else float(T)
    t0 = times[0] if times.size else 0.0
    times = times - t0

    state, _counts = excitation_features(times, marks, betas, d=d)
    tail = tail_integral(times, marks, betas, T, d)          # (d, R)
    R = betas.size
    rows = [np.flatnonzero(marks == i) for i in range(d)]
    # X[i] has one row per event of type i and one column per (j, r) pair.
    X = [state[rows[i]].reshape(len(rows[i]), d * R) for i in range(d)]
    tail_flat = tail.reshape(d * R)
    inv_beta = np.tile(1.0 / betas, d)

    def unpack(theta):
        if symmetric:
            mu = np.repeat(theta[0], 2)
            a_self = theta[1 : 1 + R]
            a_cross = theta[1 + R : 1 + 2 * R]
            alpha = np.empty((2, 2, R))
            alpha[0, 0] = alpha[1, 1] = a_self
            alpha[0, 1] = alpha[1, 0] = a_cross
            return mu, alpha
        n_mu, _ = _pack(d, R)
        mu = theta[:n_mu]
        alpha = theta[n_mu:].reshape(d, d, R)
        return mu, alpha

    def negll(theta):
        mu, alpha = unpack(theta)
        total = 0.0
        grad_mu = np.zeros(d)
        grad_alpha = np.zeros((d, d, R))
        for i in range(d):
            a_flat = alpha[i].reshape(d * R)
            lam = mu[i] + X[i] @ a_flat
            if np.any(lam <= 0):
                return np.inf, np.zeros_like(theta)
            total += np.log(lam).sum() - mu[i] * T - (a_flat * inv_beta * tail_flat).sum()
            inv = 1.0 / lam
            grad_mu[i] = inv.sum() - T
            grad_alpha[i] = (X[i].T @ inv - inv_beta * tail_flat).reshape(d, R)
        if symmetric:
            g = np.empty(1 + 2 * R)
            g[0] = grad_mu.sum()
            g[1 : 1 + R] = grad_alpha[0, 0] + grad_alpha[1, 1]
            g[1 + R :] = grad_alpha[0, 1] + grad_alpha[1, 0]
            return -total, -g
        return -total, -np.concatenate([grad_mu, grad_alpha.ravel()])

    if symmetric:
        theta0 = np.concatenate([[len(times) / T / 2], np.full(2 * R, 1e-3)])
    else:
        theta0 = np.concatenate(
            [[len(r) / T for r in rows], np.full(d * d * R, 1e-3)]
        )
    bounds = [(1e-12, None)] * theta0.size
    res = minimize(negll, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 2000, "ftol": tol, "gtol": 1e-10})
    mu, alpha = unpack(res.x)
    return HawkesExpFit(mu, alpha, betas, -res.fun, times.size, T, res.success, res.message)
