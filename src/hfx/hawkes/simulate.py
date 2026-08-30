"""Exact simulation of multivariate Hawkes processes with exponential kernels.

Ogata's modified thinning.  Between events the intensity only decays, so the
value just after the last event bounds it on the whole interval ahead; draw an
exponential candidate with that bound, then accept with probability
:math:`\\lambda(t)/M`.  The output is an exact sample of the process, which is
what makes it usable as ground truth for the estimator.
"""

from __future__ import annotations

import numpy as np


def simulate_exp(mu, alpha, beta, T: float, rng=None, max_events: int = 5_000_000):
    """Simulate on :math:`[0, T]` with :math:`\\varphi_{ij}(t)=\\alpha_{ij}e^{-\\beta_{ij}t}`.

    Parameters
    ----------
    mu : (d,) baseline intensities
    alpha, beta : (d, d) kernel amplitudes and decay rates.  ``alpha[i, j]`` is
        the jump added to :math:`\\lambda_i` by an event of type ``j``.
    T : horizon.

    Returns
    -------
    times, marks : sorted event times and their types.
    """
    mu = np.asarray(mu, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    d = mu.size
    if alpha.shape != (d, d) or beta.shape != (d, d):
        raise ValueError("alpha and beta must be (d, d) with d = len(mu)")
    if spectral_radius(alpha / beta) >= 1:
        raise ValueError(
            "the branching matrix has spectral radius >= 1: the process explodes"
        )
    rng = np.random.default_rng() if rng is None else rng

    state = np.zeros((d, d))          # state[i, j] = sum of decayed type-j events
    times: list[float] = []
    marks: list[int] = []
    t = 0.0                            # state is always the state at time t
    while True:
        bound = float((mu + (alpha * state).sum(axis=1)).sum())
        if bound <= 0:
            break
        t_next = t + rng.exponential(1.0 / bound)
        if t_next > T:
            break
        state = state * np.exp(-beta * (t_next - t))
        t = t_next
        lam = mu + (alpha * state).sum(axis=1)
        total = float(lam.sum())
        if rng.random() * bound <= total:
            k = int(np.searchsorted(np.cumsum(lam), rng.random() * total))
            times.append(t)
            marks.append(k)
            state[:, k] += 1.0
            if len(times) >= max_events:
                raise RuntimeError("simulation exceeded max_events")
    return np.asarray(times), np.asarray(marks, dtype=np.int64)


def spectral_radius(matrix) -> float:
    """Largest modulus of the eigenvalues of the branching matrix.

    For a Hawkes process the branching matrix is :math:`\\int\\varphi`, whose
    spectral radius is the expected number of children per event -- the
    endogeneity ratio.  Stationarity requires it to be below one.
    """
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(matrix, dtype=float)))))


def mean_intensity(mu, alpha, beta):
    r"""Stationary mean intensity :math:`\Lambda = (I - \int\varphi)^{-1}\mu`."""
    mu = np.asarray(mu, dtype=float)
    norms = np.asarray(alpha, dtype=float) / np.asarray(beta, dtype=float)
    return np.linalg.solve(np.eye(mu.size) - norms, mu)
