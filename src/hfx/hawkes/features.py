r"""Excitation features: the state of every exponential kernel at every event.

For a kernel bank :math:`\{e^{-\beta_r t}\}` the intensity of a Hawkes process
with exponential kernels is *linear* in its parameters,

.. math::
    \lambda_i(t) = \mu_i + \sum_{j,r} \alpha_{ijr} S^r_j(t),
    \qquad S^r_j(t) = \sum_{t^j_m < t} e^{-\beta_r (t - t^j_m)} ,

so once the :math:`S^r_j` are tabulated at the event times the log-likelihood
:math:`\sum_k \log\lambda - \int\lambda` is a **concave** function of
:math:`(\mu, \alpha)`: the maximum likelihood estimate is a convex program with
no local optima to worry about, and each evaluation is one matrix product.

The recursion is exact -- no discretisation of time anywhere -- because
:math:`S^r_j` decays deterministically between events:
:math:`S^r_j(t_k^-) = e^{-\beta_r \Delta_k}\,S^r_j(t_{k-1}^-) + e^{-\beta_r\Delta_k}
\mathbb{1}\{\text{event } k-1 \text{ is of type } j\}`.
"""

from __future__ import annotations

import numpy as np


def excitation_features(times, marks, betas, d: int | None = None):
    r"""Tabulate :math:`S^r_j(t_k^-)` and the counts :math:`N_j(t_k^-)`.

    Parameters
    ----------
    times : (n,) event times, sorted ascending.
    marks : (n,) event types in ``range(d)``.
    betas : (R,) decay rates of the kernel bank.

    Returns
    -------
    state : (n, d, R) the kernel states strictly before each event.
    counts : (n, d) the number of past events of each type, strictly before.
    """
    times = np.asarray(times, dtype=float)
    marks = np.asarray(marks, dtype=np.int64)
    betas = np.asarray(betas, dtype=float)
    if times.ndim != 1 or marks.shape != times.shape:
        raise ValueError("times and marks must be one-dimensional and the same length")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted")
    d = int(marks.max()) + 1 if d is None else d
    n, R = times.size, betas.size

    state = np.zeros((n, d, R))
    counts = np.zeros((n, d), dtype=np.int64)
    acc = np.zeros((d, R))
    seen = np.zeros(d, dtype=np.int64)
    prev = times[0] if n else 0.0
    for k in range(n):
        dt = times[k] - prev
        if dt > 0:
            acc *= np.exp(-betas * dt)
        state[k] = acc
        counts[k] = seen
        acc[marks[k]] += 1.0
        seen[marks[k]] += 1
        prev = times[k]
    return state, counts


def tail_integral(times, marks, betas, T: float, d: int):
    r"""The compensator's kernel part, :math:`\sum_m (1-e^{-\beta_r(T-t^j_m)})`.

    Multiplying by :math:`\alpha_{ijr}/\beta_r` and summing gives the integral
    :math:`\int_0^T \lambda_i - \mu_i T`.
    """
    times = np.asarray(times, dtype=float)
    marks = np.asarray(marks, dtype=np.int64)
    betas = np.asarray(betas, dtype=float)
    out = np.zeros((d, betas.size))
    for j in range(d):
        tj = times[marks == j]
        if tj.size:
            out[j] = (1.0 - np.exp(-np.outer(T - tj, betas))).sum(axis=0)
    return out
