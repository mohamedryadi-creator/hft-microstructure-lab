r"""Goodness of fit by time rescaling.

The random time change theorem: if :math:`N_i` has intensity :math:`\lambda_i`
and compensator :math:`\Lambda_i(t)=\int_0^t\lambda_i`, then the transformed
points :math:`\Lambda_i(t^i_k)` form a **unit-rate Poisson process**.  So the
increments :math:`\tau_k = \Lambda_i(t^i_k) - \Lambda_i(t^i_{k-1})` are i.i.d.
:math:`\mathrm{Exp}(1)` exactly when the fitted intensity is the true one, and a
Kolmogorov-Smirnov test against :math:`\mathrm{Exp}(1)` is a real test of the
model rather than a plot of a fit next to the data it was fitted on.

For the exponential bank the compensator is available in closed form, since
:math:`\int_0^t e^{-\beta_r(t-s)}dN_j(s)\,` integrates to
:math:`\frac{1}{\beta_r}\big(N_j(t) - S^r_j(t)\big)`.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .features import excitation_features


def compensator(times, marks, fit, d: int | None = None) -> np.ndarray:
    r""":math:`\Lambda_i(t_k^-)` for every event, shape ``(n, d)``."""
    times = np.asarray(times, float)
    marks = np.asarray(marks, np.int64)
    d = fit.mu.size if d is None else d
    t0 = times[0] if times.size else 0.0
    rel = times - t0
    state, counts = excitation_features(rel, marks, fit.betas, d=d)
    # (alpha / beta) contracted against (counts - state)
    weights = fit.alpha / fit.betas                       # (d, d, R)
    kernel_part = np.einsum("ijr,kjr->ki", weights, counts[:, :, None] - state)
    return fit.mu[None, :] * rel[:, None] + kernel_part


def rescaled_residuals(times, marks, fit) -> list[np.ndarray]:
    r"""Per dimension, the increments :math:`\Lambda_i(t^i_k)-\Lambda_i(t^i_{k-1})`.

    Under the fitted model these are i.i.d. :math:`\mathrm{Exp}(1)`.
    """
    marks = np.asarray(marks, np.int64)
    comp = compensator(times, marks, fit)
    out = []
    for i in range(fit.mu.size):
        own = comp[marks == i, i]
        out.append(np.diff(own) if own.size > 1 else np.empty(0))
    return out


def ks_exponential(residuals) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and p-value against :math:`\\mathrm{Exp}(1)`."""
    residuals = np.asarray(residuals, float)
    if residuals.size < 2:
        return float("nan"), float("nan")
    res = stats.kstest(residuals, "expon")
    return float(res.statistic), float(res.pvalue)


def qq_points(residuals, n_points: int = 200):
    """Quantiles of the residuals against those of :math:`\\mathrm{Exp}(1)`."""
    residuals = np.sort(np.asarray(residuals, float))
    if residuals.size == 0:
        return np.empty(0), np.empty(0)
    probs = np.linspace(0.5 / n_points, 1 - 0.5 / n_points, n_points)
    return stats.expon.ppf(probs), np.quantile(residuals, probs)
