r"""From a fitted order-flow model to the volatility of the price it implies.

Take the signed trade-count process :math:`P_t = N^+_t - N^-_t` with a symmetric
bivariate Hawkes flow: self-excitation :math:`\varphi_s` (a buy makes the next
buy more likely -- order splitting) and cross-excitation :math:`\varphi_c` (a buy
makes a sell more likely -- the reaction of the other side).  Write
:math:`\hat s(\omega),\hat c(\omega)` for their Fourier transforms.

Because :math:`u=(1,-1)` is an eigenvector of the symmetric branching structure,
the Bartlett spectrum of :math:`P` collapses to a scalar,

.. math::
    S_P(\omega) = \frac{2\Lambda}{2\pi\,|1-(\hat s(\omega)-\hat c(\omega))|^2},

and the variance of the increment over a scale :math:`\tau` follows from
:math:`\mathrm{Var}(P_{t+\tau}-P_t)=\int S_P(\omega)\,\frac{4\sin^2(\omega\tau/2)}{\omega^2}d\omega`.
Two limits are worth naming because they *are* the microstructure noise:

.. math::
    V(0^+) = 2\Lambda, \qquad V(\infty) = \frac{2\Lambda}{(1-(s-c))^2},

where :math:`s=\int\varphi_s` and :math:`c=\int\varphi_c`.  The ratio
:math:`V(\infty)/V(0^+) = (1-(s-c))^{-2}` is the whole signature plot in one
number, and it is predicted from the *flow* alone -- nothing about prices enters.
When self-excitation dominates (:math:`s>c`, persistent signs) the variance per
unit time **rises** with the scale; when the flow alternates (:math:`c>s`) it
falls, which is the textbook decreasing signature plot.

For a single exponential shared by both kernels the integral is explicit:

.. math::
    V(\tau) = 2\Lambda\left[1 + \frac{\delta(2-\delta)}{(1-\delta)^2}
      \left(1 - \frac{1-e^{-\beta_\delta\tau}}{\beta_\delta\tau}\right)\right],
    \quad \delta = s-c,\ \beta_\delta=\beta(1-\delta).
"""

from __future__ import annotations

import numpy as np
import warnings

from scipy.integrate import IntegrationWarning, quad


def signature_plot_closed_form(lam: float, delta: float, beta: float, taus):
    r"""``V(tau)`` for a symmetric pair of single-exponential kernels.

    ``lam`` is the mean intensity of each side, ``delta`` the net excitation
    :math:`s-c` and ``beta`` the shared decay rate.
    """
    taus = np.atleast_1d(np.asarray(taus, float))
    if not -1.0 < delta < 1.0:
        raise ValueError("delta = s - c must lie strictly between -1 and 1")
    beta_d = beta * (1.0 - delta)
    x = beta_d * taus
    shape = np.where(x > 1e-8, 1.0 - (1.0 - np.exp(-x)) / np.where(x > 0, x, 1.0), 0.0)
    return 2.0 * lam * (1.0 + delta * (2.0 - delta) / (1.0 - delta) ** 2 * shape)


def _net_transfer(alpha_self, alpha_cross, betas, omega):
    r""":math:`\hat s(\omega)-\hat c(\omega)` for an exponential bank."""
    num = np.asarray(alpha_self, float) - np.asarray(alpha_cross, float)
    return np.sum(num / (np.asarray(betas, float) + 1j * omega))


def signature_plot_spectral(alpha_self, alpha_cross, betas, lam: float, taus,
                            limit: int = 200):
    r"""``V(tau)`` from the spectrum, for any exponential bank.

    Integrating :math:`(2-2\cos\omega\tau)/\omega^2` against the transfer
    function directly is a bad idea: the weight oscillates faster and faster as
    :math:`\tau` grows while the tail decays only as :math:`\omega^{-2}`, and a
    general-purpose quadrature rule runs out of subdivisions.  Subtracting the
    zero-frequency value first fixes it,

    .. math::
        \frac{H(\omega)}{\omega^2} = \frac{H(0)}{\omega^2} + k(\omega),
        \qquad k(\omega)=\frac{H(\omega)-H(0)}{\omega^2},
        \qquad H = |1-(\hat s-\hat c)|^{-2},

    because the first piece integrates exactly to :math:`V(\infty)` and
    :math:`k` is smooth at the origin and absolutely integrable, so the
    remaining oscillatory integral is one QUADPACK ``weight="cos"`` call.
    """
    taus = np.atleast_1d(np.asarray(taus, float))
    betas = np.asarray(betas, float)

    def H(w):
        return 1.0 / abs(1.0 - _net_transfer(alpha_self, alpha_cross, betas, w)) ** 2

    h0 = H(0.0)
    scale = float(np.max(betas))

    def k(w):
        if w < 1e-6 * scale:
            eps = 1e-4 * scale
            return (H(eps) - h0) / eps**2
        return (H(w) - h0) / (w * w)

    with warnings.catch_warnings():
        # The oscillatory rule reports difficulty on the far cycles, where the
        # integrand is already below the tolerance; the answer is checked
        # against the closed form to 1e-6 in the tests.
        warnings.simplefilter("ignore", IntegrationWarning)
        flat, _ = quad(k, 0.0, np.inf, limit=limit)
    v_inf = 2.0 * lam * h0
    out = np.empty(taus.size)
    for i, tau in enumerate(taus):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            osc, _ = quad(k, 0.0, np.inf, weight="cos", wvar=float(tau), limit=limit)
        out[i] = v_inf + (2.0 * lam / (2.0 * np.pi * tau)) * 4.0 * (flat - osc)
    return out


def empirical_signature_plot(times, marks, taus, T: float | None = None):
    r"""``V(tau)`` measured from a realisation of :math:`P_t=N^+_t-N^-_t`.

    ``marks`` is 0 for :math:`N^+` and 1 for :math:`N^-`.
    """
    times = np.asarray(times, float)
    marks = np.asarray(marks, np.int64)
    signs = np.where(marks == 0, 1.0, -1.0)
    path = np.cumsum(signs)
    t0 = times[0]
    T = (times[-1] - t0) if T is None else T
    out = np.empty(len(taus))
    for i, tau in enumerate(taus):
        edges = np.arange(0.0, T + tau, tau) + t0
        if edges.size < 3:
            out[i] = np.nan
            continue
        idx = np.searchsorted(times, edges, side="right") - 1
        values = np.where(idx >= 0, path[np.clip(idx, 0, path.size - 1)], 0.0)
        out[i] = np.var(np.diff(values), ddof=1) / tau
    return out


def symmetric_parts(fit):
    r"""``(lam, s, c, delta)`` from a symmetric bivariate fit."""
    norms = fit.branching_matrix
    s, c = float(norms[0, 0]), float(norms[0, 1])
    lam = float(fit.mean_intensity()[0])
    return lam, s, c, s - c
