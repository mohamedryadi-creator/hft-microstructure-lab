r"""Integrated variance from a price observed with microstructure noise.

The model throughout is additive noise,

.. math:: Y_{t_i} = X_{t_i} + \varepsilon_i,

with :math:`X` a continuous semimartingale carrying the integrated variance
:math:`IV=\int_0^T\sigma_s^2ds` and :math:`\varepsilon` i.i.d. of variance
:math:`\omega^2`, independent of :math:`X`.  Realized variance on all :math:`n`
returns is then *not* an estimator of :math:`IV`:

.. math:: \mathbb{E}\,RV_n = IV + 2n\omega^2 ,

which diverges as the sampling gets finer.  That single line is the signature
plot, the reason nobody computes realized variance tick by tick, and the reason
for everything else in this module.

Three consistent estimators are implemented, at three different rates:

===================  ============  =========================================
estimator            rate          what it does
===================  ============  =========================================
two-scale            n^{-1/6}      subtracts a scaled all-returns RV from the
                                   average of K subsampled RVs
pre-averaged         n^{-1/4}      averages the noise away inside a window of
                                   width :math:`k\sim\sqrt n` before squaring
realized kernel      n^{-1/5}      weights the return autocovariances with a
                                   Parzen kernel; positive by construction,
                                   and the guarantee costs the rate
===================  ============  =========================================

The rates are not quoted here on authority: ``tests/test_vol.py`` measures them
by Monte Carlo and checks the slope of log RMSE against log n.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def realized_variance(prices) -> float:
    """:math:`\\sum(\\Delta Y)^2` on the observations given."""
    p = np.asarray(prices, dtype=float)
    return float(np.sum(np.diff(p) ** 2))


def subsample(prices, step: int, offset: int = 0):
    return np.asarray(prices, dtype=float)[offset::step]


def signature_plot(prices, steps):
    """Realized variance as a function of the sampling step, in observations.

    Each point averages the ``step`` possible offsets, which uses every
    observation and removes the sampling-phase noise that makes raw signature
    plots ragged.
    """
    p = np.asarray(prices, dtype=float)
    out = np.empty(len(steps))
    for i, step in enumerate(steps):
        step = int(step)
        vals = [realized_variance(p[off::step]) for off in range(step)]
        out[i] = float(np.mean(vals))
    return out


def noise_variance(prices, iv: float | None = None) -> float:
    r""":math:`\hat\omega^2 = (RV_n - IV)/(2n)`.

    The usual form drops the :math:`IV` and takes :math:`RV_n/(2n)`, which is
    consistent as :math:`n\to\infty` but carries a bias of exactly
    :math:`IV/(2n)` -- and at a day of one-second returns that bias is not
    small: with a 2% daily volatility and a one-basis-point noise it nearly
    doubles the estimate.  Passing a first-pass ``iv`` removes it.
    """
    p = np.asarray(prices, dtype=float)
    n = p.size - 1
    if n <= 0:
        return float("nan")
    rv = float(np.sum(np.diff(p) ** 2))
    return max((rv - (iv or 0.0)) / (2 * n), 0.0)


def optimal_sampling(iv: float, noise_var: float, T: float = 1.0) -> float:
    r"""Number of intervals minimising the MSE of sparse realized variance.

    Balancing the squared bias :math:`(2n\omega^2)^2` against the sampling
    variance :math:`2\,IV^2/n` gives :math:`n^\star=(IV^2/(4\omega^4))^{1/3}`
    (Zhang, Mykland and Aït-Sahalia).  ``T`` is carried only so the caller can
    convert the answer into a frequency.
    """
    if noise_var <= 0:
        return float("inf")
    return float((iv**2 / (4 * noise_var**2)) ** (1 / 3))


def two_scale(prices, K: int | None = None) -> float:
    r"""Two-scale realized variance (Zhang, Mykland, Aït-Sahalia 2005).

    .. math::
        \widehat{IV} = \Big(1-\tfrac{\bar n}{n}\Big)^{-1}
        \Big[\,\overline{[Y,Y]}^{(K)} - \tfrac{\bar n}{n}[Y,Y]^{(\text{all})}\Big],
        \qquad \bar n = \tfrac{n-K+1}{K}.

    The subsampled average still carries a bias :math:`2\bar n\omega^2`; the
    all-returns estimator carries :math:`2n\omega^2` and is used to cancel it.
    The prefactor is the finite-sample correction that makes the estimator
    unbiased rather than merely consistent.
    """
    p = np.asarray(prices, dtype=float)
    n = p.size - 1
    if K is None:
        K = max(2, int(round(n ** (2 / 3))))
    K = int(min(max(K, 2), max(n // 2, 2)))
    # Each subgrid is anchored at the first and last observation.  Without that,
    # grid ``off`` covers only [off*dt, T - (K-1-off)*dt] and its realized
    # variance estimates a *fraction* of the day: the resulting bias is
    # -(K-1)/n, which at the recommended K = n^(2/3) is -3.6% on a day of
    # one-second returns -- larger than the estimator's own error, and entirely
    # avoidable.
    grids = []
    for off in range(K):
        idx = np.arange(off, n + 1, K)
        if idx[0] != 0:
            idx = np.concatenate(([0], idx))
        if idx[-1] != n:
            idx = np.concatenate((idx, [n]))
        grids.append(p[idx])
    avg = float(np.mean([realized_variance(g) for g in grids]))
    # n_bar is the average number of returns the subgrids actually use.
    nbar = float(np.mean([max(g.size - 1, 0) for g in grids]))
    return float((avg - (nbar / n) * realized_variance(p)) / (1 - nbar / n))


def _preavg_weights(k: int):
    """Triangular weights :math:`g(x)=\\min(x,1-x)` on the window, and their
    finite-sample constants.

    Using the exact discrete sums rather than their integral limits
    (:math:`\\psi_1=1`, :math:`\\psi_2=1/12`) removes an :math:`O(1/k)` bias
    that is perfectly visible at the window sizes one actually uses.
    """
    j = np.arange(1, k)
    g = np.minimum(j / k, 1.0 - j / k)
    c = np.diff(np.concatenate(([0.0], g, [0.0])))
    return g, float(np.sum(g**2)), float(np.sum(c**2))


def pre_averaged(prices, k: int | None = None, noise_var: float | None = None) -> float:
    r"""Pre-averaged realized variance (Jacod, Li, Mykland, Podolsky, Vetter).

    With :math:`\bar Y_i=\sum_{j=1}^{k-1} g(j/k)(Y_{i+j}-Y_{i+j-1})`,

    .. math::
        \mathbb{E}\bar Y_i^2 = \frac{IV}{n}\sum_j g_j^2 + \omega^2 \sum_m c_m^2 ,
        \qquad c_m = g_m-g_{m+1},

    so averaging the squares over all :math:`i` and removing the noise term
    gives :math:`IV` directly.  The window :math:`k\propto\sqrt n` is what
    produces the :math:`n^{-1/4}` rate.
    """
    p = np.asarray(prices, dtype=float)
    n = p.size - 1
    if k is None:
        k = max(2, int(round(np.sqrt(n))))
    k = int(min(max(k, 2), n))
    g, g2, c2 = _preavg_weights(k)
    dy = np.diff(p)
    # Overlapping windowed sums of the returns.  Direct convolution costs
    # O(n k) = O(n^{3/2}); the FFT one is O(n log n), which is what makes the
    # rate study in the tests affordable at all.
    bar = fftconvolve(dy, g[::-1], mode="valid")
    if noise_var is None:
        noise_var = noise_variance(p)
    return float(n * (np.mean(bar**2) - noise_var * c2) / g2)


def parzen(x):
    """Parzen kernel: positive semi-definite, hence a non-negative estimator."""
    x = np.abs(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    lo = x <= 0.5
    hi = (x > 0.5) & (x <= 1.0)
    out[lo] = 1 - 6 * x[lo] ** 2 + 6 * x[lo] ** 3
    out[hi] = 2 * (1 - x[hi]) ** 3
    return out


def realized_kernel(prices, H: int | None = None) -> float:
    r"""Realized kernel with Parzen weights (Barndorff-Nielsen et al. 2008).

    .. math::
        K(Y)=\sum_{h=-H}^{H} k\!\left(\tfrac{h}{H+1}\right)\gamma_h,
        \qquad \gamma_h=\sum_i \Delta Y_i\,\Delta Y_{i-|h|} .

    The noise contributes to :math:`\gamma_0` and :math:`\gamma_{\pm1}`, and the
    kernel weights are chosen so those contributions cancel.  The non-flat-top
    Parzen version is guaranteed non-negative and converges at :math:`n^{-1/5}`
    with :math:`H\propto n^{3/5}` -- a slower rate than pre-averaging, which is
    the price of never returning a negative variance.
    """
    p = np.asarray(prices, dtype=float)
    dy = np.diff(p)
    n = dy.size
    if H is None:
        H = max(1, int(round(n ** (3 / 5))))
    H = int(min(H, n - 1))
    weights = parzen(np.arange(0, H + 1) / (H + 1))
    gamma = return_autocovariances(dy, H)
    return float(weights[0] * gamma[0] + 2.0 * np.sum(weights[1:] * gamma[1:]))


def return_autocovariances(returns, H: int):
    r"""``gamma[h] = sum_i dY_i dY_{i-h}`` for ``h = 0..H``, by FFT.

    The direct double loop is :math:`O(nH)`, and with :math:`H\propto n^{3/5}`
    that is :math:`O(n^{8/5})` -- minutes on a day of one-second returns.  The
    autocovariance sequence is a circular correlation, so one zero-padded FFT
    pair returns every lag at once in :math:`O(n\log n)`.
    """
    dy = np.asarray(returns, dtype=float)
    n = dy.size
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(dy, size)
    return np.fft.irfft(spec * np.conjugate(spec), size)[: H + 1]


def optimal_kernel_bandwidth(noise_var: float, iv: float, n: int) -> int:
    r"""``H`` minimising the asymptotic MSE of the Parzen realized kernel.

    :math:`H^\star = c^\star\,\xi^{4/5} n^{3/5}` with
    :math:`\xi^2=\omega^2/\sqrt{IQ}` and :math:`c^\star=3.5134` for Parzen;
    integrated quarticity is taken as :math:`IV^2` on the constant-volatility
    assumption used wherever this is called.
    """
    if noise_var <= 0 or iv <= 0:
        return max(1, int(round(n ** (3 / 5))))
    xi2 = noise_var / iv          # omega^2 / sqrt(IQ) with IQ = IV^2
    return max(1, int(round(3.5134 * xi2 ** (2 / 5) * n ** (3 / 5))))
