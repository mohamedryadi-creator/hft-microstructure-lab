"""Hawkes: the estimator against a process whose parameters we chose.

Ground truth throughout.  A simulation with a known kernel is the only way to
find out whether the likelihood, the compensator and the spectral algebra were
transcribed correctly; the market cannot tell us that.
"""

import numpy as np
import pytest

from hfx.hawkes.fit import HawkesExpFit, fit_exp_bank, log_grid
from hfx.hawkes.gof import compensator, ks_exponential, rescaled_residuals
from hfx.hawkes.simulate import mean_intensity, simulate_exp, spectral_radius
from hfx.hawkes.spectrum import (
    empirical_signature_plot,
    signature_plot_closed_form,
    signature_plot_spectral,
)


def test_spectral_radius_and_stationary_intensity():
    alpha = np.array([[0.6, 0.3], [0.3, 0.6]])
    beta = np.full((2, 2), 2.0)
    assert spectral_radius(alpha / beta) == pytest.approx(0.45)
    # Lambda = (I - int phi)^{-1} mu, here (0.5, 0.5) / (1 - 0.45).
    assert mean_intensity([0.5, 0.5], alpha, beta) == pytest.approx([0.5 / 0.55] * 2)


def test_simulation_reproduces_its_own_stationary_intensity():
    rng = np.random.default_rng(3)
    mu, alpha, beta = np.array([0.5]), np.array([[0.8]]), np.array([[2.0]])
    T = 30_000.0
    times, marks = simulate_exp(mu, alpha, beta, T, rng)
    expected = mean_intensity(mu, alpha, beta)[0]
    # Standard error of the rate is about sqrt(Lambda / T / (1 - n)^2).
    se = np.sqrt(expected / T) / (1 - 0.4)
    assert abs(times.size / T - expected) < 4 * se


def test_explosive_parameters_are_refused():
    with pytest.raises(ValueError, match="explodes"):
        simulate_exp([1.0], [[3.0]], [[2.0]], 10.0)


def test_maximum_likelihood_recovers_the_branching_ratio():
    rng = np.random.default_rng(7)
    mu, alpha, beta = np.array([0.5]), np.array([[0.8]]), np.array([[2.0]])
    T = 30_000.0
    times, marks = simulate_exp(mu, alpha, beta, T, rng)
    fit = fit_exp_bank(times, marks, betas=np.array([2.0]), T=T, d=1)
    assert fit.converged
    assert fit.branching_ratio == pytest.approx(0.4, abs=0.03)
    assert fit.mu[0] == pytest.approx(0.5, abs=0.05)
    # The kernel integral is identified even when the grid misses the true beta.
    coarse = fit_exp_bank(times, marks, betas=log_grid(0.1, 100, 6), T=T, d=1)
    assert coarse.branching_ratio == pytest.approx(0.4, abs=0.05)


def test_symmetric_fit_recovers_self_and_cross_excitation():
    rng = np.random.default_rng(19)
    mu = np.array([1.0, 1.0])
    alpha = np.array([[1.2, 0.3], [0.3, 1.2]])
    beta = np.full((2, 2), 3.0)
    T = 20_000.0
    times, marks = simulate_exp(mu, alpha, beta, T, rng)
    fit = fit_exp_bank(times, marks, betas=np.array([3.0]), T=T, d=2, symmetric=True)
    norms = fit.branching_matrix
    assert norms[0, 0] == pytest.approx(0.4, abs=0.03)   # self, 1.2 / 3
    assert norms[0, 1] == pytest.approx(0.1, abs=0.03)   # cross, 0.3 / 3
    assert norms[0, 0] == norms[1, 1] and norms[0, 1] == norms[1, 0]
    assert fit.branching_ratio == pytest.approx(0.5, abs=0.04)


def test_time_rescaling_accepts_the_true_model_and_rejects_a_poisson_one():
    rng = np.random.default_rng(23)
    mu, alpha, beta = np.array([0.4]), np.array([[1.0]]), np.array([[2.0]])
    T = 20_000.0
    times, marks = simulate_exp(mu, alpha, beta, T, rng)

    fit = fit_exp_bank(times, marks, betas=np.array([2.0]), T=T, d=1)
    residuals = rescaled_residuals(times, marks, fit)[0]
    assert residuals.mean() == pytest.approx(1.0, abs=0.05)
    _stat, p_true = ks_exponential(residuals)
    assert p_true > 0.01

    # The same data against a homogeneous Poisson process of the same rate.
    poisson = HawkesExpFit(
        [times.size / T], np.zeros((1, 1, 1)), np.array([2.0]), 0.0, times.size, T,
        True, "constructed",
    )
    _stat, p_poisson = ks_exponential(rescaled_residuals(times, marks, poisson)[0])
    assert p_poisson < 1e-10


def test_compensator_is_increasing_and_starts_at_zero():
    rng = np.random.default_rng(5)
    times, marks = simulate_exp([0.5], [[0.8]], [[2.0]], 2_000.0, rng)
    fit = fit_exp_bank(times, marks, betas=np.array([2.0]), T=2_000.0, d=1)
    comp = compensator(times, marks, fit)
    assert comp[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(comp[:, 0]) >= -1e-9)


def test_closed_form_signature_plot_matches_the_spectral_integral():
    lam, beta, a_self, a_cross = 2.0, 3.0, 1.2, 0.3
    delta = (a_self - a_cross) / beta
    taus = np.geomspace(0.005, 1000.0, 14)
    closed = signature_plot_closed_form(lam, delta, beta, taus)
    spectral = signature_plot_spectral([a_self], [a_cross], [beta], lam, taus)
    # Two independent routes -- an explicit antiderivative and a quadrature of
    # the spectrum -- over five decades of scale.
    assert np.allclose(closed, spectral, rtol=1e-6)
    # The two limits are the microstructure noise, and they are exact.
    assert signature_plot_closed_form(lam, delta, beta, [1e-9])[0] == pytest.approx(2 * lam)
    assert signature_plot_closed_form(lam, delta, beta, [1e9])[0] == pytest.approx(
        2 * lam / (1 - delta) ** 2
    )


def test_signature_plot_matches_a_simulated_price():
    """Predicted from the flow alone, checked against a realised price path."""
    rng = np.random.default_rng(11)
    mu_s, a_self, a_cross, beta = 1.0, 1.2, 0.3, 3.0
    s, c = a_self / beta, a_cross / beta
    lam = mu_s / (1 - (s + c))
    T = 60_000.0
    times, marks = simulate_exp(
        [mu_s, mu_s], [[a_self, a_cross], [a_cross, a_self]], np.full((2, 2), beta), T, rng
    )
    taus = np.geomspace(0.05, 50.0, 8)
    closed = signature_plot_closed_form(lam, s - c, beta, taus)
    empirical = empirical_signature_plot(times, marks, taus, T=T)
    assert np.allclose(empirical, closed, rtol=0.05)
    # Self-excitation dominates, so the variance per unit time *rises* with the
    # scale: persistent order flow, not a decreasing signature plot.
    assert empirical[-1] > 1.8 * empirical[0]
