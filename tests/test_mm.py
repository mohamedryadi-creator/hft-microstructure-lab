"""Market making: the closed form, the exact optimum of the discretised problem,
and a learner that has to find the second one on its own."""

import numpy as np
import pytest

from hfx.mm.glft import (
    fill_intensity_curve,
    finite_horizon_quotes,
    parameters,
    stationary_quotes,
)
from hfx.mm.mdp import action_grid, simulate_policy, solve
from hfx.mm.rl import q_learning

A, K, PHI, Q = 1.0, 1.5, 0.02, 6


def test_quotes_are_symmetric_and_skew_with_inventory():
    bid, ask, _v = stationary_quotes(A, K, 0.0, 0.0, Q, phi=PHI)
    # A maker long q faces the same problem as one short -q with the sides
    # swapped, so the two schedules are mirror images.
    assert np.allclose(bid[:-1], ask[1:][::-1])
    # Long inventory: quote the bid further away and the ask closer, to sell.
    assert np.all(np.diff(bid[:-1]) > 0)
    assert np.all(np.diff(ask[1:]) < 0)
    # Deeply short, the maker bids through the mid rather than stay short.
    assert bid[0] < 0


def test_risk_neutral_is_the_small_gamma_limit_of_exponential_utility():
    sigma = 1.0
    phi = 0.02
    gamma = 2 * phi / sigma**2          # matches alpha = k gamma sigma^2 / 2
    a_rn, e_rn, f_rn = parameters(A, K, 0.0, sigma, phi=phi)
    a_ca, e_ca, f_ca = parameters(A, K, gamma, sigma)
    assert a_ca == pytest.approx(a_rn)
    assert e_ca == pytest.approx(e_rn, rel=0.02)
    assert f_ca == pytest.approx(f_rn, rel=0.02)


def test_finite_horizon_quotes_interpolate_between_myopic_and_stationary():
    _a, _e, floor = parameters(A, K, 0.0, 0.0, phi=PHI)
    near, _ask, _v = finite_horizon_quotes(A, K, 0.0, 0.0, Q, tau=1e-6, phi=PHI)
    far, _ask2, _v2 = finite_horizon_quotes(A, K, 0.0, 0.0, Q, tau=500.0, phi=PHI)
    stat, _ask3, _v3 = stationary_quotes(A, K, 0.0, 0.0, Q, phi=PHI)
    # With no time left the inventory cannot hurt, so the maker quotes the
    # myopic half-spread whatever it is holding.
    assert np.allclose(near[:-1], floor, atol=1e-4)
    assert np.allclose(far[:-1], stat[:-1], atol=1e-6)


def test_dynamic_programming_converges_to_the_closed_form():
    bid_cf, ask_cf, _v = stationary_quotes(A, K, 0.0, 0.0, Q, phi=PHI)
    fine = action_grid(K, 241, -2, 4)
    step = fine[1] - fine[0]
    bid_dp, ask_dp, _V, gain = solve(A, K, PHI, Q, dt=0.01, deltas=fine)
    assert np.max(np.abs(bid_dp[:-1] - bid_cf[:-1])) < step
    assert np.max(np.abs(ask_dp[1:] - ask_cf[1:])) < step
    assert gain > 0


def test_q_learning_recovers_the_optimum_of_its_own_environment():
    """The check that makes any later use of the learner meaningful."""
    dt, beta = 1.0, 0.99
    grid = action_grid(K, 13, -2, 4)
    step = grid[1] - grid[0]
    bid_dp, ask_dp, _V, _g = solve(A, K, PHI, Q, dt, deltas=grid, beta=beta)
    bid_rl, ask_rl, _table = q_learning(
        A, K, PHI, Q, dt, deltas=grid, beta=beta, rng=np.random.default_rng(0)
    )
    finite_b = np.isfinite(bid_dp)
    finite_a = np.isfinite(ask_dp)
    assert np.max(np.abs(bid_rl[finite_b] - bid_dp[finite_b])) <= step + 1e-9
    assert np.max(np.abs(ask_rl[finite_a] - ask_dp[finite_a])) <= step + 1e-9

    rng = np.random.default_rng(1)
    reward_dp, _ = simulate_policy(bid_dp, ask_dp, A, K, PHI, Q, dt, 100_000, rng=rng)
    reward_rl, _ = simulate_policy(bid_rl, ask_rl, A, K, PHI, Q, dt, 100_000, rng=rng)
    flat = np.full(2 * Q + 1, 1.0 / K)
    reward_naive, _ = simulate_policy(flat, flat, A, K, PHI, Q, dt, 100_000, rng=rng)
    assert reward_rl > 0.95 * reward_dp
    assert reward_dp > 2 * reward_naive


def test_fill_intensity_fit_recovers_a_known_curve():
    true_A, true_k, seconds = 3.0, 40.0, 10_000.0
    distances = np.linspace(0.005, 0.08, 12)
    counts = true_A * np.exp(-true_k * distances) * seconds
    fitted_A, fitted_k, r2 = fill_intensity_curve(distances, counts, seconds)
    assert fitted_A == pytest.approx(true_A, rel=1e-6)
    assert fitted_k == pytest.approx(true_k, rel=1e-6)
    assert r2 == pytest.approx(1.0)
