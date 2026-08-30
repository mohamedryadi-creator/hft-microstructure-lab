"""Volatility estimators against a price whose integrated variance we chose.

The claims being checked are the ones the chapter makes: the exact bias of
realized variance under additive noise, the unbiasedness of the three
noise-robust estimators, their *convergence rates* -- measured, not quoted --
and the two identities of the uncertainty-zones model.
"""

import numpy as np
import pytest

from hfx.vol import estimators as ve
from hfx.vol import uncertainty_zones as uz

IV = 4e-4          # a 2% daily volatility
OMEGA = 1e-4       # one basis point of microstructure noise


def noisy_path(rng, n, iv=IV, omega=OMEGA):
    x = np.concatenate(([0.0], np.cumsum(rng.normal(0, np.sqrt(iv / n), n))))
    return x, x + rng.normal(0, omega, n + 1)


def test_realized_variance_bias_is_exactly_two_n_omega_squared():
    rng = np.random.default_rng(0)
    n, reps = 20_000, 300
    vals = [ve.realized_variance(noisy_path(rng, n)[1]) for _ in range(reps)]
    observed = np.mean(vals) - IV
    predicted = 2 * n * OMEGA**2
    assert observed == pytest.approx(predicted, rel=0.02)


def test_noise_variance_needs_the_signal_removed():
    rng = np.random.default_rng(1)
    n, reps = 20_000, 200
    naive, corrected = [], []
    for _ in range(reps):
        _x, y = noisy_path(rng, n)
        naive.append(ve.noise_variance(y))
        corrected.append(ve.noise_variance(y, iv=ve.pre_averaged(y)))
    # RV/(2n) carries a bias of exactly IV/(2n), which here is as large as the
    # noise variance itself.
    assert np.mean(naive) == pytest.approx(OMEGA**2 + IV / (2 * n), rel=0.03)
    assert np.mean(corrected) == pytest.approx(OMEGA**2, rel=0.03)


@pytest.mark.parametrize("estimator", ["two_scale", "pre_averaged", "realized_kernel"])
def test_noise_robust_estimators_are_unbiased(estimator):
    rng = np.random.default_rng(2)
    fn = getattr(ve, estimator)
    n, reps = 20_000, 250
    vals = np.array([fn(noisy_path(rng, n)[1]) for _ in range(reps)])
    se = vals.std(ddof=1) / np.sqrt(reps)
    assert abs(vals.mean() - IV) < 4 * se


def test_two_scale_subgrids_must_span_the_whole_day():
    """The anchoring that removes a -(K-1)/n bias, checked with no noise at all."""
    rng = np.random.default_rng(3)
    n, K, reps = 20_000, 2_000, 200
    vals = []
    for _ in range(reps):
        x, _y = noisy_path(rng, n, omega=0.0)
        vals.append(ve.two_scale(x, K=K))
    vals = np.array(vals)
    se = vals.std(ddof=1) / np.sqrt(reps)
    assert abs(vals.mean() - IV) < 4 * se
    # Unanchored subgrids would sit about (K - 1) / n = 10% low.
    assert vals.mean() > 0.95 * IV


def test_convergence_rates_are_what_the_theory_says():
    """Regress log RMSE on log n and read the slope."""
    rng = np.random.default_rng(4)
    ns = [4_000, 16_000, 64_000, 256_000]
    reps = 150
    rmse = {"two_scale": [], "pre_averaged": [], "realized_kernel": []}
    for n in ns:
        acc = {k: [] for k in rmse}
        for _ in range(reps):
            _x, y = noisy_path(rng, n)
            for name in rmse:
                acc[name].append(getattr(ve, name)(y))
        for name in rmse:
            a = np.array(acc[name])
            rmse[name].append(np.sqrt(np.mean((a - IV) ** 2)))
    slopes = {
        name: np.polyfit(np.log(ns), np.log(vals), 1)[0] for name, vals in rmse.items()
    }
    assert slopes["two_scale"] == pytest.approx(-1 / 6, abs=0.06)
    assert slopes["pre_averaged"] == pytest.approx(-1 / 4, abs=0.06)
    assert slopes["realized_kernel"] == pytest.approx(-1 / 5, abs=0.06)
    # Pre-averaging converges strictly faster than the two-scale estimator, and
    # the positive-semi-definite kernel sits between them.
    assert slopes["pre_averaged"] < slopes["realized_kernel"] < slopes["two_scale"]


def test_realized_kernel_autocovariances_by_fft_match_the_direct_sum():
    rng = np.random.default_rng(5)
    dy = rng.normal(size=5_000)
    H = 120
    direct = np.array(
        [float(dy @ dy) if h == 0 else float(dy[h:] @ dy[:-h]) for h in range(H + 1)]
    )
    assert np.allclose(direct, ve.return_autocovariances(dy, H), rtol=1e-12, atol=1e-10)


def test_realized_kernel_is_never_negative():
    rng = np.random.default_rng(6)
    for _ in range(20):
        _x, y = noisy_path(rng, 5_000, iv=1e-8)   # noise dominates completely
        assert ve.realized_kernel(y) >= 0.0


def test_optimal_sampling_balances_bias_against_variance():
    n_star = ve.optimal_sampling(IV, OMEGA**2)
    assert n_star == pytest.approx((IV**2 / (4 * OMEGA**4)) ** (1 / 3))
    # About 740 returns over the day for these numbers -- one every thirty
    # seconds.  The usual "sample every few minutes" advice, arrived at rather
    # than asserted, and it moves with the noise-to-signal ratio.
    assert 500 < n_star < 1_000
    assert ve.optimal_sampling(IV, 4 * OMEGA**2) < n_star   # noisier -> sparser


# --- uncertainty zones ---------------------------------------------------


def uz_path(rng, n=1_000_000, sigma=0.30, tick=0.01, eta=0.2):
    x = 50.0 + np.concatenate(([0.0], np.cumsum(rng.normal(0, sigma * np.sqrt(1.0 / n), n))))
    levels, _changed = uz.simulate(x, tick, eta)
    return x, levels


def test_eta_is_recovered_from_continuations_and_alternations():
    rng = np.random.default_rng(7)
    for eta_true in (0.1, 0.2, 0.35):
        _x, levels = uz_path(rng, eta=eta_true)
        eta_hat, n_cont, n_alt = uz.estimate_eta(levels)
        # Standard error from the two counts; the estimator is a ratio.
        se = eta_hat * np.sqrt(1 / max(n_cont, 1) + 1 / max(n_alt, 1))
        assert abs(eta_hat - eta_true) < 4 * se + 0.02


def test_the_efficient_price_is_known_exactly_at_every_price_change():
    rng = np.random.default_rng(8)
    tick, eta = 0.01, 0.2
    x, levels = uz_path(rng, eta=eta, tick=tick)
    idx, x_hat = uz.efficient_price_at_changes(levels, tick, eta)
    # The model says X sits exactly on the barrier the moment the price moves.
    # On a discrete grid it has overshot, by a step of the simulation -- here
    # 0.03 ticks -- and no more.
    step = 0.30 * np.sqrt(1.0 / 1_000_000)
    error = np.abs(x[idx] - x_hat)
    assert error.mean() < 1.5 * step
    assert np.max(error) < 12 * step


def test_uncertainty_zone_variance_beats_the_naive_one_on_a_tick_grid():
    rng = np.random.default_rng(9)
    sigma, tick, eta = 0.30, 0.01, 0.15
    n = 1_000_000
    x = 50.0 + np.concatenate(([0.0], np.cumsum(rng.normal(0, sigma * np.sqrt(1.0 / n), n))))
    levels, _ = uz.simulate(x, tick, eta)
    iv_true = sigma**2
    iv_uz = uz.integrated_variance(levels, tick)
    assert iv_uz == pytest.approx(iv_true, rel=0.15)
    # The naive realized variance of the grid price overstates IV by 1 / (2 eta),
    # which optional stopping predicts exactly.
    naive = ve.realized_variance(levels * tick)
    assert naive / iv_true == pytest.approx(uz.variance_inflation(eta), rel=0.15)
    assert uz.variance_inflation(0.5) == pytest.approx(1.0)


def test_implicit_spread_is_below_the_tick_and_vanishes_at_half():
    assert uz.implicit_spread(0.01, 0.0) == pytest.approx(0.01)
    assert uz.implicit_spread(0.01, 0.2) == pytest.approx(0.006)
    assert uz.implicit_spread(0.01, 0.5) == pytest.approx(0.0)
