"""Chapter 07: the first-passage solve, the models, and the labelling.

The labelling test is the important one.  Every number in the chapter is an
out-of-sample probability, and the single way to make those meaningless is to
let the answer reach the features -- so the label construction is checked on a
book small enough to verify by hand.
"""

import numpy as np
import pytest

from hfx.predict import features as F
from hfx.predict import firstpassage as fp
from hfx.predict import models as M


def constant_rates(grid=12, up=2.0, cancel=1.5, market=0.5):
    lam = np.zeros((3, grid))
    lam[0, :], lam[1, :], lam[2, :] = up, cancel, market
    return lam


def test_jump_distribution_is_a_distribution_over_whole_buckets():
    centres = np.array([0.05, 0.4, 1.2, 3.4, 9.0])
    hist = np.array([[10.0, 20.0, 30.0, 5.0, 1.0]])
    probs, steps = fp.jump_distribution(hist, centres)
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(steps >= 1)
    # Sizes below half a bucket would leave the queue unchanged and stall the
    # chain, so they are rounded up rather than dropped.
    assert steps.min() == 1
    assert probs[steps == 1].sum() > 0.5


def test_a_symmetric_book_is_a_coin_flip():
    lam = constant_rates()
    probs, steps = np.array([1.0]), np.array([1])
    u, _fallback = fp.absorption_probability(lam, probs, steps, grid=12)
    diagonal = np.array([u[k, k] for k in range(1, 11)])
    assert np.allclose(diagonal, 0.5, atol=1e-9)
    assert u[0, 5] == 0.0 and u[5, 0] == 1.0        # absorbed states


def test_the_surface_is_monotone_in_both_queues():
    lam = constant_rates()
    u, _ = fp.absorption_probability(lam, np.array([1.0]), np.array([1]), grid=16)
    # A bigger bid queue makes an up-move more likely; a bigger ask queue less.
    assert np.all(np.diff(u[1:15, 5]) > 0)
    assert np.all(np.diff(u[5, 1:15]) < 0)


def test_the_solve_matches_a_monte_carlo_of_the_same_chain():
    """Two routes to the same number: a linear system and a plain simulation."""
    lam = constant_rates(grid=16, up=2.0, cancel=1.0, market=1.0)
    probs, steps = np.array([1.0]), np.array([1])
    u, _ = fp.absorption_probability(lam, probs, steps, grid=16)
    rng = np.random.default_rng(0)
    for start in [(3, 3), (2, 6), (6, 2)]:
        n = 4000
        mc = fp.simulate_absorption(lam, probs, steps, start, n, rng=rng, grid=16)
        se = np.sqrt(0.25 / n)
        assert abs(u[start] - mc) < 4 * se


def test_labels_look_strictly_forward():
    """A hand-built book: three snapshots, two later price changes."""
    second = 1_000_000_000
    states = {"ts": np.array([1, 2, 3]) * second}
    quotes = {
        "ts": np.array([0, 5, 9]) * second,
        "bid": np.array([100, 101, 100]),
        "ask": np.array([102, 103, 102]),
    }
    keep, y = F.label_next_move(states, quotes)
    # Every snapshot's next change is the one at t = 5s, which moved the mid up.
    assert keep.tolist() == [True, True, True]
    assert y.tolist() == [1, 1, 1]

    # A snapshot after the last change has no future to be labelled with.
    states = {"ts": np.array([1, 12]) * second}
    keep, _y = F.label_next_move(states, quotes)
    assert keep.tolist() == [True, False]


def test_a_symmetric_spread_widening_carries_no_direction():
    second = 1_000_000_000
    states = {"ts": np.array([1]) * second}
    quotes = {
        "ts": np.array([0, 5]) * second,
        "bid": np.array([100, 99]),
        "ask": np.array([102, 103]),      # mid unchanged, spread doubled
    }
    keep, _y = F.label_next_move(states, quotes)
    assert keep.tolist() == [False]


def test_feature_matrix_is_built_from_the_columns_it_claims():
    second = 1_000_000_000
    states = {
        "ts": np.array([1, 2]) * second,
        "bid": np.array([100_000, 100_000]),
        "ask": np.array([100_200, 100_200]),
        "q_bid": np.array([300, 100]),
        "q_ask": np.array([100, 300]),
        "q_bid2": np.array([200, 200]),
        "q_ask2": np.array([200, 200]),
        "flow_fast": np.array([1.5, -1.5]),
        "flow_slow": np.array([0.5, -0.5]),
        "ns_since_change": np.array([second, 2 * second]),
    }
    quotes = {"ts": np.array([0, 5]) * second, "bid": np.array([100_000, 100_100]),
              "ask": np.array([100_200, 100_300])}
    X, y, extra = F.build(states, quotes, aes=100.0)
    assert X.shape == (2, len(F.FEATURES))
    imbalance = X[:, F.FEATURES.index("imbalance")]
    assert imbalance[0] == pytest.approx(0.5)     # 300 against 100
    assert imbalance[1] == pytest.approx(-0.5)
    assert X[0, F.FEATURES.index("spread_ticks")] == pytest.approx(2.0)
    assert extra["bucket_bid"].tolist() == [3, 1]
    assert y.tolist() == [1, 1]


def test_logistic_recovers_the_coefficients_it_was_generated_with():
    rng = np.random.default_rng(0)
    n = 40_000
    X = rng.normal(size=(n, 3))
    true = np.array([1.2, -0.6, 0.0])
    p = 1.0 / (1.0 + np.exp(-(X @ true + 0.3)))
    y = (rng.random(n) < p).astype(int)
    model = M.Logistic(penalty=1e-6).fit(X, y)
    assert model.converged_
    # Features are standard normal, so standardised coefficients are the raw ones.
    assert np.allclose(model.coef_, true, atol=0.05)
    assert model.intercept_ == pytest.approx(0.3, abs=0.05)
    assert model.coefficients(["a", "b", "c"])[0][0] == "a"


def test_metrics_separate_a_perfect_predictor_from_a_useless_one():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.5).astype(int)
    perfect = np.where(y == 1, 0.99, 0.01)
    useless = np.full(y.size, 0.5)
    good, bad = M.evaluate(y, perfect), M.evaluate(y, useless)
    assert good["auc"] == pytest.approx(1.0)
    assert good["brier"] < 0.01
    assert bad["auc"] == pytest.approx(0.5, abs=0.03)
    assert bad["brier"] == pytest.approx(0.25, abs=0.01)
    # A confident and wrong predictor must score worse than an ignorant one.
    assert M.evaluate(y, 1 - perfect)["log_loss"] > bad["log_loss"]


def test_surface_averaging_lands_in_the_right_cells():
    bid = np.array([1, 1, 1, 2, 2])
    ask = np.array([3, 3, 3, 4, 4])
    p = np.array([0.2, 0.4, 0.6, 0.9, 0.9])
    surface, counts = M.surface_from_scores(bid, ask, p, grid=6, min_count=2)
    assert surface[1, 3] == pytest.approx(0.4)
    assert surface[2, 4] == pytest.approx(0.9)
    assert counts[1, 3] == 3
    assert np.isnan(surface[0, 0])         # never visited
