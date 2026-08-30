"""The queue-reactive model against queues we built ourselves."""

import numpy as np
import pytest

from hfx.book.replay import size_bucket_centres
from hfx.queue import reactive as qr


def test_intensities_are_counts_over_time_in_state():
    events = np.array([[10.0, 0.0], [4.0, 2.0], [6.0, 1.0]])
    time_ns = np.array([2e9, 0.5e9])          # two seconds, then half a second
    lam, valid = qr.intensities(events, time_ns, min_seconds=1.0)
    assert valid.tolist() == [True, False]
    assert lam[:, 0] == pytest.approx([5.0, 2.0, 3.0])
    assert np.all(np.isnan(lam[:, 1]))        # too little time to say anything


def test_birth_death_invariant_is_geometric_for_constant_rates():
    n = 30
    up = np.full(n, 2.0)
    down = np.full(n, 4.0)
    pi = qr.birth_death_invariant(up, down)
    ratio = 0.5
    expected = ratio ** np.arange(n)
    expected /= expected.sum()
    assert np.allclose(pi, expected)
    assert pi.sum() == pytest.approx(1.0)


def test_size_sampler_reproduces_its_histogram():
    centres = size_bucket_centres()
    counts = np.zeros(centres.size)
    counts[10] = 3.0
    counts[20] = 1.0
    sampler = qr.SizeSampler(counts, centres)
    assert sampler.mean == pytest.approx(0.75 * centres[10] + 0.25 * centres[20])
    draws = sampler.draw(np.random.default_rng(0).random(20_000))
    assert np.mean(draws) == pytest.approx(sampler.mean, rel=0.05)


def _unit_sampler():
    centres = size_bucket_centres()
    counts = np.zeros(centres.size)
    counts[np.argmin(np.abs(centres - 1.0))] = 1.0
    return qr.SizeSampler(counts, centres), float(centres[np.argmin(np.abs(centres - 1.0))])


def test_two_pure_death_queues_empty_at_the_analytic_rate():
    """Cancellations only, one unit resting on each side.

    A queue holding less than one unit is emptied by the very next event on its
    side, so the first of the two to go is Exp(2 lambda) and the mean holding
    time is 1 / (2 lambda) exactly.
    """
    nq, rate = 8, 3.0
    lam = np.zeros((3, nq))
    lam[1, :] = rate
    sampler, step = _unit_sampler()
    assert step > 0.5                       # one event clears a half-full bucket
    regen = np.zeros(nq)
    regen[0] = 1.0                          # queues restart at half a unit
    directions, holding, _ = qr.simulate(
        lam, [sampler] * 3, regen, 20_000, rng=np.random.default_rng(0)
    )
    assert holding.mean() == pytest.approx(1 / (2 * rate), rel=0.05)
    # Symmetric queues, so the price is a driftless walk on the tick grid.
    assert abs(directions.mean()) < 0.03


def test_holding_times_match_an_independent_construction():
    """Two deterministic countdowns of three events each, raced against a clock.

    The simulator's answer is compared with the same experiment written from
    scratch in three lines of numpy: two Erlang(3, lambda) waits, and the first
    one wins.
    """
    nq, rate, start = 12, 2.0, 2          # bucket 2 -> 2.5 units -> 3 events
    lam = np.zeros((3, nq))
    lam[1, :] = rate
    sampler, _step = _unit_sampler()
    regen = np.zeros(nq)
    regen[start] = 1.0
    _d, holding, _v = qr.simulate(
        lam, [sampler] * 3, regen, 40_000, rng=np.random.default_rng(2)
    )
    rng = np.random.default_rng(3)
    # Each side needs three of its own events; each side's events arrive at
    # ``rate``, so its countdown is a sum of three Exp(rate) waits.
    a = rng.gamma(3, 1 / rate, 200_000)
    b = rng.gamma(3, 1 / rate, 200_000)
    assert holding.mean() == pytest.approx(np.minimum(a, b).mean(), rel=0.05)


def test_a_faster_queue_turnover_means_a_higher_implied_volatility():
    slow = qr.implied_volatility(np.full(100, 4.0), tick=0.01, price=50.0)
    fast = qr.implied_volatility(np.full(100, 1.0), tick=0.01, price=50.0)
    assert fast == pytest.approx(2 * slow)
    # One tick per second on a $50 stock over a 6.5 hour session.
    assert qr.implied_volatility(np.ones(10), 0.01, 50.0) == pytest.approx(
        np.sqrt(0.01**2 * 23_400) / 50.0
    )
