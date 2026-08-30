"""Make-take fees: the agent's problem, the principal's, and the one place they
have to agree."""

import numpy as np
import pytest

from hfx.design.maketake import exchange_gain, maker_solution, optimal_rebate
from hfx.mm.glft import stationary_quotes
from hfx.mm.mdp import action_grid, solve

A, K, PHI, Q = 1.0, 1.5, 0.02, 6


def test_a_zero_rebate_is_exactly_the_market_making_chapter():
    """The two chapters share no code path beyond the eigen-problem, so this is
    a real check that the contract has been added correctly."""
    sol = maker_solution(A, K, PHI, Q, rebate=0.0)
    bid, ask, _v = stationary_quotes(A, K, 0.0, 0.0, Q, phi=PHI)
    assert np.allclose(sol.bid[:-1], bid[:-1])
    assert np.allclose(sol.ask[1:], ask[1:])


def test_the_makers_gain_matches_the_dynamic_programme():
    """Ergodic gain from the principal eigenvalue against value iteration."""
    sol = maker_solution(A, K, PHI, Q, rebate=0.0)
    _b, _a, _V, gain = solve(A, K, PHI, Q, dt=0.005, deltas=action_grid(K, 241, -2, 4))
    assert sol.gain == pytest.approx(gain, rel=2e-3)


def test_the_round_trip_capture_absorbs_the_rebate_exactly():
    r"""An exact identity, and the cleanest statement of what a rebate buys.

    Buying at inventory :math:`q` and selling back from :math:`q+1` captures

    .. math::
        \delta^b(q)+\delta^a(q+1)
        = \tfrac2k - 2z - (\theta_{q+1}-\theta_q) - (\theta_q-\theta_{q+1})
        = \tfrac2k - 2z ,

    with the inventory terms cancelling.  So the full rebate reaches the round
    trip whatever the maker's risk aversion or inventory limit -- but *not* one
    quote at a time: individual quotes move by amounts that vary across the
    inventory range, tilting the schedule rather than translating it.
    """
    z = 0.2
    base = maker_solution(A, K, PHI, Q, rebate=0.0)
    paid = maker_solution(A, K, PHI, Q, rebate=z)
    for sol, rebate in ((base, 0.0), (paid, z)):
        round_trip = sol.bid[:-1] + sol.ask[1:]
        assert np.allclose(round_trip, 2.0 / K - 2.0 * rebate)

    shift = base.bid[:-1] - paid.bid[:-1]
    assert np.all(shift > 0)                        # every quote moves inward
    assert shift.max() - shift.min() > 0.1 * z      # but not by the same amount
    assert paid.fill_rate > base.fill_rate
    assert paid.gain > base.gain

    # With no inventory penalty there is no skew left to tilt, and the whole
    # schedule simply translates by the rebate.
    flat_base = maker_solution(A, K, 0.0, Q, rebate=0.0)
    flat_paid = maker_solution(A, K, 0.0, Q, rebate=z)
    assert np.allclose(flat_paid.bid[:-1], flat_base.bid[:-1] - z)


def test_the_optimal_rebate_is_the_taker_fee_minus_one_over_k():
    for taker_fee in (1.0 / K, 2.0 / K, 3.0 / K):
        z, sol, _grid, _obj = optimal_rebate(A, K, PHI, Q, taker_fee=taker_fee)
        assert z == pytest.approx(taker_fee - 1.0 / K, abs=0.03)
        # And it beats both corners.
        assert exchange_gain(sol, taker_fee) > exchange_gain(
            maker_solution(A, K, PHI, Q, 0.0), taker_fee
        )


def test_a_regulator_who_values_a_tight_spread_pays_the_maker_more():
    taker_fee = 2.0 / K
    z_profit, _s, _g, _o = optimal_rebate(A, K, PHI, Q, taker_fee)
    z_quality, sol_q, _g2, _o2 = optimal_rebate(A, K, PHI, Q, taker_fee, spread_weight=1.0)
    assert z_quality > z_profit
    sol_p = maker_solution(A, K, PHI, Q, z_profit)
    assert sol_q.spread < sol_p.spread


def test_the_participation_constraint_binds_when_the_maker_is_paid_too_little():
    taker_fee = 1.0 / K
    free, _s, _g, _o = optimal_rebate(A, K, PHI, Q, taker_fee)
    outside = maker_solution(A, K, PHI, Q, free).gain * 1.5
    tied, sol, _g2, _o2 = optimal_rebate(A, K, PHI, Q, taker_fee, participation=outside)
    assert tied > free
    assert sol.gain >= outside
