"""Chapter 08: the environment, and the policies that live in it.

The environment is where the chapter's claim about the value of information is
measured, so what has to be checked is not that it looks plausible but that its
mechanics are the ones described: queue position decides fills, the price moves
when a queue empties, and a maker who is filled is holding a position the market
has already moved against.
"""

import numpy as np
import pytest

from hfx.mm.queue_env import AT_TOUCH, OUT, QueueBookEnv
from hfx.mm import queue_agent as qa
from hfx.queue.reactive import SizeSampler


def unit_sampler(size=1.0):
    centres = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    counts = np.zeros(centres.size)
    counts[np.argmin(np.abs(centres - size))] = 1.0
    return SizeSampler(counts, centres)


def build(up=1.0, cancel=0.0, market=1.0, grid=12, **kwargs):
    lam = np.zeros((3, grid))
    lam[0, :], lam[1, :], lam[2, :] = up, cancel, market
    regen = np.zeros(grid)
    regen[4] = 1.0
    samplers = [unit_sampler(), unit_sampler(), unit_sampler()]
    kwargs.setdefault("batch", 256)
    kwargs.setdefault("dt", 0.02)
    kwargs.setdefault("rng", np.random.default_rng(0))
    return QueueBookEnv(lam, samplers, regen, **kwargs)


def test_staying_out_never_trades_and_never_earns():
    env = build()
    obs = env.reset()
    total = 0.0
    for _ in range(400):
        obs, reward, _info = env.step(np.full(env.batch, OUT), np.full(env.batch, OUT))
        total += float(reward.sum())
    assert total == 0.0
    assert np.all(env.inventory == 0)


def test_a_queue_that_empties_moves_the_price_and_is_redrawn():
    env = build(up=0.0, cancel=0.0, market=40.0, grid=12)
    moved = 0
    for _ in range(200):
        _obs, _reward, info = env.step(np.full(env.batch, OUT), np.full(env.batch, OUT))
        moved += int(info["moved"].sum())
    assert moved > 0
    # Regeneration always draws bucket 4, so no queue can be far from it.
    assert np.all(env.q_bid <= 12) and np.all(env.q_bid > 0)


def test_queue_position_decides_who_is_filled():
    """Two makers, same book, different places in the queue.

    The one at the front of a queue trades; the one behind four average orders
    does not, over a horizon short enough that four orders cannot be consumed.
    """
    env = build(up=0.0, cancel=0.0, market=1.0, grid=12, batch=4096)
    env.reset()
    env.q_bid[:] = 5.0
    env.q_ask[:] = 5.0
    env.inventory[:] = 0
    half = env.batch // 2
    env.ahead_bid[:half] = 0.0        # at the front
    env.ahead_bid[half:] = 4.0        # four orders back
    front_before, back_before = 0, 0
    for _ in range(3):
        env.step(np.full(env.batch, AT_TOUCH), np.full(env.batch, OUT))
    front = int((env.inventory[:half] > 0).sum())
    back = int((env.inventory[half:] > 0).sum())
    assert front > 5 * max(back, 1)


def test_inventory_never_passes_the_limit():
    env = build(up=0.5, cancel=0.5, market=4.0, inventory_limit=3)
    for _ in range(500):
        env.step(np.full(env.batch, AT_TOUCH), np.full(env.batch, AT_TOUCH))
    assert np.all(np.abs(env.inventory) <= 3)


def test_the_queue_is_reflected_at_the_cap():
    env = build(up=20.0, cancel=0.0, market=0.1, grid=12, q_max=6.0)
    for _ in range(300):
        env.step(np.full(env.batch, OUT), np.full(env.batch, OUT))
    assert np.all(env.q_bid <= 6.0 + 1e-9)
    assert np.all(env.q_ask <= 6.0 + 1e-9)


def test_being_filled_is_bad_news_which_is_what_adverse_selection_means():
    """A maker quoting into a real book keeps less than the half-spread it earns.

    Fills arrive because a queue is being consumed, and a queue being consumed
    is a queue on its way to empty -- which takes the price through the fill.
    Nothing in the environment models this; it follows from the mechanics.
    """
    env = build(up=1.0, cancel=1.0, market=3.0, grid=12, batch=4096, tick=0.01)
    metrics = qa.evaluate(env, qa.always_at_touch, n_steps=1500, warmup=200)
    per_fill = metrics["reward_per_second"] / metrics["fills_per_second"]
    assert metrics["fills_per_second"] > 0
    # Half a tick is what the fill itself pays; the maker keeps strictly less.
    assert per_fill < 0.5 * env.tick
    assert env.multi_event_share < 0.25


def test_a_rebate_is_worth_exactly_a_rebate():
    """Paid per fill, so the gain is the rebate times the fill rate -- and the
    environment's randomness does not depend on the agent, so the two runs see
    the same market and the comparison is exact."""
    rebate = 0.002
    plain = qa.evaluate(build(batch=1024, rebate=0.0), qa.always_at_touch, n_steps=600, warmup=100)
    paid = qa.evaluate(build(batch=1024, rebate=rebate), qa.always_at_touch, n_steps=600, warmup=100)
    assert paid["fills_per_second"] == pytest.approx(plain["fills_per_second"], rel=1e-12)
    gain = paid["reward_per_second"] - plain["reward_per_second"]
    assert gain == pytest.approx(rebate * plain["fills_per_second"], rel=0.05)


def test_the_blind_family_really_is_blind():
    """Disabling the imbalance rule must reproduce the inventory-only policy."""
    obs = np.column_stack([
        np.array([0.0, 2.0, -2.0, 0.0]),
        np.array([-0.9, 0.0, 0.9, 0.5]),      # imbalance varies
        np.full(4, -1.0), np.full(4, -1.0),
    ])
    blind = qa.threshold_policy(3, imbalance_min=-1.1)
    bid, ask = blind(obs, 3)
    assert bid.tolist() == [AT_TOUCH] * 4     # imbalance ignored entirely
    assert ask.tolist() == [AT_TOUCH] * 4
    sighted = qa.threshold_policy(3, imbalance_min=0.2)
    bid, _ask = sighted(obs, 3)
    assert bid.tolist() == [OUT, OUT, AT_TOUCH, AT_TOUCH]


def test_the_environment_is_reproducible():
    a = qa.evaluate(build(batch=256), qa.always_at_touch, n_steps=300, warmup=50)
    b = qa.evaluate(build(batch=256), qa.always_at_touch, n_steps=300, warmup=50)
    assert a["reward_per_second"] == b["reward_per_second"]
