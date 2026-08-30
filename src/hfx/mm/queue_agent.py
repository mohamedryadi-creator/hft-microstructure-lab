r"""Learning to quote in a book that reacts, and measuring what the book is worth.

The comparison this module exists to make is not "reinforcement learning beats a
formula".  It is an **ablation**.

Two agents learn in the same environment by the same algorithm, differing only
in what they are allowed to see:

* the **blind** agent sees its inventory and nothing else.  That is exactly the
  information the closed form of chapter 05 uses -- Avellaneda-Stoikov and
  Guéant-Lehalle-Fernandez-Tapia quote as a function of inventory and time, and
  of no feature of the book.  Learned to convergence, it is the best any such
  policy can do, which makes it a fairer benchmark than the closed form itself.
* the **sighted** agent additionally sees queue imbalance and its own position
  in each queue.

The difference between them, in reward per unit time, is the value of seeing the
book.  If it is zero, the chapter says so.
"""

from __future__ import annotations

import numpy as np

from .queue_env import AT_TOUCH, OUT

#: Buckets for the two book features.  Coarse on purpose, and the coarseness is
#: forced rather than chosen: the per-step reward is dominated by marking
#: inventory across a price move, with a standard deviation near 3e-2 against a
#: signal near 1e-5.  A state-action cell needs of order (3e-2 / signal)^2
#: samples before its value is anything but noise, so a table with thousands of
#: cells cannot be estimated from any feasible amount of experience -- and its
#: argmax will happily pick whichever cell got lucky.  Three imbalance buckets
#: and a resting flag leave 156 states, which twelve million transitions can
#: actually resolve.
IMBALANCE_EDGES = np.array([-0.25, 0.25])
POSITION_EDGES = np.array([])

N_IMBALANCE = IMBALANCE_EDGES.size + 1
N_POSITION = POSITION_EDGES.size + 2      # +1 for "not resting"
ACTIONS = ((OUT, OUT), (AT_TOUCH, OUT), (OUT, AT_TOUCH), (AT_TOUCH, AT_TOUCH))


def encode(obs, limit: int, sighted: bool = True):
    """Map observations onto a flat state index."""
    inventory = np.clip(obs[:, 0].astype(np.int64) + limit, 0, 2 * limit)
    if not sighted:
        return inventory
    imbalance = np.searchsorted(IMBALANCE_EDGES, obs[:, 1])
    pos_bid = np.where(obs[:, 2] < 0, 0, 1 + np.searchsorted(POSITION_EDGES, obs[:, 2]))
    pos_ask = np.where(obs[:, 3] < 0, 0, 1 + np.searchsorted(POSITION_EDGES, obs[:, 3]))
    idx = inventory
    idx = idx * N_IMBALANCE + imbalance
    idx = idx * N_POSITION + pos_bid
    idx = idx * N_POSITION + pos_ask
    return idx


def n_states(limit: int, sighted: bool = True) -> int:
    base = 2 * limit + 1
    return base * N_IMBALANCE * N_POSITION * N_POSITION if sighted else base


def q_learning(env, sighted: bool = True, n_steps: int = 60_000, beta: float = 0.999,
               lr0: float = 1.0, epsilon0: float = 0.5, rng=None):
    """Batched tabular Q-learning against a :class:`QueueBookEnv`.

    Every book in the batch is an independent copy of the environment sharing one
    table, which is still Q-learning -- each transition is a real sample and the
    update is the standard one -- and lets the table see tens of millions of
    transitions in a minute of numpy.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_act = len(ACTIONS)
    table = np.zeros((n_states(env.limit, sighted), n_act))
    counts = np.zeros_like(table)
    bid_of = np.array([a for a, _ in ACTIONS])
    ask_of = np.array([b for _, b in ACTIONS])

    obs = env.reset()
    state = encode(obs, env.limit, sighted)
    history = []
    for step in range(n_steps):
        frac = step / n_steps
        eps = epsilon0 * (1.0 - 0.98 * frac)
        greedy = table[state].argmax(axis=1)
        explore = rng.random(env.batch) < eps
        action = np.where(explore, rng.integers(0, n_act, env.batch), greedy)

        obs, reward, _info = env.step(bid_of[action], ask_of[action])
        nxt = encode(obs, env.limit, sighted)
        target = reward + beta * table[nxt].max(axis=1)
        np.add.at(counts, (state, action), 1.0)
        lr = lr0 / (1.0 + counts[state, action]) ** 0.7
        np.add.at(table, (state, action), lr * (target - table[state, action]))
        state = nxt
        if (step + 1) % max(n_steps // 40, 1) == 0:
            history.append(float(np.mean(reward)) / env.dt)
    return table, np.asarray(history)


def collect(env, policy, n_steps: int, sighted: bool, rng, epsilon: float = 0.2):
    r"""Interact with the environment and count what happened.

    Returns transition counts, summed rewards and visit counts over the tabular
    state-action space.  Nothing about the environment's parameters is used:
    the agent sees observations, chooses actions and receives rewards, which is
    what makes this reinforcement learning rather than dynamic programming on a
    known model.
    """
    n_state = n_states(env.limit, sighted)
    n_act = len(ACTIONS)
    transitions = np.zeros((n_state, n_act, n_state), dtype=np.float32)
    reward_sum = np.zeros((n_state, n_act))
    visits = np.zeros((n_state, n_act))
    bid_of = np.array([a for a, _ in ACTIONS])
    ask_of = np.array([b for _, b in ACTIONS])

    obs = env.reset()
    state = encode(obs, env.limit, sighted)
    for _ in range(n_steps):
        action = policy(obs, env.limit) if policy is not None else None
        if action is None:
            action = rng.integers(0, n_act, env.batch)
        explore = rng.random(env.batch) < epsilon
        action = np.where(explore, rng.integers(0, n_act, env.batch), action)
        obs, reward, _info = env.step(bid_of[action], ask_of[action])
        nxt = encode(obs, env.limit, sighted)
        np.add.at(transitions, (state, action, nxt), 1.0)
        np.add.at(reward_sum, (state, action), reward)
        np.add.at(visits, (state, action), 1.0)
        state = nxt
    return transitions, reward_sum, visits


def solve(transitions, reward_sum, visits, min_visits: int = 30, sweeps: int = 400,
          tol: float = 1e-12):
    r"""Average-reward value iteration on the counted model.

    The reward of a state-action pair is estimated by **averaging** its samples,
    whose error falls as :math:`1/\sqrt{n}`.  Temporal-difference learning
    instead bootstraps through a value that is itself noisy, and in this
    environment the per-step reward is dominated by marking inventory across a
    price move: a term with a mean near :math:`10^{-5}` and a standard deviation
    near :math:`3\times10^{-2}`.  Averaging resolves that with thousands of
    samples; bootstrapping needs millions.

    Returns ``(policy, bias, gain, usable)``.
    """
    n_state, n_act, _ = transitions.shape
    totals = transitions.sum(axis=2, keepdims=True)
    probs = np.divide(transitions, np.maximum(totals, 1.0))
    mean_reward = np.divide(reward_sum, np.maximum(visits, 1.0))
    usable = visits >= min_visits
    # Never-tried actions are not "worth minus infinity", they are unknown; the
    # policy simply may not pick them.
    penalty = np.where(usable, 0.0, -np.inf)

    bias = np.zeros(n_state)
    gain = 0.0
    for _ in range(sweeps):
        q = mean_reward + penalty + probs @ bias
        updated = np.max(np.where(np.isfinite(q), q, -np.inf), axis=1)
        updated = np.where(np.isfinite(updated), updated, 0.0)
        gain = float(updated[0])
        updated = updated - gain
        if np.max(np.abs(updated - bias)) < tol:
            bias = updated
            break
        bias = updated
    q = mean_reward + penalty + probs @ bias
    policy = np.argmax(np.where(np.isfinite(q), q, -np.inf), axis=1)
    policy = np.where(usable.any(axis=1), policy, 0)
    return policy, bias, gain, usable


def learn(env_factory, sighted: bool, rounds: int = 3, steps: int = 12_000,
          rng=None, min_visits: int = 30):
    """Alternate between acting and re-solving, which is all Dyna ever was.

    The first round explores at random; later rounds follow the policy found so
    far with a little exploration, so the states a good policy actually visits
    are the ones that end up well estimated.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_state = n_states(env_factory().limit, sighted)
    transitions = np.zeros((n_state, len(ACTIONS), n_state), dtype=np.float32)
    reward_sum = np.zeros((n_state, len(ACTIONS)))
    visits = np.zeros((n_state, len(ACTIONS)))
    policy = None
    history = []
    for round_index in range(rounds):
        acting = None
        if policy is not None:
            table_policy = policy

            def acting(obs, limit, table_policy=table_policy):
                return table_policy[encode(obs, limit, sighted)]

        env = env_factory()
        t, r, v = collect(env, acting, steps, sighted, rng,
                          epsilon=0.5 if round_index == 0 else 0.15)
        transitions += t
        reward_sum += r
        visits += v
        policy, bias, gain, usable = solve(transitions, reward_sum, visits,
                                           min_visits=min_visits)
        history.append({"round": round_index, "gain_per_second": gain / env.dt,
                        "covered": float(usable.mean())})
    return policy, np.asarray([h["gain_per_second"] for h in history]), history


def table_policy_fn(policy, sighted: bool = True):
    """Wrap a solved policy so :func:`evaluate` can run it."""
    bid_of = np.array([a for a, _ in ACTIONS])
    ask_of = np.array([b for _, b in ACTIONS])

    def run(obs, limit):
        action = policy[encode(obs, limit, sighted)]
        return bid_of[action], ask_of[action]

    return run


def greedy_policy(table, sighted: bool = True):
    """Turn a learned table into a callable the evaluator can run."""
    best = table.argmax(axis=1)
    bid_of = np.array([a for a, _ in ACTIONS])
    ask_of = np.array([b for _, b in ACTIONS])

    def policy(obs, limit):
        action = best[encode(obs, limit, sighted)]
        return bid_of[action], ask_of[action]

    return policy


def threshold_policy(inventory_max: int, imbalance_min: float = -1.1):
    r"""Quote the bid unless already long, and unless the bid queue looks thin.

    Two readable parameters:

    ``inventory_max``
        stop bidding once inventory reaches it, and mirror on the ask.  This is
        the whole of what a policy that cannot see the book can do, and it is
        the shape the closed form of chapter 05 has: quote as a function of
        inventory alone.
    ``imbalance_min``
        only bid when the imbalance is at least this.  A large bid queue means
        the *ask* is the side likely to empty, so the next move is likely up,
        which is when a buyer wants to be filled.  Setting it below -1 disables
        the rule and recovers the blind family exactly.
    """

    def policy(obs, limit):
        inventory, imbalance = obs[:, 0], obs[:, 1]
        bid = np.where((inventory < inventory_max) & (imbalance >= imbalance_min),
                       AT_TOUCH, OUT)
        ask = np.where((inventory > -inventory_max) & (-imbalance >= imbalance_min),
                       AT_TOUCH, OUT)
        return bid, ask

    return policy


def policy_search(env_factory, inventory_grid, imbalance_grid, seeds=(11, 12, 13),
                  n_steps: int = 20_000, warmup: int = 1_000):
    r"""Choose a policy by simulating every candidate on the *same* price paths.

    The environment's randomness does not depend on the agent -- the maker's
    order is never part of the queue, so it never changes the book's dynamics --
    which means two policies run under the same seed see an identical market.
    Their difference is then a paired comparison, and the noise that makes
    tabular value estimation hopeless here largely cancels.

    Returns ``(best, table)`` where ``table`` has one row per candidate.
    """
    rows = []
    for inventory_max in inventory_grid:
        for imbalance_min in imbalance_grid:
            policy = threshold_policy(int(inventory_max), float(imbalance_min))
            per_seed = []
            for seed in seeds:
                env = env_factory(seed)
                per_seed.append(evaluate(env, policy, n_steps=n_steps, warmup=warmup))
            reward = np.array([m["reward_per_second"] for m in per_seed])
            rows.append({
                "inventory_max": int(inventory_max),
                "imbalance_min": float(imbalance_min),
                "reward_per_second": float(reward.mean()),
                "reward_se": float(reward.std(ddof=1) / np.sqrt(reward.size)) if reward.size > 1 else 0.0,
                "fills_per_second": float(np.mean([m["fills_per_second"] for m in per_seed])),
                "inventory_abs": float(np.mean([m["inventory_abs"] for m in per_seed])),
                "sighted": bool(imbalance_min > -1.0),
            })
    table = rows
    best = max(rows, key=lambda r: r["reward_per_second"])
    return best, table


def rebate_frontier(table, rebates):
    r"""What each policy family earns as a function of the make rebate.

    A rebate is paid per fill and does not change the book, and the environment's
    randomness does not depend on the agent, so for a *fixed* policy

    .. math:: \text{reward}(z) = \text{reward}(0) + z \times \text{fills per second}

    exactly.  The frontier is therefore the upper envelope of a set of straight
    lines, and it costs nothing beyond the simulations already run at zero
    rebate.  Where a family's envelope crosses zero is the rebate an exchange
    would have to pay to keep that kind of maker in the market -- which is the
    quantity chapter 06 solves for, arriving from the other side.
    """
    rebates = np.asarray(rebates, dtype=float)
    out = {}
    for sighted in (False, True):
        rows = [r for r in table if r["sighted"] == sighted]
        if not rows:
            continue
        base = np.array([r["reward_per_second"] for r in rows])
        fills = np.array([r["fills_per_second"] for r in rows])
        envelope = np.max(base[:, None] + rebates[None, :] * fills[:, None], axis=0)
        positive = np.flatnonzero(envelope > 0)
        break_even = float(rebates[positive[0]]) if positive.size else float("nan")
        out["sighted" if sighted else "blind"] = {
            "rebates": rebates, "reward": envelope, "break_even": break_even,
        }
    return out


def always_at_touch(obs, limit):
    n = obs.shape[0]
    return np.full(n, AT_TOUCH), np.full(n, AT_TOUCH)


def stay_out(obs, limit):
    n = obs.shape[0]
    return np.full(n, OUT), np.full(n, OUT)


def evaluate(env, policy, n_steps: int = 40_000, warmup: int = 2_000):
    """Run a policy and report what it earns and the risk it runs.

    Reward is per unit time, not per step, so environments with different clocks
    are comparable.  Inventory is reported too: a policy that earns more by
    carrying more is not obviously better, and the chapter should be able to say
    which one it is.
    """
    obs = env.reset()
    total = 0.0
    inventories = []
    fills = 0
    for step in range(n_steps):
        bid, ask = policy(obs, env.limit)
        obs, reward, info = env.step(bid, ask)
        if step >= warmup:
            total += float(reward.sum())
            fills += int(info["fills"].sum())
            inventories.append(env.inventory.copy())
    seconds = (n_steps - warmup) * env.dt * env.batch
    inventory = np.concatenate(inventories)
    return {
        "reward_per_second": total / seconds,
        "fills_per_second": fills / seconds,
        "inventory_sd": float(inventory.std()),
        "inventory_abs": float(np.abs(inventory).mean()),
        "at_limit_share": float(np.mean(np.abs(inventory) >= env.limit)),
    }
