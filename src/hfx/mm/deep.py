r"""A deep Q-network for the reacting book, and an honest test of whether it helps.

Chapter 08's tabular agent has to bucket the book coarsely, because a table with
thousands of cells cannot be estimated from a reward whose standard deviation is
three thousand times its mean.  A network does not have that problem in the same
way: it shares statistical strength across nearby states instead of estimating
each one alone, so it can in principle use the *continuous* imbalance and queue
position that the table has to round off.

Whether it actually does is the question, and it is asked against the same
benchmarks as everything else -- the blind policy, the threshold family found by
direct search -- on the same environment and the same paired price paths.

``torch`` is an optional dependency.  Everything else in the repository runs
without it, the test suite skips this module when it is absent, and continuous
integration never installs it.
"""

from __future__ import annotations

import numpy as np

from .queue_agent import ACTIONS

try:  # pragma: no cover - exercised by whether torch is installed
    import torch
    from torch import nn

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def available() -> bool:
    """Whether the deep variant can run at all in this environment."""
    return HAVE_TORCH


def features(obs, limit: int):
    """Scale the observation so a network sees inputs of comparable size."""
    inventory = obs[:, 0] / max(limit, 1)
    imbalance = obs[:, 1]
    resting_bid = (obs[:, 2] >= 0).astype(float)
    resting_ask = (obs[:, 3] >= 0).astype(float)
    position_bid = np.where(obs[:, 2] >= 0, obs[:, 2], 0.0)
    position_ask = np.where(obs[:, 3] >= 0, obs[:, 3], 0.0)
    return np.column_stack([inventory, imbalance, resting_bid, resting_ask,
                            position_bid, position_ask]).astype(np.float32)


N_FEATURES = 6


def build_network(hidden: int = 64, seed: int = 0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(N_FEATURES, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, len(ACTIONS)),
    )


def train(env, n_steps: int = 6_000, gamma: float = 0.99, lr: float = 1e-3,
          epsilon0: float = 0.5, hidden: int = 64, target_every: int = 200,
          seed: int = 0, rng=None, reward_scale: float | None = None):
    """Deep Q-learning against a :class:`QueueBookEnv`.

    The whole batch of books is one gradient step: every environment step yields
    ``env.batch`` transitions, which is what makes this affordable on a CPU.

    ``reward_scale`` defaults to one tick.  Raw rewards here are of order
    :math:`10^{-4}`, and a squared-error loss on targets that small produces
    gradients that never move the network off its initialisation; measuring the
    reward in ticks is the same problem with numbers an optimiser can see.
    """
    if not HAVE_TORCH:
        raise RuntimeError("torch is not installed; install the 'deep' extra")
    rng = np.random.default_rng(seed) if rng is None else rng
    scale = 1.0 / env.tick if reward_scale is None else float(reward_scale)
    net = build_network(hidden, seed)
    target = build_network(hidden, seed)
    target.load_state_dict(net.state_dict())
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    bid_of = np.array([a for a, _ in ACTIONS])
    ask_of = np.array([b for _, b in ACTIONS])
    n_act = len(ACTIONS)

    obs = env.reset()
    losses = []
    for step in range(n_steps):
        eps = epsilon0 * (1.0 - 0.95 * step / n_steps)
        x = torch.from_numpy(features(obs, env.limit))
        with torch.no_grad():
            greedy = net(x).argmax(dim=1).numpy()
        explore = rng.random(env.batch) < eps
        action = np.where(explore, rng.integers(0, n_act, env.batch), greedy)

        nxt_obs, reward, _info = env.step(bid_of[action], ask_of[action])
        xn = torch.from_numpy(features(nxt_obs, env.limit))
        with torch.no_grad():
            bootstrap = target(xn).max(dim=1).values
        y = torch.from_numpy((reward * scale).astype(np.float32)) + gamma * bootstrap
        q = net(x).gather(1, torch.from_numpy(action[:, None])).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimiser.step()
        if (step + 1) % target_every == 0:
            target.load_state_dict(net.state_dict())
        losses.append(float(loss.item()))
        obs = nxt_obs
    return net, np.asarray(losses)


def policy_from(net):
    """Wrap a trained network so :func:`hfx.mm.queue_agent.evaluate` can run it."""
    bid_of = np.array([a for a, _ in ACTIONS])
    ask_of = np.array([b for _, b in ACTIONS])

    def policy(obs, limit):
        with torch.no_grad():
            action = net(torch.from_numpy(features(obs, limit))).argmax(dim=1).numpy()
        return bid_of[action], ask_of[action]

    return policy
