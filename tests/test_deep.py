"""The optional deep agent of chapter 08.

Skipped in full when torch is absent, which is how continuous integration runs
it -- the point of the ``deep`` extra is that nothing else in the repository
depends on it.  What is checked here is that the module is wired correctly: the
observation is scaled into something a network can read, the network produces one
value per action, and a trained policy returns legal actions.

Deliberately *not* checked is that the deep agent performs well.  It does not,
and chapter 08 reports why: the per-step reward is dominated by marking inventory
across a price move, and a bootstrap through that noise diverges.  A test that
demanded good performance would be a test tuned until it passed.
"""

import numpy as np
import pytest

from hfx.mm import deep
from hfx.mm.queue_agent import ACTIONS
from hfx.mm.queue_env import AT_TOUCH, OUT
from test_queue_env import build   # tests/ is on sys.path under pytest

pytestmark = pytest.mark.skipif(not deep.available(), reason="torch is not installed")


def test_features_scale_the_observation():
    obs = np.array([
        [3.0, 0.5, -1.0, 0.25],       # long, bid-heavy, resting only on the ask
        [-3.0, -0.5, 0.75, -1.0],
    ])
    x = deep.features(obs, limit=6)
    assert x.shape == (2, deep.N_FEATURES)
    assert x.dtype == np.float32
    assert x[0, 0] == pytest.approx(0.5) and x[1, 0] == pytest.approx(-0.5)
    # "not resting" is a flag, not a negative position: a network should never
    # see -1 as a queue position and try to interpolate through it.
    assert x[0, 2] == 0.0 and x[0, 3] == 1.0
    assert x[0, 4] == 0.0 and x[0, 5] == pytest.approx(0.25)
    assert np.all(x[:, 4:] >= 0.0)


def test_the_network_gives_one_value_per_action():
    import torch

    net = deep.build_network(hidden=8, seed=0)
    out = net(torch.zeros(5, deep.N_FEATURES))
    assert out.shape == (5, len(ACTIONS))


def test_a_short_run_trains_and_returns_legal_actions():
    env = build(batch=64)
    net, losses = deep.train(env, n_steps=25, hidden=8, seed=0)
    assert losses.shape == (25,)
    assert np.all(np.isfinite(losses))

    policy = deep.policy_from(net)
    bid, ask = policy(env.reset(), env.limit)
    assert bid.shape == (env.batch,) and ask.shape == (env.batch,)
    assert set(np.unique(bid)) <= {OUT, AT_TOUCH}
    assert set(np.unique(ask)) <= {OUT, AT_TOUCH}


def test_training_refuses_to_run_without_torch(monkeypatch):
    monkeypatch.setattr(deep, "HAVE_TORCH", False)
    with pytest.raises(RuntimeError, match="torch"):
        deep.train(build(batch=8), n_steps=1)
