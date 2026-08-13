import numpy as np

from DQN.replay import ReplayBuffer, TransitionBatch


def _batch(n=3):
    obs = {"board": np.zeros((n, 20, 10), dtype=np.uint8), "active": np.zeros((n, 20, 10), dtype=np.uint8)}
    next_obs = {k: v.copy() for k, v in obs.items()}
    return TransitionBatch(obs, np.arange(n) % 6, np.ones(n, dtype=np.float32), next_obs, np.zeros(n, dtype=bool))


def test_replay_add_and_sample():
    replay = ReplayBuffer(5, seed=1)
    replay.add_batch(_batch(7))
    assert len(replay) == 5
    sample = replay.sample(4)
    assert sample.obs["board"].shape == (4, 20, 10)
    assert sample.actions.shape == (4,)

