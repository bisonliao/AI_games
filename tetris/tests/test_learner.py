from types import SimpleNamespace

import numpy as np

from DQN.learner import Learner
from DQN.replay import TransitionBatch


def _learner_config():
    return SimpleNamespace(
        replay_capacity=64,
        batch_size=2,
        learning_starts=4,
        update_every=2,
        target_update_every=100,
        learning_rate=1e-4,
        gamma=0.99,
        gradient_clip_norm=10.0,
        to_dict=lambda: {},
    )


def _batch(size: int) -> TransitionBatch:
    obs = {
        "board": np.zeros((size, 20, 10), dtype=np.uint8),
        "active": np.zeros((size, 20, 10), dtype=np.uint8),
        "current_piece": np.zeros((size, 7), dtype=np.int8),
        "next_piece": np.zeros((size, 7), dtype=np.int8),
        "rotation": np.zeros((size, 4), dtype=np.int8),
        "position": np.zeros((size, 2), dtype=np.float32),
    }
    obs["current_piece"][:, 0] = 1
    obs["next_piece"][:, 1] = 1
    obs["rotation"][:, 0] = 1
    return TransitionBatch(
        obs=obs,
        actions=np.zeros(size, dtype=np.int64),
        rewards=np.ones(size, dtype=np.float32),
        next_obs={key: value.copy() for key, value in obs.items()},
        terminated=np.zeros(size, dtype=np.bool_),
    )


def test_learner_catches_up_updates_for_large_ipc_batch():
    learner = Learner(_learner_config(), device="cpu", seed=0)
    learner.optimizer.param_groups[0]["lr"] = 5e-5
    learner.add(_batch(10))
    updates = 0
    assert learner.update()
    updates += 1
    while learner.update():
        updates += 1
    assert updates == 4
    assert learner.gradient_updates == 4
    assert learner._next_update_transition == 12
    stats = learner.pop_training_stats()
    assert stats["lr"] == 5e-5
    assert stats["replay_size"] == 10
    assert all(np.isfinite(stats[key]) for key in ("loss", "q_mean", "target_mean", "gradient_norm"))
    assert learner.pop_training_stats() == {}
