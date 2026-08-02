from pathlib import Path

import numpy as np
import torch

from dqn.checkpoint import load_checkpoint, save_checkpoint
from dqn.config import DQNConfig
from dqn.learner import DQNLearner
from dqn.nstep import NStepTransition
from dqn.replay import ReplayBuffer


def sample_transition(index: int) -> NStepTransition:
    return NStepTransition(
        observation=np.full(5, index, np.float32),
        action=index % 3,
        reward=1.0,
        next_observation=np.full(5, index + 1, np.float32),
        discount=0.99,
        terminated=False,
        truncated=False,
    )


def test_full_checkpoint_roundtrip(tmp_path: Path):
    config = DQNConfig(
        hidden_dim=16,
        replay_capacity=32,
        replay_warmup=4,
        batch_size=4,
    )
    learner = DQNLearner(config, torch.device("cpu"))
    replay = ReplayBuffer(32, (5,))
    for index in range(8):
        replay.add(sample_transition(index))
    rng = np.random.default_rng(9)
    learner.update(replay, rng)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, learner, config, 123, rng, replay, best_success_rate=0.75)

    restored_learner = DQNLearner(config, torch.device("cpu"))
    restored_replay = ReplayBuffer(32, (5,))
    state = load_checkpoint(path, restored_learner, restored_replay)
    assert state["env_steps"] == 123
    assert state["best_success_rate"] == 0.75
    assert len(restored_replay) == len(replay)
    for expected, actual in zip(
        learner.online.parameters(), restored_learner.online.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual)
