from dataclasses import replace

import pytest
import torch

from DQN.config import DQNConfig
from DQN.train import run_training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_single_gpu_learner_path(tmp_path) -> None:
    base = DQNConfig()
    config = replace(
        base,
        num_actors=1,
        actor_env=replace(
            base.actor_env,
            num_envs=2,
            noop_max=0,
            repeat_action_probability=0.0,
        ),
        actor_transition_batch_size=8,
        rollout_queue_capacity=2,
        total_transitions=16,
        replay_capacity=32,
        learning_starts=8,
        learner_batch_size=8,
        updates_per_transition=0.25,
        tensorboard_interval_transitions=16,
        checkpoint_interval_transitions=1_000,
        evaluation_enabled=False,
        learner_device="cuda:0",
        runs_dir=tmp_path / "runs",
        checkpoints_dir=tmp_path / "chkpt",
        shutdown_timeout_seconds=60.0,
    )
    run_training(config)
    assert len(list((tmp_path / "runs").iterdir())) == 1
