from dataclasses import replace

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch

from DQN.config import DQNConfig
from DQN.train import run_training


def test_real_multiprocess_training_checkpoint_and_evaluation(tmp_path) -> None:
    base = DQNConfig()
    actor_env = replace(
        base.actor_env,
        num_envs=2,
        frame_skip=4,
        noop_max=0,
        repeat_action_probability=0.0,
    )
    config = replace(
        base,
        num_actors=1,
        actor_env=actor_env,
        actor_transition_batch_size=8,
        rollout_queue_capacity=2,
        metrics_queue_capacity=16,
        total_transitions=32,
        replay_capacity=64,
        learning_starts=8,
        learner_batch_size=8,
        updates_per_transition=0.25,
        target_sync_interval_updates=2,
        tensorboard_interval_transitions=16,
        checkpoint_interval_transitions=32,
        evaluation_episodes=1,
        evaluation_enabled=False,
        learner_device="cpu",
        runs_dir=tmp_path / "runs",
        checkpoints_dir=tmp_path / "chkpt",
        shutdown_timeout_seconds=60.0,
        evaluation_shutdown_timeout_seconds=60.0,
    )
    run_training(config)

    run_directories = list((tmp_path / "runs").iterdir())
    checkpoint_directories = list((tmp_path / "chkpt").iterdir())
    assert len(run_directories) == 1
    assert "-pid" in run_directories[0].name
    assert any(run_directories[0].glob("events.out.tfevents.*"))
    event_accumulator = EventAccumulator(str(run_directories[0]))
    event_accumulator.Reload()
    scalar_tags = set(event_accumulator.Tags()["scalars"])
    assert "rollout/transitions_per_second" in scalar_tags
    assert "learner/consumed_transitions_per_second" in scalar_tags
    assert "train/epsilon" in scalar_tags
    assert "train/loss" in scalar_tags
    assert "rollout/observation_unique_fraction" in scalar_tags
    assert len(checkpoint_directories) == 1
    checkpoint_paths = list(checkpoint_directories[0].glob("checkpoint_step_*.pt"))
    assert len(checkpoint_paths) == 1
    checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    assert checkpoint["global_transitions"] == 32
    assert checkpoint["learner_updates"] == 8
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["config"]["actor_env"]["step_cost"] == 0.0
    assert checkpoint["config"]["actor_env"]["clip_training_reward"] is False
