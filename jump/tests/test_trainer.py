from __future__ import annotations

import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from td3.trainer import (
    TrainConfig,
    _create_run_directory,
    train_distributed,
)


def test_timestamped_run_directories_do_not_collide(tmp_path) -> None:
    first = _create_run_directory(str(tmp_path), "same name")
    second = _create_run_directory(str(tmp_path), "same name")
    pattern = r"\d{8}-\d{6}-pid\d+-same_name(?:-\d{2})?"
    assert re.fullmatch(pattern, first.name)
    assert re.fullmatch(pattern, second.name)
    assert first != second


def test_actor_learner_cpu_smoke(tmp_path) -> None:
    checkpoint = tmp_path / "smoke.pt"
    config = TrainConfig(
        total_transitions=256,
        num_actors=1,
        envs_per_actor=2,
        actor_chunk_size=64,
        replay_capacity=512,
        batch_size=32,
        learning_starts=64,
        random_steps=64,
        updates_per_transition=0.05,
        log_interval=64,
        eval_interval=0,
        final_eval_episodes=10,
        device="cpu",
        run_root=str(tmp_path / "runs"),
        run_name="smoke",
        checkpoint_path=str(checkpoint),
    )
    result = train_distributed(config)
    assert result.transitions >= 256
    assert result.updates > 0
    assert checkpoint.exists()
    run_dir = Path(result.run_dir)
    assert re.fullmatch(r"\d{8}-\d{6}-pid\d+-smoke", run_dir.name)
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    assert event_files

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    assert {
        "rollout/success_rate",
        "evaluation/final_success_rate",
        "queue/transition_size",
        "queue/actor_blocked_seconds_total",
        "queue/actor_queue_full_events_total",
        "queue/actor_dropped_transitions_total",
        "queue/learner_wait_seconds_mean",
        "queue/learner_wait_seconds_max",
        "queue/learner_long_wait_count_total",
        "queue/learner_wait_timeout_count_total",
        "queue/learner_discarded_prefetch_transitions_total",
        "actors/actor_0/queue_blocked_seconds_total",
        "actors/actor_0/queue_full_events_total",
        "actors/actor_0/dropped_transitions_total",
    } <= scalar_tags
