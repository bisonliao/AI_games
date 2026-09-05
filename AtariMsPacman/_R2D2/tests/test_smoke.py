from __future__ import annotations

import ast
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
from queue import Empty

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch

from _R2D2.config import R2D2Config
from _R2D2.evaluator import evaluator_process
from _R2D2.messages import EvaluationRequest, EvaluationResult, EvaluatorStop
from _R2D2.network import RecurrentDuelingQNetwork
from _R2D2.train import run_training


def test_package_import_boundary() -> None:
    package = Path(__file__).parents[1]
    forbidden = {"DQN", "R2D2", "_HRA", "hra"}
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(name.split(".")[0] in forbidden for name in names)


def test_recurrent_evaluator_reports_capped_raw_score_and_return(tmp_path) -> None:
    base = R2D2Config()
    config = replace(
        base,
        actor_env=replace(
            base.actor_env,
            num_envs=1,
            noop_max=0,
            repeat_action_probability=0.0,
        ),
        hidden_size=32,
        evaluation_episodes=1,
        evaluation_max_episode_steps=5,
        learner_device="cpu",
    )
    model = RecurrentDuelingQNetwork(
        config.observation_shape, config.action_count, config.hidden_size
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint_path = tmp_path / "constant_noop.pt"
    torch.save({"online_state_dict": model.state_dict()}, checkpoint_path)
    context = mp.get_context("spawn")
    request_queue = context.Queue()
    result_queue = context.Queue()
    error_queue = context.Queue()
    process = context.Process(
        target=evaluator_process,
        args=(config, request_queue, result_queue, error_queue),
    )
    process.start()
    request_queue.put(EvaluationRequest(str(checkpoint_path), 123))
    request_queue.put(EvaluatorStop())
    try:
        result: EvaluationResult = result_queue.get(timeout=30.0)
    except Empty:
        try:
            error = error_queue.get_nowait()
        except Empty:
            error = None
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        raise AssertionError(f"evaluator did not return: {error}")
    process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
    try:
        error = error_queue.get_nowait()
    except Empty:
        error = None
    assert error is None
    assert process.exitcode == 0
    assert result.checkpoint_transition == 123
    assert result.episode_lengths == [5]
    assert result.capped_episodes == 1
    assert len(result.episode_returns) == len(result.episode_raw_scores) == 1


def test_tiny_multiprocess_training_writes_events_and_checkpoint(tmp_path) -> None:
    base = R2D2Config()
    config = replace(
        base,
        num_actors=1,
        actor_env=replace(
            base.actor_env,
            num_envs=1,
            noop_max=0,
            repeat_action_probability=0.0,
        ),
        actor_sequence_chunk_size=1,
        total_transitions=48,
        replay_capacity_sequences=16,
        learning_starts=1,
        learner_batch_size=1,
        tensorboard_interval_transitions=16,
        checkpoint_interval_transitions=48,
        evaluation_enabled=False,
        learner_device="cpu",
        runs_dir=tmp_path / "runs",
        checkpoints_dir=tmp_path / "checkpoint",
        shutdown_timeout_seconds=60.0,
    )
    run_training(config)
    run_dirs = list(config.runs_dir.iterdir())
    checkpoint_dirs = list(config.checkpoints_dir.iterdir())
    assert len(run_dirs) == len(checkpoint_dirs) == 1
    assert "-pid" in run_dirs[0].name
    accumulator = EventAccumulator(str(run_dirs[0]))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert {
        "rollout/transitions_per_second",
        "rollout/actor_collection_transitions_per_second",
        "rollout/queue_wait_fraction",
        "rollout/episode_return_mean",
        "rollout/raw_score_mean",
        "replay/learning_transitions",
        "replay/is_weight_min",
        "replay/is_weight_mean",
        "replay/is_weight_max",
    } <= tags
    paths = list(checkpoint_dirs[0].glob("checkpoint_step_*.pt"))
    assert len(paths) == 1
    checkpoint = torch.load(paths[0], map_location="cpu", weights_only=False)
    assert checkpoint["global_transitions"] == 48
    assert "online_state_dict" in checkpoint
    assert "target_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_tiny_training_waits_for_async_evaluation(tmp_path) -> None:
    base = R2D2Config()
    config = replace(
        base,
        num_actors=1,
        actor_env=replace(
            base.actor_env,
            num_envs=1,
            noop_max=0,
            repeat_action_probability=0.0,
        ),
        hidden_size=32,
        actor_sequence_chunk_size=1,
        total_transitions=48,
        replay_capacity_sequences=16,
        learning_starts=1,
        learner_batch_size=1,
        updates_per_sequence=0.01,
        tensorboard_interval_transitions=16,
        checkpoint_interval_transitions=48,
        evaluation_enabled=True,
        evaluation_episodes=1,
        evaluation_max_episode_steps=5,
        learner_device="cpu",
        runs_dir=tmp_path / "runs",
        checkpoints_dir=tmp_path / "checkpoint",
        shutdown_timeout_seconds=60.0,
        evaluation_shutdown_timeout_seconds=60.0,
    )
    run_training(config)
    run_dir = next(config.runs_dir.iterdir())
    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert {"eval/episode_return_mean", "eval/raw_score_mean"} <= tags
