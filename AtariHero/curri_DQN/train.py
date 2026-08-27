"""Train exactly one H.E.R.O. curriculum stage with actor-learner DDDQN."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .actor import actor_process, epsilon_at
from .config import TrainConfig
from .envs import make_training_env
from .evaluator import stage_evaluator_process
from .messages import (
    EpisodeSummary,
    PackedTransition,
    StageEvaluationResult,
    WorkerFailure,
    unpack_observations,
)
from .model import DuelingDQN
from .replay import ReplayBuffer


class TrainingMetrics:
    def __init__(self, config: TrainConfig) -> None:
        self.returns: deque[float] = deque(maxlen=config.episode_window)
        self.ale_score_returns: deque[float] = deque(maxlen=config.episode_window)
        self.lengths: deque[int] = deque(maxlen=config.episode_window)
        self.current_successes: deque[bool] = deque(maxlen=config.episode_window)
        self.earlier_successes: deque[bool] = deque(maxlen=config.episode_window)
        self.after_curri_successes: deque[bool] = deque(maxlen=config.episode_window)
        self.timeouts: deque[bool] = deque(maxlen=config.episode_window)
        self.timeout_lengths: deque[int] = deque(maxlen=config.episode_window)
        self.walls_destroyed: deque[int] = deque(maxlen=config.episode_window)
        self.creatures_killed: deque[int] = deque(maxlen=config.episode_window)
        self.miner_rescue_events: deque[int] = deque(maxlen=config.episode_window)
        self.dynamite_bonus_sticks: deque[int] = deque(maxlen=config.episode_window)
        self.unmapped_ale_rewards: deque[float] = deque(maxlen=config.episode_window)
        self.task_successes: dict[str, deque[bool]] = {}
        self.checkpoint_successes: dict[str, deque[bool]] = {}
        self.checkpoint_events: dict[str, deque[tuple[int, int, int]]] = {}
        self.episode_window = config.episode_window
        self.episode_count = 0

    def consume(self, summary: EpisodeSummary, target_stage: int) -> None:
        self.returns.append(summary.episode_return)
        self.ale_score_returns.append(summary.ale_score_return)
        self.lengths.append(summary.episode_length)
        self.timeouts.append(summary.timeout)
        if summary.timeout:
            self.timeout_lengths.append(summary.episode_length)
        self.walls_destroyed.append(summary.walls_destroyed)
        self.creatures_killed.append(summary.creatures_killed)
        self.miner_rescue_events.append(summary.miner_rescue_events)
        self.dynamite_bonus_sticks.append(summary.dynamite_bonus_sticks)
        self.unmapped_ale_rewards.append(summary.unmapped_ale_reward)
        if summary.reset_stage == target_stage:
            self.current_successes.append(summary.success)
        elif 1 <= summary.reset_stage < target_stage:
            self.earlier_successes.append(summary.success)
        elif summary.reset_stage == 0:
            self.after_curri_successes.append(summary.success)
        self.task_successes.setdefault(
            summary.task_id, deque(maxlen=self.episode_window)
        ).append(summary.success)
        self.checkpoint_successes.setdefault(
            summary.checkpoint_id, deque(maxlen=self.episode_window)
        ).append(summary.success)
        self.checkpoint_events.setdefault(
            summary.checkpoint_id,
            deque(maxlen=self.episode_window),
        ).append(
            (
                summary.walls_destroyed,
                summary.creatures_killed,
                summary.miner_rescue_events,
            )
        )
        self.episode_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "returns": list(self.returns),
            "ale_score_returns": list(self.ale_score_returns),
            "lengths": list(self.lengths),
            "current_successes": list(self.current_successes),
            "earlier_successes": list(self.earlier_successes),
            "after_curri_successes": list(self.after_curri_successes),
            "timeouts": list(self.timeouts),
            "timeout_lengths": list(self.timeout_lengths),
            "walls_destroyed": list(self.walls_destroyed),
            "creatures_killed": list(self.creatures_killed),
            "miner_rescue_events": list(self.miner_rescue_events),
            "dynamite_bonus_sticks": list(self.dynamite_bonus_sticks),
            "unmapped_ale_rewards": list(self.unmapped_ale_rewards),
            "task_successes": {
                identifier: list(values)
                for identifier, values in self.task_successes.items()
            },
            "checkpoint_successes": {
                identifier: list(values)
                for identifier, values in self.checkpoint_successes.items()
            },
            "checkpoint_events": {
                identifier: list(values)
                for identifier, values in self.checkpoint_events.items()
            },
            "episode_count": self.episode_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.returns.extend(state.get("returns", []))
        self.ale_score_returns.extend(state.get("ale_score_returns", []))
        self.lengths.extend(state.get("lengths", []))
        self.current_successes.extend(state.get("current_successes", []))
        self.earlier_successes.extend(state.get("earlier_successes", []))
        self.after_curri_successes.extend(state.get("after_curri_successes", []))
        self.timeouts.extend(state.get("timeouts", []))
        self.timeout_lengths.extend(state.get("timeout_lengths", []))
        self.walls_destroyed.extend(state.get("walls_destroyed", []))
        self.creatures_killed.extend(state.get("creatures_killed", []))
        self.miner_rescue_events.extend(state.get("miner_rescue_events", []))
        self.dynamite_bonus_sticks.extend(state.get("dynamite_bonus_sticks", []))
        self.unmapped_ale_rewards.extend(state.get("unmapped_ale_rewards", []))
        for identifier, values in state.get("task_successes", {}).items():
            self.task_successes[str(identifier)] = deque(
                values, maxlen=self.episode_window
            )
        for identifier, values in state.get("checkpoint_successes", {}).items():
            self.checkpoint_successes[str(identifier)] = deque(
                values, maxlen=self.episode_window
            )
        for identifier, values in state.get("checkpoint_events", {}).items():
            self.checkpoint_events[str(identifier)] = deque(
                (tuple(item) for item in values),
                maxlen=self.episode_window,
            )
        self.episode_count = int(state.get("episode_count", 0))


def _mean(values: Any) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def publish_weights(
    online_model: DuelingDQN,
    shared_model: DuelingDQN,
    weight_lock: Any,
    weight_version: Any,
) -> None:
    state = _cpu_state_dict(online_model)
    with weight_lock:
        shared_model.load_state_dict(state)
        weight_version.value += 1


def decode_batch(
    transitions: list[PackedTransition], device: torch.device
) -> tuple[torch.Tensor, ...]:
    observations = np.empty((len(transitions), 4, 84, 84), dtype=np.uint8)
    next_observations = np.empty_like(observations)
    actions = np.empty(len(transitions), dtype=np.int64)
    rewards = np.empty(len(transitions), dtype=np.float32)
    terminated = np.empty(len(transitions), dtype=np.float32)
    for index, transition in enumerate(transitions):
        observation, next_observation = unpack_observations(transition.observations)
        observations[index] = observation
        next_observations[index] = next_observation
        actions[index] = transition.action
        rewards[index] = transition.reward
        terminated[index] = float(transition.terminated)
    return (
        torch.from_numpy(observations).to(device, non_blocking=True),
        torch.from_numpy(actions).to(device, non_blocking=True),
        torch.from_numpy(rewards).to(device, non_blocking=True),
        torch.from_numpy(next_observations).to(device, non_blocking=True),
        torch.from_numpy(terminated).to(device, non_blocking=True),
    )


def learner_update(
    online_model: DuelingDQN,
    target_model: DuelingDQN,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: tuple[torch.Tensor, ...],
    gamma: float,
    max_grad_norm: float,
) -> tuple[float, float, float]:
    observations, actions, rewards, next_observations, terminated = batch
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        q_values = online_model(observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_actions = online_model(next_observations).argmax(dim=1)
            next_q_values = target_model(next_observations).gather(
                1, next_actions[:, None]
            ).squeeze(1)
            targets = rewards + gamma * (1.0 - terminated) * next_q_values
        loss = F.smooth_l1_loss(q_values.float(), targets.float())
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(online_model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    td_error = (targets - q_values).detach().abs().mean()
    return float(loss.item()), float(q_values.detach().mean()), float(td_error)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(
    *,
    config: TrainConfig,
    checkpoint_dir: Path,
    online_model: DuelingDQN,
    target_model: DuelingDQN,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    replay: ReplayBuffer,
    metrics: TrainingMetrics,
    transitions: int,
    gradient_steps: int,
    update_credit: float,
    next_checkpoint_transition: int,
    next_eval_transition: int,
    eval_in_flight: bool,
    generated_transitions: int,
    curriculum_identity: dict[str, Any],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_{transitions:012d}.pt"
    payload = {
        "format_version": 11,
        "config": config.as_dict(),
        "curriculum_identity": curriculum_identity,
        "target_stage": config.target_stage,
        "online_model": _cpu_state_dict(online_model),
        "target_model": _cpu_state_dict(target_model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "replay": replay.state_dict() if config.save_replay else None,
        "metrics": metrics.state_dict(),
        "transitions": transitions,
        "gradient_steps": gradient_steps,
        "update_credit": update_credit,
        "next_checkpoint_transition": next_checkpoint_transition,
        "next_eval_transition": next_eval_transition,
        "eval_in_flight": eval_in_flight,
        "generated_transitions": generated_transitions,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(config.gpu),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    atomic_torch_save(payload, path)
    write_json_atomic(
        {"checkpoint": path.name, "transitions": transitions},
        checkpoint_dir / "latest.json",
    )
    return path


def resolve_checkpoint(reference: str, checkpoint_path: Path) -> Path:
    if reference != "latest":
        return Path(reference).resolve()
    latest_path = checkpoint_path / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    return latest_path.parent / latest["checkpoint"]


def validate_resume(
    config: TrainConfig,
    payload: dict[str, Any],
    curriculum_identity: dict[str, Any],
) -> None:
    if int(payload.get("format_version", 0)) != 11:
        raise ValueError(
            "resume requires a format-v11 checkpoint bound to a frozen curriculum"
        )
    if payload.get("curriculum_identity") != curriculum_identity:
        raise ValueError("resume checkpoint curriculum manifest differs from the active manifest")
    if int(payload["target_stage"]) != config.target_stage:
        raise ValueError("resume checkpoint target_stage differs from --stage")
    saved = payload["config"]
    immutable = (
        "target_stage",
        "replay_capacity",
        "max_curriculum_stage",
        "frame_stack",
        "screen_size",
        "action_repeat",
        "train_current_stage_fraction",
        "eval_current_stage_fraction",
        "after_curri",
        "decision_step_penalty",
        "wall_event_reward",
        "creature_event_reward",
        "miner_event_reward",
        "episode_timeout_decisions",
        "timeout_terminal_reward",
        "life_lost_terminal_reward",
    )
    for name in immutable:
        if saved.get(name) != getattr(config, name):
            raise ValueError(f"resume configuration mismatch for {name}")


def drain_episode_metrics(
    metrics_queue: Any,
    metrics: TrainingMetrics,
    target_stage: int,
) -> None:
    while True:
        try:
            message = metrics_queue.get_nowait()
        except queue.Empty:
            return
        if isinstance(message, WorkerFailure):
            raise RuntimeError(f"{message.worker} failed:\n{message.traceback}")
        metrics.consume(message, target_stage)


def drain_eval_results(result_queue: Any) -> list[StageEvaluationResult]:
    results = []
    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            return results
        if isinstance(result, WorkerFailure):
            raise RuntimeError(f"{result.worker} failed:\n{result.traceback}")
        assert isinstance(result, StageEvaluationResult)
        results.append(result)


def split_eval_success(
    result: StageEvaluationResult,
    target_stage: int,
) -> tuple[float, float, int, int]:
    current = [
        success
        for stage, success in zip(result.reset_stages, result.successes, strict=True)
        if stage == target_stage
    ]
    earlier = [
        success
        for stage, success in zip(result.reset_stages, result.successes, strict=True)
        if stage < target_stage
    ]
    return _mean(current), _mean(earlier), len(current), len(earlier)


def log_eval_result(
    writer: SummaryWriter,
    result: StageEvaluationResult,
    target_stage: int,
) -> None:
    if result.reset_stages and all(stage == 0 for stage in result.reset_stages):
        after_rate = _mean(result.successes)
        step = result.checkpoint_step
        writer.add_scalar("success/eval_after_curri", after_rate, step)
        writer.add_scalar("eval/after_curri_episodes", len(result.successes), step)
    current_rate, earlier_rate, current_count, earlier_count = split_eval_success(
        result, target_stage
    )
    step = result.checkpoint_step
    writer.add_scalar("success/eval_current_stage", current_rate, step)
    writer.add_scalar("success/eval_earlier_stages", earlier_rate, step)
    writer.add_scalar("eval/current_stage_episodes", current_count, step)
    writer.add_scalar("eval/earlier_stage_episodes", earlier_count, step)
    writer.add_scalar("eval/episode_return_mean", _mean(result.episode_returns), step)
    writer.add_scalar(
        "eval/ale_score_return_mean", _mean(result.ale_score_returns), step
    )
    writer.add_scalar("eval/episode_length_mean", _mean(result.episode_lengths), step)
    writer.add_scalar("timeout/eval_rate", _mean(result.timeouts), step)
    task_successes: dict[str, list[bool]] = {}
    checkpoint_successes: dict[str, list[bool]] = {}
    checkpoint_timeouts: dict[str, list[bool]] = {}
    checkpoint_walls: dict[str, list[int]] = {}
    checkpoint_creatures: dict[str, list[int]] = {}
    checkpoint_miners: dict[str, list[int]] = {}
    for task_id, checkpoint_id, success in zip(
        result.task_ids,
        result.checkpoint_ids,
        result.successes,
        strict=True,
    ):
        task_successes.setdefault(task_id, []).append(success)
        checkpoint_successes.setdefault(checkpoint_id, []).append(success)
    for checkpoint_id, timed_out in zip(
        result.checkpoint_ids,
        result.timeouts,
        strict=True,
    ):
        checkpoint_timeouts.setdefault(checkpoint_id, []).append(timed_out)
    for checkpoint_id, wall_count, creature_count, miner_count in zip(
        result.checkpoint_ids,
        result.walls_destroyed,
        result.creatures_killed,
        result.miner_rescue_events,
        strict=True,
    ):
        checkpoint_walls.setdefault(checkpoint_id, []).append(wall_count)
        checkpoint_creatures.setdefault(checkpoint_id, []).append(creature_count)
        checkpoint_miners.setdefault(checkpoint_id, []).append(miner_count)
    for identifier, values in sorted(task_successes.items()):
        writer.add_scalar(f"success/eval_task/{identifier}", _mean(values), step)
        writer.add_scalar(f"eval/task_episodes/{identifier}", len(values), step)
    for identifier, values in sorted(checkpoint_successes.items()):
        writer.add_scalar(
            f"success/eval_checkpoint/{identifier}", _mean(values), step
        )
        writer.add_scalar(
            f"eval/checkpoint_episodes/{identifier}", len(values), step
        )
    for identifier, values in sorted(checkpoint_timeouts.items()):
        writer.add_scalar(
            f"timeout/eval_checkpoint/{identifier}", _mean(values), step
        )
    for identifier, values in sorted(checkpoint_walls.items()):
        writer.add_scalar(
            f"events/eval_checkpoint/{identifier}/walls_destroyed",
            _mean(values),
            step,
        )
    for identifier, values in sorted(checkpoint_creatures.items()):
        writer.add_scalar(
            f"events/eval_checkpoint/{identifier}/creatures_killed",
            _mean(values),
            step,
        )
    for identifier, values in sorted(checkpoint_miners.items()):
        writer.add_scalar(
            f"events/eval_checkpoint/{identifier}/miner_rescued",
            _mean(values),
            step,
        )
    writer.flush()


def log_training(
    *,
    writer: SummaryWriter,
    config: TrainConfig,
    metrics: TrainingMetrics,
    replay: ReplayBuffer,
    transitions: int,
    generated_transitions: int,
    gradient_steps: int,
    update_credit: float,
    seconds: float,
    actor_transitions: int,
    current_stage_transitions: int,
    earlier_stage_transitions: int,
    updates: int,
    losses: list[float],
    q_values: list[float],
    td_errors: list[float],
) -> None:
    writer.add_scalar(
        "throughput/actor_rollout_tps", actor_transitions / seconds, transitions
    )
    writer.add_scalar(
        "throughput/learner_updates_per_sec", updates / seconds, transitions
    )
    writer.add_scalar(
        "throughput/learner_samples_per_sec",
        updates * config.batch_size / seconds,
        transitions,
    )
    classified = current_stage_transitions + earlier_stage_transitions
    writer.add_scalar(
        "distribution/rollout_current_stage_transition_fraction",
        current_stage_transitions / max(1, classified),
        transitions,
    )
    writer.add_scalar(
        "distribution/rollout_earlier_stages_transition_fraction",
        earlier_stage_transitions / max(1, classified),
        transitions,
    )
    writer.add_scalar(
        "success/rollout_current_stage", _mean(metrics.current_successes), transitions
    )
    writer.add_scalar(
        "success/rollout_earlier_stages", _mean(metrics.earlier_successes), transitions
    )
    writer.add_scalar(
        "success/rollout_after_curri",
        _mean(metrics.after_curri_successes),
        transitions,
    )
    writer.add_scalar(
        "rollout/current_stage_episodes", len(metrics.current_successes), transitions
    )
    writer.add_scalar(
        "rollout/earlier_stage_episodes", len(metrics.earlier_successes), transitions
    )
    writer.add_scalar(
        "rollout/after_curri_episodes",
        len(metrics.after_curri_successes),
        transitions,
    )
    writer.add_scalar("train/episode_return_mean", _mean(metrics.returns), transitions)
    writer.add_scalar(
        "train/ale_score_return_mean",
        _mean(metrics.ale_score_returns),
        transitions,
    )
    writer.add_scalar("train/episode_length_mean", _mean(metrics.lengths), transitions)
    writer.add_scalar("train/timeout_rate", _mean(metrics.timeouts), transitions)
    writer.add_scalar(
        "train/timeout_episode_length_mean",
        _mean(metrics.timeout_lengths),
        transitions,
    )
    writer.add_scalar(
        "train/walls_destroyed_mean", _mean(metrics.walls_destroyed), transitions
    )
    writer.add_scalar(
        "train/creatures_killed_mean", _mean(metrics.creatures_killed), transitions
    )
    writer.add_scalar(
        "train/miner_rescue_events_mean",
        _mean(metrics.miner_rescue_events),
        transitions,
    )
    writer.add_scalar(
        "train/dynamite_bonus_sticks_mean",
        _mean(metrics.dynamite_bonus_sticks),
        transitions,
    )
    writer.add_scalar(
        "train/unmapped_ale_reward_mean",
        _mean(metrics.unmapped_ale_rewards),
        transitions,
    )
    for identifier, values in sorted(metrics.task_successes.items()):
        writer.add_scalar(
            f"success/rollout_task/{identifier}", _mean(values), transitions
        )
    for identifier, values in sorted(metrics.checkpoint_successes.items()):
        writer.add_scalar(
            f"success/rollout_checkpoint/{identifier}", _mean(values), transitions
        )
    for identifier, values in sorted(metrics.checkpoint_events.items()):
        writer.add_scalar(
            f"events/rollout_checkpoint/{identifier}/walls_destroyed",
            _mean([item[0] for item in values]),
            transitions,
        )
        writer.add_scalar(
            f"events/rollout_checkpoint/{identifier}/creatures_killed",
            _mean([item[1] for item in values]),
            transitions,
        )
        writer.add_scalar(
            f"events/rollout_checkpoint/{identifier}/miner_rescued",
            _mean([item[2] for item in values]),
            transitions,
        )
    writer.add_scalar("training/target_stage", config.target_stage, transitions)
    writer.add_scalar(
        "training/decision_step_penalty",
        config.decision_step_penalty,
        transitions,
    )
    writer.add_scalar(
        "training/episode_timeout_decisions",
        config.episode_timeout_decisions,
        transitions,
    )
    writer.add_scalar(
        "training/timeout_terminal_reward",
        config.timeout_terminal_reward,
        transitions,
    )
    writer.add_scalar(
        "exploration/epsilon", epsilon_at(config, generated_transitions), transitions
    )
    writer.add_scalar("replay/total_size", len(replay), transitions)
    stage_sizes = replay.stage_sizes()
    current_size = stage_sizes.get(config.target_stage, 0)
    writer.add_scalar("replay/current_stage_size", current_size, transitions)
    earlier_size = sum(
        stage_sizes.get(stage, 0) for stage in range(1, config.target_stage)
    )
    writer.add_scalar("replay/earlier_stages_size", earlier_size, transitions)
    writer.add_scalar(
        "replay/current_stage_fraction",
        current_size / max(1, current_size + earlier_size),
        transitions,
    )
    writer.add_scalar(
        "replay/event_wall_count",
        float(sum(metrics.walls_destroyed)),
        transitions,
    )
    writer.add_scalar(
        "replay/event_creature_count",
        float(sum(metrics.creatures_killed)),
        transitions,
    )
    writer.add_scalar(
        "replay/event_miner_count",
        float(sum(metrics.miner_rescue_events)),
        transitions,
    )
    writer.add_scalar("learner/gradient_steps", gradient_steps, transitions)
    writer.add_scalar("learner/update_credit", update_credit, transitions)
    if losses:
        writer.add_scalar("learner/loss", _mean(losses), transitions)
        writer.add_scalar("learner/q_mean", _mean(q_values), transitions)
        writer.add_scalar("learner/td_error_abs", _mean(td_errors), transitions)
    writer.flush()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="atomic curriculum Stage; required unless --after-curri is used",
    )
    parser.add_argument("--run-dir", default=TrainConfig.run_dir)
    parser.add_argument("--checkpoint-root", default=TrainConfig.checkpoint_root)
    parser.add_argument("--hero-checkpoint-dir", default=TrainConfig.hero_checkpoint_dir)
    parser.add_argument("--total-transitions", type=int, default=TrainConfig.total_transitions)
    parser.add_argument("--actors", type=int, default=TrainConfig.num_actors)
    parser.add_argument("--gpu", type=int, default=TrainConfig.gpu)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--load-checkpoint")
    source.add_argument("--resume", help="same-stage checkpoint path or 'latest'")
    parser.add_argument(
        "--train-current-stage-fraction",
        type=float,
        default=TrainConfig.train_current_stage_fraction,
    )
    parser.add_argument(
        "--eval-current-stage-fraction",
        type=float,
        default=TrainConfig.eval_current_stage_fraction,
    )
    parser.add_argument("--replay-capacity", type=int, default=TrainConfig.replay_capacity)
    parser.add_argument("--learning-starts", type=int, default=TrainConfig.learning_starts)
    parser.add_argument(
        "--decision-step-penalty",
        type=float,
        default=TrainConfig.decision_step_penalty,
        help="training reward cost charged once per DQN decision (default: 0.002)",
    )
    parser.add_argument("--wall-event-reward", type=float, default=TrainConfig.wall_event_reward)
    parser.add_argument(
        "--creature-event-reward",
        type=float,
        default=TrainConfig.creature_event_reward,
    )
    parser.add_argument(
        "--miner-event-reward",
        type=float,
        default=TrainConfig.miner_event_reward,
    )
    parser.add_argument(
        "--epsilon-decay-transitions",
        type=int,
        default=TrainConfig.epsilon_decay_transitions,
    )
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--eval-episodes", type=int, default=TrainConfig.eval_episodes)
    parser.add_argument(
        "--checkpoint-interval", type=int, default=TrainConfig.checkpoint_interval
    )
    parser.add_argument("--no-save-replay", action="store_true")
    parser.add_argument(
        "--after-curri",
        action="store_true",
        help="train single-Level episodes from Level 1/2 start checkpoints",
    )
    args = parser.parse_args()
    if args.stage is None:
        if not args.after_curri:
            parser.error("--stage is required for atomic curriculum training")
        # The after-curriculum path never uses a reset stage. Keep a valid
        # internal value for shared checkpoint/config structures.
        args.stage = 1
    return TrainConfig(
        run_dir=args.run_dir,
        checkpoint_root=args.checkpoint_root,
        hero_checkpoint_dir=args.hero_checkpoint_dir,
        target_stage=args.stage,
        total_transitions=args.total_transitions,
        num_actors=args.actors,
        gpu=args.gpu,
        seed=args.seed,
        train_current_stage_fraction=args.train_current_stage_fraction,
        eval_current_stage_fraction=args.eval_current_stage_fraction,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        decision_step_penalty=args.decision_step_penalty,
        wall_event_reward=args.wall_event_reward,
        creature_event_reward=args.creature_event_reward,
        miner_event_reward=args.miner_event_reward,
        epsilon_decay_transitions=args.epsilon_decay_transitions,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_interval=args.checkpoint_interval,
        save_replay=not args.no_save_replay,
        load_checkpoint=args.load_checkpoint,
        resume=args.resume,
        after_curri=args.after_curri,
    )


def run_training(config: TrainConfig) -> None:
    config.validate()
    run_path = config.run_path
    checkpoint_dir = config.checkpoint_path
    resume_path = (
        resolve_checkpoint(config.resume, checkpoint_dir) if config.resume else None
    )
    if resume_path is None:
        if run_path.exists() and any(run_path.iterdir()):
            raise ValueError(f"run directory is not empty: {run_path}")
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise ValueError(f"checkpoint directory is not empty: {checkpoint_dir}")

    # Validate the frozen curriculum before creating run/checkpoint output.
    probe_env = make_training_env(config, config.target_stage)
    action_count = int(probe_env.action_space.n)
    curriculum_identity = probe_env.curriculum_identity
    probe_env.close()
    if config.after_curri:
        if curriculum_identity is None:
            raise ValueError(
                "after-curri training requires a frozen curriculum manifest "
                "with Level 1 and Level 2 start checkpoints"
            )
        curriculum_identity = {
            "mode": "after_curri",
            "curriculum_version": curriculum_identity["version"],
            "curriculum_sha256": curriculum_identity["manifest_sha256"],
            "start_levels": (1, 2),
            "action_repeat": config.action_repeat,
            "sticky_action_probability": config.sticky_action_probability,
        }
    elif curriculum_identity is None:
        raise ValueError("training requires a frozen curriculum manifest")

    session = f"{datetime.now().astimezone():%Y%m%d-%H%M%S}_pid-{os.getpid()}"
    tensorboard_dir = run_path / "tensorboard" / session
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(config.as_dict(), run_path / "config.json")

    torch.cuda.set_device(config.gpu)
    device = torch.device(f"cuda:{config.gpu}")
    torch.backends.cudnn.benchmark = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    write_json_atomic(
        curriculum_identity, run_path / "curriculum_identity.json"
    )
    online_model = DuelingDQN(action_count, config.frame_stack).to(device)
    target_model = DuelingDQN(action_count, config.frame_stack).to(device)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(
        online_model.parameters(), lr=config.learning_rate, eps=config.adam_eps
    )
    scaler = torch.amp.GradScaler("cuda")
    replay = ReplayBuffer(
        capacity=config.replay_capacity,
        seed=config.seed + 100,
    )
    metrics = TrainingMetrics(config)
    transitions = 0
    gradient_steps = 0
    update_credit = 0.0
    next_checkpoint_transition = config.checkpoint_interval
    next_eval_transition = config.eval_interval
    eval_in_flight = False

    if config.load_checkpoint is not None:
        source = torch.load(
            Path(config.load_checkpoint).resolve(),
            map_location=device,
            weights_only=False,
        )
        state = source.get("online_model", source)
        online_model.load_state_dict(state)
        target_model.load_state_dict(state)
        print(f"Initialized weights from {config.load_checkpoint}", flush=True)

    if resume_path is not None:
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume(config, payload, curriculum_identity)
        online_model.load_state_dict(payload["online_model"])
        target_model.load_state_dict(payload["target_model"])
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        if payload.get("replay") is not None:
            replay.load_state_dict(payload["replay"])
            update_credit = float(payload["update_credit"])
        metrics.load_state_dict(payload["metrics"])
        transitions = int(payload["transitions"])
        gradient_steps = int(payload["gradient_steps"])
        next_checkpoint_transition = int(payload["next_checkpoint_transition"])
        next_eval_transition = int(payload["next_eval_transition"])
        if bool(payload["eval_in_flight"]):
            next_eval_transition = min(next_eval_transition, transitions)
        eval_in_flight = False
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), config.gpu)
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
        print(f"Resumed {resume_path} at transition {transitions}", flush=True)

    shared_model = DuelingDQN(action_count, config.frame_stack).cpu()
    shared_model.load_state_dict(_cpu_state_dict(online_model))
    shared_model.share_memory()
    context = mp.get_context("spawn")
    weight_lock = context.Lock()
    weight_version = context.Value("q", 1)
    global_transition_count = context.Value("q", transitions)
    stop_event = context.Event()
    transition_queue = context.Queue(maxsize=config.queue_size)
    metrics_queue = context.Queue()
    eval_jobs = context.Queue()
    eval_results = context.Queue()

    evaluator = context.Process(
        target=stage_evaluator_process,
        name=f"hero-stage-{config.target_stage:02d}-evaluator",
        args=(
            config,
            action_count,
            shared_model,
            weight_lock,
            eval_jobs,
            eval_results,
            stop_event,
        ),
    )
    evaluator.start()
    actors = []
    for actor_id in range(config.num_actors):
        process = context.Process(
            target=actor_process,
            name=f"hero-stage-{config.target_stage:02d}-actor-{actor_id}",
            args=(
                actor_id,
                config,
                shared_model,
                weight_lock,
                weight_version,
                global_transition_count,
                transition_queue,
                metrics_queue,
                stop_event,
            ),
        )
        process.start()
        actors.append(process)

    writer = SummaryWriter(
        log_dir=str(tensorboard_dir),
        purge_step=(transitions + 1) if transitions else None,
        filename_suffix=f".{session}",
    )
    last_log_time = time.monotonic()
    interval_actor_transitions = 0
    interval_current_stage_transitions = 0
    interval_earlier_stage_transitions = 0
    interval_updates = 0
    losses: list[float] = []
    q_values: list[float] = []
    td_errors: list[float] = []
    last_checkpoint_step = -1
    normal_completion = False

    try:
        while transitions < config.total_transitions or (
            update_credit >= 1 and len(replay) >= config.learning_starts
        ):
            drain_episode_metrics(
                metrics_queue, metrics, config.target_stage
            )
            for result in drain_eval_results(eval_results):
                log_eval_result(writer, result, config.target_stage)
                eval_in_flight = False
                next_eval_transition = max(
                    transitions, result.checkpoint_step + config.eval_interval
                )

            max_ingest = min(
                256,
                config.total_transitions - transitions,
                max(0, next_checkpoint_transition - transitions),
            )
            ingested = 0
            while ingested < max_ingest:
                try:
                    transition = transition_queue.get(
                        timeout=0.1 if ingested == 0 else 0.0
                    )
                except queue.Empty:
                    break
                replay.add(transition)
                transitions += 1
                ingested += 1
                update_credit += config.update_ratio
                interval_actor_transitions += 1
                if transition.stage == config.target_stage:
                    interval_current_stage_transitions += 1
                elif transition.stage < config.target_stage:
                    interval_earlier_stage_transitions += 1

            updates_this_cycle = 0
            while (
                update_credit >= 1
                and len(replay) >= config.learning_starts
                and updates_this_cycle < config.max_updates_per_cycle
            ):
                packed = replay.sample(config.batch_size)
                result = learner_update(
                    online_model,
                    target_model,
                    optimizer,
                    scaler,
                    decode_batch(packed, device),
                    config.gamma,
                    config.max_grad_norm,
                )
                update_credit -= 1
                gradient_steps += 1
                updates_this_cycle += 1
                interval_updates += 1
                losses.append(result[0])
                q_values.append(result[1])
                td_errors.append(result[2])
                if gradient_steps % config.target_update_interval == 0:
                    target_model.load_state_dict(online_model.state_dict())
                if gradient_steps % config.publish_interval == 0:
                    publish_weights(
                        online_model, shared_model, weight_lock, weight_version
                    )

            if not eval_in_flight and transitions >= next_eval_transition:
                publish_weights(
                    online_model, shared_model, weight_lock, weight_version
                )
                eval_jobs.put(transitions)
                eval_in_flight = True

            if transitions == next_checkpoint_transition:
                next_checkpoint_transition += config.checkpoint_interval
                path = save_checkpoint(
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                    online_model=online_model,
                    target_model=target_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    replay=replay,
                    metrics=metrics,
                    transitions=transitions,
                    gradient_steps=gradient_steps,
                    update_credit=update_credit,
                    next_checkpoint_transition=next_checkpoint_transition,
                    next_eval_transition=next_eval_transition,
                    eval_in_flight=eval_in_flight,
                    generated_transitions=int(global_transition_count.value),
                    curriculum_identity=curriculum_identity,
                )
                last_checkpoint_step = transitions
                print(f"Saved {path}", flush=True)

            now = time.monotonic()
            if now - last_log_time >= config.log_interval_seconds:
                seconds = now - last_log_time
                generated = int(global_transition_count.value)
                log_training(
                    writer=writer,
                    config=config,
                    metrics=metrics,
                    replay=replay,
                    transitions=transitions,
                    generated_transitions=generated,
                    gradient_steps=gradient_steps,
                    update_credit=update_credit,
                    seconds=seconds,
                    actor_transitions=interval_actor_transitions,
                    current_stage_transitions=(
                        interval_current_stage_transitions
                    ),
                    earlier_stage_transitions=(
                        interval_earlier_stage_transitions
                    ),
                    updates=interval_updates,
                    losses=losses,
                    q_values=q_values,
                    td_errors=td_errors,
                )
                print(
                    f"stage={config.target_stage} steps={transitions:,} "
                    f"updates={gradient_steps:,} "
                    f"epsilon={epsilon_at(config, generated):.3f} "
                    f"actor_tps={interval_actor_transitions / seconds:.1f} "
                    f"learner_ups={interval_updates / seconds:.1f}",
                    flush=True,
                )
                interval_actor_transitions = 0
                interval_current_stage_transitions = 0
                interval_earlier_stage_transitions = 0
                interval_updates = 0
                losses.clear()
                q_values.clear()
                td_errors.clear()
                last_log_time = now

            if ingested == 0 and updates_this_cycle == 0:
                if stop_event.is_set():
                    drain_episode_metrics(
                        metrics_queue, metrics, config.target_stage
                    )
                    raise RuntimeError("worker requested shutdown")
                time.sleep(0.01)

        publish_weights(online_model, shared_model, weight_lock, weight_version)
        for process in actors:
            process.join(timeout=5)
        if transitions != last_checkpoint_step:
            path = save_checkpoint(
                config=config,
                checkpoint_dir=checkpoint_dir,
                online_model=online_model,
                target_model=target_model,
                optimizer=optimizer,
                scaler=scaler,
                replay=replay,
                metrics=metrics,
                transitions=transitions,
                gradient_steps=gradient_steps,
                update_credit=update_credit,
                next_checkpoint_transition=next_checkpoint_transition,
                next_eval_transition=next_eval_transition,
                eval_in_flight=eval_in_flight,
                generated_transitions=int(global_transition_count.value),
                curriculum_identity=curriculum_identity,
            )
            print(f"Saved final {path}", flush=True)
        if not eval_in_flight:
            eval_jobs.put(transitions)
            eval_in_flight = True
        normal_completion = True
    except KeyboardInterrupt:
        print("Interrupted; saving emergency checkpoint", flush=True)
        if transitions != last_checkpoint_step:
            path = save_checkpoint(
                config=config,
                checkpoint_dir=checkpoint_dir,
                online_model=online_model,
                target_model=target_model,
                optimizer=optimizer,
                scaler=scaler,
                replay=replay,
                metrics=metrics,
                transitions=transitions,
                gradient_steps=gradient_steps,
                update_credit=update_credit,
                next_checkpoint_transition=next_checkpoint_transition,
                next_eval_transition=next_eval_transition,
                eval_in_flight=eval_in_flight,
                generated_transitions=int(global_transition_count.value),
                curriculum_identity=curriculum_identity,
            )
            print(f"Saved emergency {path}", flush=True)
    finally:
        if normal_completion:
            eval_jobs.put(None)
            while evaluator.is_alive():
                for result in drain_eval_results(eval_results):
                    log_eval_result(writer, result, config.target_stage)
                evaluator.join(timeout=0.5)
            for result in drain_eval_results(eval_results):
                log_eval_result(writer, result, config.target_stage)
        stop_event.set()
        if not normal_completion and evaluator.is_alive():
            eval_jobs.put(None)
        for process in actors:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        evaluator.join(timeout=5)
        if evaluator.is_alive():
            evaluator.terminate()
            evaluator.join()
        writer.close()
        transition_queue.close()
        metrics_queue.close()
        eval_jobs.close()
        eval_results.close()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
