"""Single-GPU learner, checkpoint owner, and sole TensorBoard writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from DQN.config import DQNConfig
from DQN.messages import (
    ActorReport,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorStop,
    ParameterUpdate,
    ProcessError,
    TransitionChunk,
)
from DQN.network import DuelingQNetwork
from DQN.replay import PrioritizedReplayBuffer
from DQN.utils import (
    atomic_torch_save,
    increment_counter,
    linear_beta,
    linear_epsilon,
    put_latest,
    serialize_state_dict,
)


@dataclass(slots=True)
class IntervalMetrics:
    actor_transitions: int = 0
    actor_collection_seconds: float = 0.0
    actor_queue_wait_seconds: float = 0.0
    observation_diversity: list[float] = field(default_factory=list)
    policy_lags: list[int] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_raw_scores: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    q_values: list[float] = field(default_factory=list)
    td_errors: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)

    def add_actor_report(self, report: ActorReport, current_version: int) -> None:
        self.actor_transitions += report.transitions
        self.actor_collection_seconds += report.collection_seconds
        self.actor_queue_wait_seconds += report.queue_wait_seconds
        self.observation_diversity.append(report.unique_observation_fraction)
        self.policy_lags.append(max(current_version - report.policy_version, 0))
        self.episode_lengths.extend(report.episode_lengths)
        self.episode_returns.extend(report.episode_returns)
        self.episode_raw_scores.extend(report.episode_raw_scores)

    def add_train_metrics(self, metrics: dict[str, float]) -> None:
        self.losses.append(metrics["loss"])
        self.q_values.append(metrics["q_mean"])
        self.td_errors.append(metrics["td_error_abs_mean"])
        self.gradient_norms.append(metrics["gradient_norm"])


def train_batch(
    online_network: DuelingQNetwork,
    target_network: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: PrioritizedReplayBuffer,
    config: DQNConfig,
    device: torch.device,
    beta: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    sample = replay.sample(config.learner_batch_size, beta, rng)
    observations = torch.from_numpy(sample.observations).to(device)
    actions = torch.from_numpy(sample.actions).to(device)
    rewards = torch.from_numpy(sample.rewards).to(device)
    next_observations = torch.from_numpy(sample.next_observations).to(device)
    terminated = torch.from_numpy(sample.terminated).to(device)
    weights = torch.from_numpy(sample.weights).to(device)

    q_values = online_network(observations)
    chosen_q_values = q_values.gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_actions = online_network(next_observations).argmax(dim=1)
        next_target_q = target_network(next_observations).gather(
            1, next_actions[:, None]
        ).squeeze(1)
        targets = rewards + config.gamma * (~terminated).float() * next_target_q

    td_errors = targets - chosen_q_values
    elementwise_loss = F.smooth_l1_loss(
        chosen_q_values, targets, reduction="none"
    )
    loss = torch.mean(weights * elementwise_loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        online_network.parameters(), config.gradient_clip_norm
    )
    optimizer.step()

    absolute_td_errors = np.maximum(
        np.abs(td_errors.detach().cpu().numpy()), config.priority_epsilon
    )
    replay.update_priorities(sample.indices, absolute_td_errors)
    return {
        "loss": float(loss.detach().cpu()),
        "q_mean": float(chosen_q_values.detach().mean().cpu()),
        "td_error_abs_mean": float(np.mean(absolute_td_errors)),
        "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
    }


def drain_actor_reports(metrics_queue, interval: IntervalMetrics, version: int) -> None:
    while True:
        try:
            report = metrics_queue.get_nowait()
        except Empty:
            return
        if not isinstance(report, ActorReport):
            raise TypeError(f"Unexpected metrics message: {type(report)}")
        interval.add_actor_report(report, version)


def mean_or_none(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def log_interval(
    writer: SummaryWriter,
    interval: IntervalMetrics,
    *,
    global_transitions: int,
    learner_updates: int,
    replay_size: int,
    policy_version: int,
    beta: float,
    config: DQNConfig,
    elapsed_seconds: float,
    global_delta: int,
) -> None:
    step = global_transitions
    writer.add_scalar(
        "rollout/transitions_per_second",
        interval.actor_transitions / elapsed_seconds,
        step,
    )
    writer.add_scalar(
        "learner/consumed_transitions_per_second",
        global_delta / elapsed_seconds,
        step,
    )
    writer.add_scalar("global/transitions", global_transitions, step)
    writer.add_scalar("learner/updates", learner_updates, step)
    writer.add_scalar("replay/size", replay_size, step)
    writer.add_scalar("replay/beta", beta, step)
    writer.add_scalar("train/epsilon", linear_epsilon(global_transitions, config), step)
    writer.add_scalar("train/policy_version", policy_version, step)

    if interval.actor_collection_seconds > 0:
        writer.add_scalar(
            "rollout/actor_collection_transitions_per_second",
            interval.actor_transitions / interval.actor_collection_seconds,
            step,
        )
    denominator = interval.actor_collection_seconds + interval.actor_queue_wait_seconds
    if denominator > 0:
        writer.add_scalar(
            "rollout/queue_wait_fraction",
            interval.actor_queue_wait_seconds / denominator,
            step,
        )

    scalar_groups = {
        "rollout/episode_length_mean": interval.episode_lengths,
        "rollout/episode_return_mean": interval.episode_returns,
        "rollout/raw_score_mean": interval.episode_raw_scores,
        "rollout/observation_unique_fraction": interval.observation_diversity,
        "rollout/policy_lag_mean": interval.policy_lags,
        "train/loss": interval.losses,
        "train/q_mean": interval.q_values,
        "train/td_error_abs_mean": interval.td_errors,
        "train/gradient_norm": interval.gradient_norms,
    }
    for name, values in scalar_groups.items():
        value = mean_or_none(values)
        if value is not None:
            writer.add_scalar(name, value, step)
    writer.add_scalar("rollout/completed_episodes", len(interval.episode_lengths), step)
    writer.flush()


def log_evaluation(writer: SummaryWriter, result: EvaluationResult) -> None:
    step = result.checkpoint_transition
    lengths = np.asarray(result.episode_lengths, dtype=np.float64)
    returns = np.asarray(result.episode_returns, dtype=np.float64)
    raw_scores = np.asarray(result.episode_raw_scores, dtype=np.float64)
    writer.add_scalar("eval/episode_length_mean", lengths.mean(), step)
    writer.add_scalar("eval/episode_return_mean", returns.mean(), step)
    writer.add_scalar("eval/raw_score_mean", raw_scores.mean(), step)
    writer.add_scalar("eval/raw_score_median", np.median(raw_scores), step)
    writer.add_scalar("eval/raw_score_p25", np.percentile(raw_scores, 25), step)
    writer.add_scalar("eval/raw_score_p75", np.percentile(raw_scores, 75), step)
    writer.add_scalar("eval/capped_episode_count", result.capped_episodes, step)
    writer.add_scalar(
        "eval/capped_episode_fraction",
        result.capped_episodes / len(lengths),
        step,
    )
    writer.add_scalar(
        "eval/episodes_per_second", len(lengths) / result.elapsed_seconds, step
    )
    writer.flush()


def save_checkpoint(
    path: Path,
    online_network: DuelingQNetwork,
    target_network: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    config: DQNConfig,
    global_transitions: int,
    learner_updates: int,
    policy_version: int,
) -> None:
    atomic_torch_save(
        {
            "online_state_dict": {
                key: value.detach().cpu()
                for key, value in online_network.state_dict().items()
            },
            "target_state_dict": {
                key: value.detach().cpu()
                for key, value in target_network.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "global_transitions": global_transitions,
            "learner_updates": learner_updates,
            "policy_version": policy_version,
            "config": asdict(config),
        },
        path,
    )


def learner_process(
    config: DQNConfig,
    rollout_queue,
    metrics_queue,
    parameter_queues,
    evaluation_request_queue,
    evaluation_result_queue,
    global_transition_counter,
    stop_event,
    error_queue,
) -> None:
    process_name = "learner"
    writer: SummaryWriter | None = None
    try:
        for parameter_queue in parameter_queues:
            parameter_queue.cancel_join_thread()
        if not torch.cuda.is_available() and config.learner_device.startswith("cuda"):
            raise RuntimeError("CUDA learner requested but torch.cuda.is_available() is false")
        device = torch.device(config.learner_device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.manual_seed(config.seed)
        np_rng = np.random.default_rng(config.seed)

        online_network = DuelingQNetwork(
            config.observation_shape, config.action_count
        ).to(device)
        target_network = DuelingQNetwork(
            config.observation_shape, config.action_count
        ).to(device)
        target_network.load_state_dict(online_network.state_dict())
        target_network.eval()
        optimizer = torch.optim.Adam(
            online_network.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
        )
        global_transitions = 0
        learner_updates = 0
        policy_version = 0

        if config.resume_checkpoint is not None:
            checkpoint = torch.load(
                config.resume_checkpoint, map_location=device, weights_only=False
            )
            online_network.load_state_dict(checkpoint["online_state_dict"])
            target_network.load_state_dict(checkpoint["target_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            global_transitions = int(
                checkpoint.get(
                    "global_transitions", checkpoint.get("learner_transitions", 0)
                )
            )
            learner_updates = int(checkpoint["learner_updates"])
            policy_version = int(checkpoint["policy_version"])
            with global_transition_counter.get_lock():
                global_transition_counter.value = global_transitions

        replay = PrioritizedReplayBuffer(
            config.replay_capacity,
            config.observation_shape,
            alpha=config.prioritized_replay_alpha,
            priority_epsilon=config.priority_epsilon,
        )
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
        run_dir = config.runs_dir / run_name
        checkpoint_dir = config.checkpoints_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        writer = SummaryWriter(log_dir=str(run_dir), max_queue=10, flush_secs=120)
        writer.add_text(
            "config/json",
            json.dumps(asdict(config), default=str, indent=2, sort_keys=True),
            0,
        )
        print(f"TensorBoard run: {run_dir}", flush=True)
        print(f"Checkpoints: {checkpoint_dir}", flush=True)

        initial_message = ParameterUpdate(
            version=policy_version,
            global_transitions=global_transitions,
            state_dict_bytes=serialize_state_dict(online_network),
        )
        for parameter_queue in parameter_queues:
            put_latest(parameter_queue, initial_message)

        next_tensorboard = (
            global_transitions // config.tensorboard_interval_transitions + 1
        ) * config.tensorboard_interval_transitions
        next_checkpoint = (
            global_transitions // config.checkpoint_interval_transitions + 1
        ) * config.checkpoint_interval_transitions
        interval = IntervalMetrics()
        interval_started = time.monotonic()
        last_global = global_transitions
        update_budget = 0.0
        pending_evaluations: set[int] = set()
        normal_completion = False

        while global_transitions < config.total_transitions and not stop_event.is_set():
            try:
                chunk = rollout_queue.get(timeout=config.queue_retry_timeout_seconds)
            except Empty:
                drain_actor_reports(metrics_queue, interval, policy_version)
                while True:
                    try:
                        result = evaluation_result_queue.get_nowait()
                    except Empty:
                        break
                    log_evaluation(writer, result)
                    pending_evaluations.discard(result.checkpoint_transition)
                continue
            if not isinstance(chunk, TransitionChunk):
                raise TypeError(f"Unexpected rollout message: {type(chunk)}")

            replay.add(chunk)
            global_transitions = increment_counter(
                global_transition_counter, len(chunk)
            )
            if len(replay) >= config.learning_starts:
                update_budget += len(chunk) * config.updates_per_transition
                updates_to_run = int(update_budget)
                update_budget -= updates_to_run
            else:
                updates_to_run = 0

            for _ in range(updates_to_run):
                beta = linear_beta(global_transitions, config)
                train_metrics = train_batch(
                    online_network,
                    target_network,
                    optimizer,
                    replay,
                    config,
                    device,
                    beta,
                    np_rng,
                )
                learner_updates += 1
                interval.add_train_metrics(train_metrics)
                if learner_updates % config.target_sync_interval_updates == 0:
                    target_network.load_state_dict(online_network.state_dict())

            if updates_to_run > 0:
                policy_version += 1
                parameter_message = ParameterUpdate(
                    version=policy_version,
                    global_transitions=global_transitions,
                    state_dict_bytes=serialize_state_dict(online_network),
                )
                for parameter_queue in parameter_queues:
                    put_latest(parameter_queue, parameter_message)

            drain_actor_reports(metrics_queue, interval, policy_version)
            while True:
                try:
                    result = evaluation_result_queue.get_nowait()
                except Empty:
                    break
                log_evaluation(writer, result)
                pending_evaluations.discard(result.checkpoint_transition)

            if global_transitions >= next_tensorboard:
                now = time.monotonic()
                log_interval(
                    writer,
                    interval,
                    global_transitions=global_transitions,
                    learner_updates=learner_updates,
                    replay_size=len(replay),
                    policy_version=policy_version,
                    beta=linear_beta(global_transitions, config),
                    config=config,
                    elapsed_seconds=max(now - interval_started, 1.0e-9),
                    global_delta=global_transitions - last_global,
                )
                interval = IntervalMetrics()
                interval_started = now
                last_global = global_transitions
                while next_tensorboard <= global_transitions:
                    next_tensorboard += config.tensorboard_interval_transitions

            if global_transitions >= next_checkpoint:
                checkpoint_path = checkpoint_dir / (
                    f"checkpoint_step_{next_checkpoint:012d}.pt"
                )
                save_checkpoint(
                    checkpoint_path,
                    online_network,
                    target_network,
                    optimizer,
                    config,
                    global_transitions,
                    learner_updates,
                    policy_version,
                )
                if config.evaluation_enabled:
                    evaluation_request_queue.put(
                        EvaluationRequest(
                            checkpoint_path=str(checkpoint_path),
                            checkpoint_transition=next_checkpoint,
                        )
                    )
                    pending_evaluations.add(next_checkpoint)
                writer.flush()
                while next_checkpoint <= global_transitions:
                    next_checkpoint += config.checkpoint_interval_transitions

        normal_completion = global_transitions >= config.total_transitions
        stop_event.set()
        evaluation_request_queue.put(EvaluatorStop())

        if normal_completion and pending_evaluations:
            deadline = time.monotonic() + config.evaluation_shutdown_timeout_seconds
            while pending_evaluations and time.monotonic() < deadline:
                try:
                    result = evaluation_result_queue.get(timeout=1.0)
                except Empty:
                    continue
                log_evaluation(writer, result)
                pending_evaluations.discard(result.checkpoint_transition)
            if pending_evaluations:
                raise TimeoutError(
                    f"Timed out waiting for evaluations: {sorted(pending_evaluations)}"
                )
        writer.flush()
    except BaseException:
        error_queue.put(
            ProcessError(process_name=process_name, traceback=traceback.format_exc())
        )
        stop_event.set()
        raise
    finally:
        if writer is not None:
            writer.close()
