"""Single-device R2D2 learner, checkpoint writer, and metric owner."""

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

from _R2D2.config import R2D2Config
from _R2D2.messages import (
    ActorReport,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorStop,
    ParameterUpdate,
    ProcessError,
    SequenceChunk,
)
from _R2D2.network import RecurrentDuelingQNetwork
from _R2D2.replay import SequenceReplay
from _R2D2.sequence import mixed_priority
from _R2D2.utils import (
    actor_epsilon,
    atomic_torch_save,
    increment_counter,
    linear_beta,
    put_latest,
    serialize_state_dict,
)


@dataclass(slots=True)
class IntervalMetrics:
    transitions: int = 0
    collection_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    unique_observations: list[float] = field(default_factory=list)
    policy_lags: list[int] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    episode_raw_scores: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    q_values: list[float] = field(default_factory=list)
    td_errors: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    is_weight_mins: list[float] = field(default_factory=list)
    is_weight_means: list[float] = field(default_factory=list)
    is_weight_maxes: list[float] = field(default_factory=list)

    def add_report(self, report: ActorReport, version: int) -> None:
        self.transitions += report.transitions
        self.collection_seconds += report.collection_seconds
        self.queue_wait_seconds += report.queue_wait_seconds
        self.unique_observations.append(report.unique_observation_fraction)
        self.policy_lags.append(max(0, version - report.policy_version))
        self.episode_lengths.extend(report.episode_lengths)
        self.episode_returns.extend(report.episode_returns)
        self.episode_raw_scores.extend(report.episode_raw_scores)


def _mean(values: list[float] | list[int]) -> float | None:
    return None if not values else float(np.mean(values))


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors after loading a CPU checkpoint on CUDA."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def transformed_double_q_targets(
    n_step_rewards: torch.Tensor,
    discounts: torch.Tensor,
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
) -> torch.Tensor:
    """Build value-rescaled n-step Double-Q targets.

    ``discounts`` already encodes both ``gamma ** n`` and episode boundaries.
    A zero discount therefore suppresses bootstrapping for terminal and
    incomplete tail transitions.
    """
    if online_next_q.shape != target_next_q.shape:
        raise ValueError("online and target next-Q tensors must have the same shape")
    if online_next_q.ndim != 2:
        raise ValueError("next-Q tensors must have shape (N, actions)")
    if n_step_rewards.shape != discounts.shape:
        raise ValueError("n-step rewards and discounts must have the same shape")
    if n_step_rewards.numel() != online_next_q.shape[0]:
        raise ValueError("one next-Q row is required for each target")
    next_actions = online_next_q.argmax(dim=1)
    next_values = target_next_q.gather(1, next_actions[:, None]).squeeze(1)
    unscaled_next_values = RecurrentDuelingQNetwork.inverse_value_rescale(next_values)
    return RecurrentDuelingQNetwork.value_rescale(
        n_step_rewards + discounts * unscaled_next_values
    )


def train_batch(
    online: RecurrentDuelingQNetwork,
    target: RecurrentDuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    replay: SequenceReplay,
    config: R2D2Config,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    sample = replay.sample(config.learner_batch_size, rng)
    sequences = sample.sequences
    batch_size = len(sequences)
    max_time = max(sequence.previous_actions.shape[0] for sequence in sequences)
    channels, height, width = config.observation_shape
    observations = torch.zeros(batch_size, max_time, channels, height, width, device=device)
    previous_actions = torch.zeros(batch_size, max_time, config.action_count, device=device)
    previous_rewards = torch.zeros(batch_size, max_time, device=device)
    hidden_h = torch.zeros(1, batch_size, config.hidden_size, device=device)
    hidden_c = torch.zeros_like(hidden_h)
    for index, sequence in enumerate(sequences):
        packed = sequence.unpack_observations(config.actor_env.frame_stack)
        length = packed.shape[0]
        observations[index, :length] = torch.from_numpy(np.array(packed, copy=True)).to(device)
        previous_actions[index, :length] = torch.from_numpy(sequence.previous_actions).to(device)
        previous_rewards[index, :length] = torch.from_numpy(sequence.previous_rewards).to(device)
        hidden_h[0, index] = torch.from_numpy(sequence.initial_hidden[0]).reshape(-1).to(device)
        hidden_c[0, index] = torch.from_numpy(sequence.initial_hidden[1]).reshape(-1).to(device)

    burn_in = torch.as_tensor(
        [sequence.burn_in_steps for sequence in sequences], device=device
    )
    q_online_all, _ = online.unroll(
        observations,
        previous_actions,
        previous_rewards,
        (hidden_h, hidden_c),
        burn_in_steps=burn_in,
    )
    with torch.no_grad():
        q_target_all, _ = target.unroll(observations, previous_actions, previous_rewards, (hidden_h, hidden_c))

    chosen: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    for index, sequence in enumerate(sequences):
        begin = sequence.burn_in_steps
        end = begin + sequence.learning_steps
        current_q = q_online_all[index, begin:end]
        actions = torch.from_numpy(sequence.actions.astype(np.int64, copy=False)).to(device)
        chosen.append(current_q.gather(1, actions[:, None]).squeeze(1))
        target_indices = torch.arange(sequence.learning_steps, device=device) + begin + sequence.forward_steps
        online_next = q_online_all[index].index_select(0, target_indices)
        target_next = q_target_all[index].index_select(0, target_indices)
        n_step_reward = torch.from_numpy(sequence.n_step_rewards).to(device)
        discounts = torch.from_numpy(sequence.discounts).to(device)
        target_values.append(
            transformed_double_q_targets(
                n_step_reward, discounts, online_next, target_next
            )
        )

    chosen_flat = torch.cat(chosen)
    targets_flat = torch.cat(target_values).detach()
    repeated_weights = torch.from_numpy(
        np.concatenate([
            np.full(sequence.learning_steps, sample.weights[index], dtype=np.float32)
            for index, sequence in enumerate(sequences)
        ])
    ).to(device)
    td_errors = targets_flat - chosen_flat
    # The reference implementation uses an unreduced MSE objective.
    loss = (repeated_weights * F.mse_loss(chosen_flat, targets_flat, reduction="none")).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(online.parameters(), config.gradient_clip_norm)
    optimizer.step()

    start = 0
    priorities = []
    abs_errors = td_errors.detach().abs().cpu().numpy()
    for sequence in sequences:
        part = abs_errors[start : start + sequence.learning_steps]
        priorities.append(max(mixed_priority(part, config.priority_mix), config.priority_epsilon))
        start += sequence.learning_steps
    replay.update_priorities(sample.indices, np.asarray(priorities), sample.generations)
    return {
        "loss": float(loss.detach().cpu()),
        "q_mean": float(chosen_flat.detach().mean().cpu()),
        "td_error_abs_mean": float(abs_errors.mean()),
        "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
        "is_weight_min": float(sample.weights.min()),
        "is_weight_mean": float(sample.weights.mean()),
        "is_weight_max": float(sample.weights.max()),
    }


def _drain_reports(metrics_queue, interval: IntervalMetrics, version: int) -> None:
    while True:
        try:
            report = metrics_queue.get_nowait()
        except Empty:
            return
        if not isinstance(report, ActorReport):
            raise TypeError(f"unexpected metrics message: {type(report)}")
        interval.add_report(report, version)


def _log_interval(
    writer: SummaryWriter,
    metrics: IntervalMetrics,
    *,
    transitions: int,
    updates: int,
    replay_size: int,
    replay_learning_transitions: int,
    version: int,
    elapsed: float,
    delta: int,
    config: R2D2Config,
) -> None:
    step = transitions
    elapsed = max(elapsed, 1.0e-9)
    scalars = {
        "rollout/transitions_per_second": metrics.transitions / elapsed,
        "learner/consumed_transitions_per_second": delta / elapsed,
        "global/transitions": transitions,
        "learner/updates": updates,
        "replay/size": replay_size,
        "replay/learning_transitions": replay_learning_transitions,
        "replay/beta": linear_beta(transitions, config),
        "train/epsilon": actor_epsilon(0, config),
        "train/policy_version": version,
        "rollout/completed_episodes": len(metrics.episode_lengths),
        "train/epsilon_max": max(
            actor_epsilon(actor_id, config) for actor_id in range(config.num_actors)
        ),
        "train/epsilon_min": min(
            actor_epsilon(actor_id, config) for actor_id in range(config.num_actors)
        ),
    }
    if metrics.collection_seconds > 0:
        scalars["rollout/actor_collection_transitions_per_second"] = (
            metrics.transitions / metrics.collection_seconds
        )
    rollout_time = metrics.collection_seconds + metrics.queue_wait_seconds
    if rollout_time > 0:
        scalars["rollout/queue_wait_fraction"] = (
            metrics.queue_wait_seconds / rollout_time
        )
    groups = {
        "rollout/episode_length_mean": metrics.episode_lengths,
        "rollout/episode_return_mean": metrics.episode_returns,
        "rollout/raw_score_mean": metrics.episode_raw_scores,
        "rollout/observation_unique_fraction": metrics.unique_observations,
        "rollout/policy_lag_mean": metrics.policy_lags,
        "train/loss": metrics.losses,
        "train/q_mean": metrics.q_values,
        "train/td_error_abs_mean": metrics.td_errors,
        "train/gradient_norm": metrics.gradient_norms,
        "replay/is_weight_min": metrics.is_weight_mins,
        "replay/is_weight_mean": metrics.is_weight_means,
        "replay/is_weight_max": metrics.is_weight_maxes,
    }
    for name, values in groups.items():
        value = _mean(values)
        # Keep the complete metric schema visible from the first interval.
        # NaN accurately means that no episode/update completed in that window.
        scalars[name] = float("nan") if value is None else value
    for name, value in scalars.items():
        writer.add_scalar(name, value, step)
    writer.flush()


def _log_evaluation(writer: SummaryWriter, result: EvaluationResult) -> None:
    lengths = np.asarray(result.episode_lengths, dtype=np.float64)
    returns = np.asarray(result.episode_returns, dtype=np.float64)
    scores = np.asarray(result.episode_raw_scores, dtype=np.float64)
    step = result.checkpoint_transition
    writer.add_scalar("eval/episode_length_mean", lengths.mean(), step)
    writer.add_scalar("eval/episode_return_mean", returns.mean(), step)
    writer.add_scalar("eval/raw_score_mean", scores.mean(), step)
    writer.add_scalar("eval/raw_score_median", np.median(scores), step)
    writer.add_scalar("eval/raw_score_p25", np.percentile(scores, 25), step)
    writer.add_scalar("eval/raw_score_p75", np.percentile(scores, 75), step)
    writer.add_scalar("eval/capped_episode_count", result.capped_episodes, step)
    writer.add_scalar("eval/capped_episode_fraction", result.capped_episodes / len(scores), step)
    writer.add_scalar("eval/episodes_per_second", len(scores) / max(result.elapsed_seconds, 1.0e-9), step)
    writer.flush()


def save_checkpoint(
    path: Path,
    online: RecurrentDuelingQNetwork,
    target: RecurrentDuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    config: R2D2Config,
    transitions: int,
    updates: int,
    version: int,
) -> None:
    atomic_torch_save(
        {
            "online_state_dict": {k: v.detach().cpu() for k, v in online.state_dict().items()},
            "target_state_dict": {k: v.detach().cpu() for k, v in target.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "global_transitions": transitions,
            "learner_updates": updates,
            "policy_version": version,
            "config": asdict(config),
        },
        path,
    )


def learner_process(
    config: R2D2Config,
    rollout_queue,
    metrics_queue,
    parameter_queues,
    evaluation_request_queue,
    evaluation_result_queue,
    global_transition_counter,
    stop_event,
    error_queue,
) -> None:
    writer: SummaryWriter | None = None
    try:
        torch.set_num_threads(1)
        device = torch.device(config.learner_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA learner requested but CUDA is unavailable")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)
        online = RecurrentDuelingQNetwork(config.observation_shape, config.action_count, config.hidden_size).to(device)
        target = RecurrentDuelingQNetwork(config.observation_shape, config.action_count, config.hidden_size).to(device)
        target.load_state_dict(online.state_dict())
        target.eval()
        optimizer = torch.optim.Adam(online.parameters(), lr=config.learning_rate, eps=config.adam_epsilon)
        transitions = updates = version = 0
        if config.resume_checkpoint is not None:
            checkpoint = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
            online.load_state_dict(checkpoint["online_state_dict"])
            target.load_state_dict(checkpoint["target_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _optimizer_to_device(optimizer, device)
            transitions = int(checkpoint.get("global_transitions", 0))
            updates = int(checkpoint.get("learner_updates", 0))
            version = int(checkpoint.get("policy_version", 0))
            with global_transition_counter.get_lock():
                global_transition_counter.value = transitions
        replay = SequenceReplay(
            config.replay_capacity_sequences,
            alpha=config.prioritized_replay_alpha,
            beta=config.importance_sampling_beta,
            priority_epsilon=config.priority_epsilon,
            priority_mix=config.priority_mix,
        )
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"
        run_dir = config.runs_dir / run_name
        checkpoint_dir = config.checkpoints_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        writer = SummaryWriter(str(run_dir), max_queue=10, flush_secs=120)
        writer.add_text("config/json", json.dumps(asdict(config), default=str, indent=2, sort_keys=True), 0)
        print(f"TensorBoard run: {run_dir}", flush=True)
        print(f"Checkpoints: {checkpoint_dir}", flush=True)
        initial = ParameterUpdate(version, transitions, serialize_state_dict(online))
        for queue in parameter_queues:
            put_latest(queue, initial)

        next_tb = (transitions // config.tensorboard_interval_transitions + 1) * config.tensorboard_interval_transitions
        next_checkpoint = (transitions // config.checkpoint_interval_transitions + 1) * config.checkpoint_interval_transitions
        # If a short run ends before the first configured interval, still leave
        # a usable final checkpoint and a complete throughput/rollout schema.
        interval = IntervalMetrics()
        interval_started = time.monotonic()
        last_transitions = transitions
        update_budget = 0.0
        pending_evaluations: set[int] = set()
        last_broadcast_update = updates
        while transitions < config.total_transitions and not stop_event.is_set():
            try:
                chunk = rollout_queue.get(timeout=config.queue_timeout_seconds)
            except Empty:
                _drain_reports(metrics_queue, interval, version)
                while True:
                    try:
                        result = evaluation_result_queue.get_nowait()
                    except Empty:
                        break
                    _log_evaluation(writer, result)
                    pending_evaluations.discard(result.checkpoint_transition)
                continue
            if not isinstance(chunk, SequenceChunk):
                raise TypeError(f"unexpected rollout message: {type(chunk)}")
            for sequence in chunk.sequences:
                replay.add(sequence)
            accepted = min(chunk.transitions, max(0, config.total_transitions - transitions))
            if accepted <= 0:
                break
            transitions = increment_counter(global_transition_counter, accepted)
            if (
                replay.learning_transitions >= config.learning_starts
                and len(replay) >= config.learner_batch_size
            ):
                seq_num = len(chunk.sequences) # 本次获得的sequence个数
                update_budget += seq_num * config.updates_per_sequence
                updates_to_run = int(update_budget)
                update_budget -= updates_to_run
            else:
                updates_to_run = 0
            for _ in range(updates_to_run):
                metrics = train_batch(online, target, optimizer, replay, config, device, rng)
                updates += 1
                interval.losses.append(metrics["loss"])
                interval.q_values.append(metrics["q_mean"])
                interval.td_errors.append(metrics["td_error_abs_mean"])
                interval.gradient_norms.append(metrics["gradient_norm"])
                interval.is_weight_mins.append(metrics["is_weight_min"])
                interval.is_weight_means.append(metrics["is_weight_mean"])
                interval.is_weight_maxes.append(metrics["is_weight_max"])
                if updates % config.target_sync_interval_updates == 0:
                    target.load_state_dict(online.state_dict())
            if (
                updates_to_run
                and updates - last_broadcast_update
                >= config.parameter_broadcast_interval_updates
            ):
                version += 1
                message = ParameterUpdate(version, transitions, serialize_state_dict(online))
                for queue in parameter_queues:
                    put_latest(queue, message)
                last_broadcast_update = updates
            _drain_reports(metrics_queue, interval, version)
            while True:
                try:
                    result = evaluation_result_queue.get_nowait()
                except Empty:
                    break
                _log_evaluation(writer, result)
                pending_evaluations.discard(result.checkpoint_transition)
            if transitions >= next_tb:
                now = time.monotonic()
                _log_interval(
                    writer,
                    interval,
                    transitions=transitions,
                    updates=updates,
                    replay_size=len(replay),
                    replay_learning_transitions=replay.learning_transitions,
                    version=version,
                    elapsed=now - interval_started,
                    delta=transitions - last_transitions,
                    config=config,
                )
                interval = IntervalMetrics()
                interval_started = now
                last_transitions = transitions
                while next_tb <= transitions:
                    next_tb += config.tensorboard_interval_transitions
            if transitions >= next_checkpoint:
                path = checkpoint_dir / f"checkpoint_step_{next_checkpoint:012d}.pt"
                save_checkpoint(path, online, target, optimizer, config, transitions, updates, version)
                if config.evaluation_enabled:
                    evaluation_request_queue.put(EvaluationRequest(str(path), next_checkpoint))
                    pending_evaluations.add(next_checkpoint)
                while next_checkpoint <= transitions:
                    next_checkpoint += config.checkpoint_interval_transitions
        normal_completion = transitions >= config.total_transitions
        if normal_completion:
            now = time.monotonic()
            _log_interval(
                writer,
                interval,
                transitions=transitions,
                updates=updates,
                replay_size=len(replay),
                replay_learning_transitions=replay.learning_transitions,
                version=version,
                elapsed=now - interval_started,
                delta=transitions - last_transitions,
                config=config,
            )
            final_checkpoint = checkpoint_dir / f"checkpoint_step_{transitions:012d}.pt"
            if transitions > 0 and not final_checkpoint.exists():
                save_checkpoint(final_checkpoint, online, target, optimizer, config, transitions, updates, version)
                if config.evaluation_enabled:
                    evaluation_request_queue.put(EvaluationRequest(str(final_checkpoint), transitions))
                    pending_evaluations.add(transitions)
        stop_event.set()
        evaluation_request_queue.put(EvaluatorStop())
        if normal_completion and pending_evaluations:
            deadline = time.monotonic() + config.evaluation_shutdown_timeout_seconds
            while pending_evaluations and time.monotonic() < deadline:
                try:
                    result = evaluation_result_queue.get(timeout=1.0)
                except Empty:
                    continue
                _log_evaluation(writer, result)
                pending_evaluations.discard(result.checkpoint_transition)
            if pending_evaluations:
                raise TimeoutError(f"timed out waiting for evaluations: {sorted(pending_evaluations)}")
        if writer:
            writer.flush()
    except BaseException:
        error_queue.put(ProcessError("learner", traceback.format_exc()))
        stop_event.set()
        raise
    finally:
        if writer is not None:
            writer.close()
