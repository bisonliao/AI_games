"""Multiprocess actor-learner orchestration."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import queue
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .actor import actor_process
from .evaluator import evaluator_process
from .learner import Learner
from .utils import run_name, seed_everything


class _TensorBoardBuffer:
    """Accumulate scalar samples and write one point per transition interval."""

    def __init__(self, writer: SummaryWriter) -> None:
        self.writer = writer
        self._means: dict[str, tuple[float, int]] = {}
        self._latest: dict[str, float] = {}

    def mean(self, tag: str, value: float) -> None:
        total, count = self._means.get(tag, (0.0, 0))
        self._means[tag] = (total + float(value), count + 1)

    def latest(self, tag: str, value: float) -> None:
        self._latest[tag] = float(value)

    def flush(self, step: int) -> None:
        for tag, (total, count) in self._means.items():
            if count:
                self.writer.add_scalar(tag, total / count, step)
        for tag, value in self._latest.items():
            self.writer.add_scalar(tag, value, step)
        self._means.clear()
        self._latest.clear()


def train(config, *, device: str = "cuda", total_transitions: int | None = None, num_actors: int | None = None, envs_per_actor: int | None = None) -> Path:
    config.validate()
    seed_everything(config.seed)
    total = int(total_transitions if total_transitions is not None else config.total_transitions)
    actors_count = int(num_actors if num_actors is not None else config.num_actors)
    env_count = int(envs_per_actor if envs_per_actor is not None else config.envs_per_actor)
    if actors_count < 1 or env_count < 1:
        raise ValueError("num_actors and envs_per_actor must both be positive")
    learner = Learner(config, device=device, seed=config.seed)
    ctx = mp.get_context("spawn")
    transition_queue = ctx.Queue(maxsize=config.queue_size)
    metric_queue = ctx.Queue(maxsize=config.queue_size * 2)
    weight_queues = [ctx.Queue(maxsize=1) for _ in range(actors_count)]
    stop_event = ctx.Event()
    evaluator_stop_event = ctx.Event()
    evaluation_queue = ctx.Queue(maxsize=config.max_pending_evals)
    evaluation_result_queue = ctx.Queue()
    initial = learner.state_dict_cpu()
    for q in weight_queues:
        q.put(initial)
    actors = []
    if actors_count == 1:
        epsilons = [0.4]
    else:
        epsilons = [
            float(0.05 * (0.4 / 0.05) ** (i / (actors_count - 1)))
            for i in range(actors_count)
        ]
    for actor_id in range(actors_count):
        actor_seed = config.seed + actor_id * 1_000_000
        process = ctx.Process(
            target=actor_process,
            args=(
                actor_id,
                env_count,
                actor_seed,
                epsilons[actor_id],
                transition_queue,
                metric_queue,
                weight_queues[actor_id],
                stop_event,
                config.transition_put_poll_timeout,
                config.actor_stats_every,
                config.transition_batch_size,
                config.transition_batch_max_wait,
                config.gamma,
                config.piece_placed_reward,
                config.line_clear_reward,
                config.terminal_penalty,
            ),
            daemon=True,
        )
        process.start()
        actors.append(process)

    evaluator = ctx.Process(
        target=evaluator_process,
        args=(
            evaluation_queue,
            evaluation_result_queue,
            evaluator_stop_event,
        ),
        kwargs={
            "episodes": config.eval_episodes,
            "max_steps": config.eval_max_steps,
            "seed": config.seed + 10_000_000,
            "device": config.eval_device,
        },
        daemon=True,
    )
    evaluator.start()

    run_id = run_name()
    log_dir = Path(config.log_root) / run_id
    writer = SummaryWriter(log_dir=str(log_dir))
    tb = _TensorBoardBuffer(writer)
    # Report one compact exploration curve: the fixed mean across actors.
    # Reading LR from the optimizer keeps that metric correct if scheduling is
    # added later.
    tb.latest("train/lr", learner.optimizer.param_groups[0]["lr"])
    # A per-run directory prevents a fresh experiment from overwriting an old
    # checkpoint with the same transition step (including pre-placement models).
    checkpoint_dir = Path(config.checkpoint_root) / run_id
    last_checkpoint = 0
    learner_get_wait_seconds = 0.0
    learner_get_poll_timeouts = 0
    learner_get_empty_polls = 0
    learner_get_empty_seconds = 0.0
    actor_queue_wait_seconds: dict[int, float] = {}
    actor_queue_wait_timeouts: dict[int, int] = {}
    actor_transition_messages: dict[int, int] = {}
    actor_action_counts: dict[int, np.ndarray] = {}
    actor_line_clear_transitions: dict[int, int] = {}
    actor_terminal_transitions: dict[int, int] = {}
    last_evaluation = 0
    scheduled_evaluations = 0
    completed_evaluations = 0
    last_tb_log = 0

    def flush_tensorboard(*, force: bool = False) -> None:
        nonlocal last_tb_log
        if force or learner.transitions - last_tb_log >= config.tb_log_every:
            for key, value in learner.pop_training_stats().items():
                tb.latest(f"train/{key}", value)
            tb.latest("train/epsilon", float(np.mean(epsilons)))
            tb.flush(learner.transitions)
            last_tb_log = learner.transitions

    def drain_metrics() -> None:
        while True:
            try:
                metric = metric_queue.get_nowait()
            except queue.Empty:
                break
            if metric.get("kind") == "actor_communication":
                actor_id = int(metric["actor_id"])
                actor_queue_wait_seconds[actor_id] = float(metric["queue_wait_seconds"])
                actor_queue_wait_timeouts[actor_id] = int(metric["queue_wait_timeouts"])
                actor_transition_messages[actor_id] = int(metric.get("messages_sent", 0))
                actor_action_counts[actor_id] = np.asarray(metric.get("action_counts", ()), dtype=np.int64)
                actor_line_clear_transitions[actor_id] = int(metric.get("line_clear_transitions", 0))
                actor_terminal_transitions[actor_id] = int(metric.get("terminal_transitions", 0))
                tb.latest(
                    "communication/actors_transition_put_wait_seconds_cumulative",
                    sum(actor_queue_wait_seconds.values()),
                )
                tb.latest(
                    "communication/actors_transition_put_poll_timeout_count_cumulative",
                    sum(actor_queue_wait_timeouts.values()),
                )
                tb.latest(
                    "communication/actors_transition_put_message_count_cumulative",
                    sum(actor_transition_messages.values()),
                )
                nonempty_counts = [counts for counts in actor_action_counts.values() if counts.size]
                if nonempty_counts:
                    total_action_counts = np.stack(nonempty_counts).sum(axis=0)
                    total_actions = max(int(total_action_counts.sum()), 1)
                    tb.latest(
                        "events/line_clear_transition_fraction_cumulative",
                        sum(actor_line_clear_transitions.values()) / total_actions,
                    )
                    tb.latest(
                        "events/terminal_transition_fraction_cumulative",
                        sum(actor_terminal_transitions.values()) / total_actions,
                    )
                    rotations = total_action_counts.reshape(4, 10).sum(axis=1)
                    columns = total_action_counts.reshape(4, 10).sum(axis=0)
                    for index, count in enumerate(rotations):
                        tb.latest(f"action/rotation_{index}_fraction_cumulative", count / total_actions)
                    for index, count in enumerate(columns):
                        tb.latest(f"action/target_column_{index}_fraction_cumulative", count / total_actions)
                continue
            for key in (
                "return",
                "length",
                "lines",
                "survival_pieces",
                "aggregate_height",
                "max_height",
                "holes",
                "bumpiness",
                "wells",
            ):
                tb.mean(f"episode/{key}", metric[key])

    def drain_evaluations() -> None:
        nonlocal completed_evaluations
        while True:
            try:
                result = evaluation_result_queue.get_nowait()
            except queue.Empty:
                break
            completed_evaluations += 1
            step = int(result.get("transition_step", learner.transitions))
            if "error" in result:
                writer.add_text("evaluation/error", result["error"], step)
                continue
            for key in ("mean_return", "mean_survival_pieces", "mean_lines", "mean_length", "truncated_episodes"):
                writer.add_scalar(f"evaluation/{key}", result[key], step)

    def save_checkpoint(path: Path, *, schedule_evaluation: bool) -> None:
        nonlocal scheduled_evaluations
        learner.checkpoint(path)
        if schedule_evaluation:
            try:
                evaluation_queue.put_nowait(str(path))
                scheduled_evaluations += 1
            except queue.Full:
                writer.add_scalar("evaluation/schedule_failed", 1, learner.transitions)

    try:
        while learner.transitions < total:
            wait_started = time.perf_counter()
            try:
                if config.transition_get_poll_timeout > 0:
                    batch = transition_queue.get(timeout=config.transition_get_poll_timeout)
                else:
                    batch = transition_queue.get_nowait()
            except queue.Empty:
                idle_seconds = time.perf_counter() - wait_started
                learner_get_wait_seconds += idle_seconds
                learner_get_empty_seconds += idle_seconds
                learner_get_empty_polls += 1
                if config.transition_get_poll_timeout > 0:
                    learner_get_poll_timeouts += 1
                drain_metrics()
                drain_evaluations()
                tb.latest(
                    "communication/learner/transition_get_empty_seconds_cumulative",
                    learner_get_empty_seconds,
                )
                tb.latest(
                    "communication/learner/transition_get_empty_poll_count_cumulative",
                    learner_get_empty_polls,
                )
                flush_tensorboard()
                if not any(p.is_alive() for p in actors):
                    raise RuntimeError("all actor processes exited before reaching the transition budget")
                if config.learner_idle_sleep > 0:
                    time.sleep(config.learner_idle_sleep)
                continue
            learner_get_wait_seconds += time.perf_counter() - wait_started
            learner.add(batch)
            # 一个 IPC batch 可能包含数百条 transition。这里按 transition
            # cadence 补齐所有到期 update，而不是“收到一条消息只更新一次”。
            # 因此 transition_batch_size 只影响通信效率，不改变每条样本对应的
            # update_every 训练频率。
            while True:
                if not learner.update():
                    break
                if learner.gradient_updates % config.broadcast_every == 0:
                    state = learner.state_dict_cpu()
                    for q in weight_queues:
                        try:
                            while True:
                                q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            q.put_nowait(state)
                        except queue.Full:
                            pass
            drain_metrics()
            drain_evaluations()
            tb.latest(
                "communication/learner/transition_get_wait_seconds_cumulative",
                learner_get_wait_seconds,
            )
            tb.latest(
                "communication/learner/transition_get_poll_timeout_count_cumulative",
                learner_get_poll_timeouts,
            )
            flush_tensorboard()
            if learner.transitions - last_checkpoint >= config.checkpoint_every:
                checkpoint_path = checkpoint_dir / f"dddqn_{learner.transitions}.pt"
                should_evaluate = learner.transitions - last_evaluation >= config.eval_every
                save_checkpoint(checkpoint_path, schedule_evaluation=should_evaluate)
                if should_evaluate:
                    last_evaluation = learner.transitions
                last_checkpoint = learner.transitions
        final_path = checkpoint_dir / f"dddqn_{learner.transitions}.pt"
        should_evaluate = learner.transitions != last_evaluation
        save_checkpoint(final_path, schedule_evaluation=should_evaluate)
        drain_metrics()
        drain_evaluations()
        flush_tensorboard(force=True)
    finally:
        stop_event.set()
        for process in actors:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        # Let already scheduled checkpoints finish without holding up the learner loop.
        deadline = time.perf_counter() + config.eval_shutdown_timeout
        while completed_evaluations < scheduled_evaluations and time.perf_counter() < deadline:
            drain_evaluations()
            if completed_evaluations >= scheduled_evaluations:
                break
            time.sleep(0.05)
        evaluator_stop_event.set()
        try:
            evaluation_queue.put_nowait(None)
        except queue.Full:
            pass
        evaluator.join(timeout=config.eval_shutdown_timeout)
        if evaluator.is_alive():
            evaluator.terminate()
            evaluator.join(timeout=2)
        drain_metrics()
        drain_evaluations()
        flush_tensorboard(force=True)
        writer.flush()
        writer.close()
        transition_queue.close()
        metric_queue.close()
        for q in weight_queues:
            q.close()
        evaluation_queue.close()
        evaluation_result_queue.close()
    return log_dir
