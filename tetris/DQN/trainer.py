"""Multiprocess actor-learner orchestration."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import queue
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .actor import _put_latest_weight
from .evaluator import evaluator_process
from .gather import GatheredRound, gather_process
from .learner import Learner
from .schedule import epsilon_for_schedule, final_actor_epsilons
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


def train(
    config,
    *,
    device: str = "cuda",
    total_transitions: int | None = None,
    num_actors: int | None = None,
    envs_per_actor: int | None = None,
    resume_from: str | Path | None = None,
) -> Path:
    config.validate()
    seed_everything(config.seed)
    total = int(total_transitions if total_transitions is not None else config.total_transitions)
    actors_count = int(num_actors if num_actors is not None else config.num_actors)
    env_count = int(envs_per_actor if envs_per_actor is not None else config.envs_per_actor)
    if actors_count < 1 or env_count < 1:
        raise ValueError("num_actors and envs_per_actor must both be positive")
    if config.transition_batch_size % env_count:
        raise ValueError("transition_batch_size must be divisible by envs_per_actor")
    learner = Learner(config, device=device, seed=config.seed)
    if resume_from is not None:
        learner.load_checkpoint(
            resume_from,
            total_transitions=total,
        )
    ctx = mp.get_context("spawn")
    # The gather-to-learner queue is the only reliable training-data boundary.
    # maxsize=1 bounds staleness while allowing rollout N+1 to overlap updates N.
    gathered_round_queue = ctx.Queue(maxsize=1)
    gather_error_queue = ctx.Queue(maxsize=1)
    metric_queue = ctx.Queue(maxsize=config.queue_size * 2)
    weight_queues = [ctx.Queue(maxsize=1) for _ in range(actors_count)]
    stop_event = ctx.Event()
    shared_decay_progress = ctx.Value("d", learner.decay_progress_at())
    evaluator_stop_event = ctx.Event()
    evaluation_queue = ctx.Queue(maxsize=config.max_pending_evals)
    evaluation_result_queue = ctx.Queue()
    initial = learner.state_dict_cpu()
    for weight_queue in weight_queues:
        weight_queue.put((learner.gradient_updates, initial))
    if actors_count == 1:
        base_epsilons = [0.4]
    else:
        base_epsilons = [
            float(0.05 * (0.4 / 0.05) ** (i / (actors_count - 1)))
            for i in range(actors_count)
        ]
    final_epsilons = final_actor_epsilons(
        actors_count,
        config.final_epsilon,
    )
    gather = ctx.Process(
        target=gather_process,
        kwargs={
            "actors_count": actors_count,
            "env_count": env_count,
            "base_epsilons": base_epsilons,
            "final_epsilons": final_epsilons,
            "seed": config.seed,
            "weight_queues": weight_queues,
            "metric_queue": metric_queue,
            "gathered_round_queue": gathered_round_queue,
            "error_queue": gather_error_queue,
            "decay_progress": shared_decay_progress,
            "stop_event": stop_event,
            "start_transition": learner.transitions,
            "total_transitions": total,
            "transition_batch_size": config.transition_batch_size,
            "transition_put_poll_timeout": config.transition_put_poll_timeout,
            "learner_idle_sleep": config.learner_idle_sleep,
            "actor_stats_every": config.actor_stats_every,
            "gamma": learner.gamma,
            "piece_placed_reward": config.piece_placed_reward,
            "line_clear_reward": config.line_clear_reward,
            "terminal_penalty": config.terminal_penalty,
        },
    )
    gather.start()

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
    # Report the current scheduled exploration and optimizer values.
    tb.latest("train/lr", learner.optimizer.param_groups[0]["lr"])
    tb.latest("train/gamma", learner.gamma)
    tb.latest("train/updates_per_transition", learner.updates_per_transition_at())
    tb.latest(
        "train/gradient_updates_cumulative",
        float(learner.gradient_updates),
    )
    tb.latest("train/replay_warming_up", float(learner.replay_warming_up))
    # A per-run directory prevents a fresh experiment from overwriting an old
    # checkpoint with the same transition step (including pre-placement models).
    checkpoint_dir = Path(config.checkpoint_root) / run_id
    last_checkpoint = learner.transitions
    learner_get_wait_seconds = 0.0
    learner_get_empty_polls = 0
    learner_get_empty_seconds = 0.0
    actor_queue_wait_seconds: dict[int, float] = {}
    actor_queue_wait_timeouts: dict[int, int] = {}
    actor_transition_messages: dict[int, int] = {}
    actor_transitions_sent: dict[int, int] = {}
    actor_action_counts: dict[int, np.ndarray] = {}
    actor_line_clear_transitions: dict[int, int] = {}
    actor_terminal_transitions: dict[int, int] = {}
    gather_batch_queue_wait_seconds = 0.0
    gather_batch_queue_wait_timeouts = 0
    last_evaluation = learner.transitions
    scheduled_evaluations = 0
    completed_evaluations = 0
    last_tb_log = 0

    def flush_tensorboard(*, force: bool = False) -> None:
        nonlocal last_tb_log
        if force or learner.transitions - last_tb_log >= config.tb_log_every:
            for key, value in learner.pop_training_stats().items():
                tb.latest(f"train/{key}", value)
            current_epsilons = [
                epsilon_for_schedule(
                    base_epsilon,
                    learner.decay_progress_at(),
                    final_epsilon,
                )
                for base_epsilon, final_epsilon in zip(
                    base_epsilons,
                    final_epsilons,
                )
            ]
            tb.latest("train/epsilon", float(np.mean(current_epsilons)))
            for actor_id, actor_epsilon in enumerate(current_epsilons):
                tb.latest(f"train/actor_{actor_id}_epsilon", actor_epsilon)
            # Report the optimizer's actual LR. During replay warmup the
            # optimizer is frozen, but the configured schedule is still visible.
            tb.latest("train/lr", learner.optimizer.param_groups[0]["lr"])
            tb.latest("train/gamma", learner.gamma)
            tb.latest("train/replay_size", float(len(learner.replay)))
            tb.latest(
                "train/updates_per_transition",
                learner.updates_per_transition_at(),
            )
            tb.latest(
                "train/gradient_updates_cumulative",
                float(learner.gradient_updates),
            )
            tb.latest("train/replay_warming_up", float(learner.replay_warming_up))
            tb.flush(learner.transitions)
            last_tb_log = learner.transitions

    def drain_metrics() -> None:
        nonlocal gather_batch_queue_wait_seconds, gather_batch_queue_wait_timeouts
        while True:
            try:
                metric = metric_queue.get_nowait()
            except queue.Empty:
                break
            if metric.get("kind") == "gather_communication":
                gather_batch_queue_wait_seconds = float(
                    metric.get("batch_queue_wait_seconds", 0.0)
                )
                gather_batch_queue_wait_timeouts = int(
                    metric.get("batch_queue_wait_timeouts", 0)
                )
                tb.latest(
                    "communication/gather/batch_queue_wait_seconds_cumulative",
                    gather_batch_queue_wait_seconds,
                )
                tb.latest(
                    "communication/gather/batch_queue_wait_timeout_count_cumulative",
                    gather_batch_queue_wait_timeouts,
                )
                tb.latest(
                    "communication/synchronous_round_collection_seconds",
                    float(metric["round_collection_seconds"]),
                )
                for actor_id, arrival in enumerate(metric.get("actor_arrival_seconds", ())):
                    tb.latest(
                        f"communication/actor_{actor_id}/round_arrival_seconds",
                        float(arrival),
                    )
                continue
            if metric.get("kind") == "actor_communication":
                actor_id = int(metric["actor_id"])
                actor_queue_wait_seconds[actor_id] = float(metric["queue_wait_seconds"])
                actor_queue_wait_timeouts[actor_id] = int(metric["queue_wait_timeouts"])
                actor_transition_messages[actor_id] = int(metric.get("messages_sent", 0))
                actor_transitions_sent[actor_id] = int(metric.get("transitions", 0))
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
                tb.latest(
                    f"communication/actor_{actor_id}/transitions_sent_cumulative",
                    actor_transitions_sent[actor_id],
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

    def save_checkpoint(path: Path, *, evaluate: bool) -> None:
        nonlocal scheduled_evaluations
        learner.checkpoint(path)
        if evaluate:
            try:
                evaluation_queue.put_nowait(str(path))
                scheduled_evaluations += 1
            except queue.Full:
                writer.add_scalar("evaluation/queue_full", 1, learner.transitions)

    def publish_latest_weights() -> None:
        state = learner.state_dict_cpu()
        message = (learner.gradient_updates, state)
        for weight_queue in weight_queues:
            if not _put_latest_weight(
                weight_queue,
                message,
                poll_timeout=config.transition_put_poll_timeout,
                stop_event=stop_event,
            ):
                raise RuntimeError("training stopped while publishing actor weights")

    def gather_failure() -> RuntimeError:
        try:
            error = gather_error_queue.get(timeout=0.1)
        except queue.Empty:
            return RuntimeError("gather process exited before delivering the final round")
        return RuntimeError(
            f"gather process failed with {error['type']}: {error['message']}\n"
            f"{error['traceback']}"
        )

    try:
        expected_round = 0
        completed_rounds = 0
        final_received = False
        while not final_received:
            wait_started = time.perf_counter()
            try:
                gathered_round: GatheredRound = gathered_round_queue.get_nowait()
            except queue.Empty:
                poll_seconds = time.perf_counter() - wait_started
                learner_get_wait_seconds += poll_seconds
                learner_get_empty_seconds += poll_seconds
                learner_get_empty_polls += 1
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
                if stop_event.is_set() or not gather.is_alive():
                    raise gather_failure()
                if config.learner_idle_sleep > 0:
                    time.sleep(config.learner_idle_sleep)
                continue
            learner_get_wait_seconds += time.perf_counter() - wait_started

            if gathered_round.round_id != expected_round:
                raise RuntimeError(
                    f"learner received round {gathered_round.round_id}, "
                    f"expected {expected_round}"
                )
            if len(gathered_round.batch.actions) != actors_count * config.transition_batch_size:
                raise RuntimeError(
                    f"gathered round {expected_round} has "
                    f"{len(gathered_round.batch.actions)} transitions, expected "
                    f"{actors_count * config.transition_batch_size}"
                )
            learner.add(gathered_round.batch)
            shared_decay_progress.value = learner.decay_progress_at()
            completed_rounds += 1
            tb.latest("communication/synchronous_rounds_cumulative", completed_rounds)
            for actor_id in range(actors_count):
                tb.latest(
                    f"communication/actor_{actor_id}/accepted_transitions_cumulative",
                    completed_rounds * config.transition_batch_size,
                )
            update_started = time.perf_counter()
            while True:
                if not learner.update():
                    break
                if learner.gradient_updates % config.broadcast_every == 0:
                    publish_latest_weights()
            tb.latest(
                "communication/learner/round_update_seconds",
                time.perf_counter() - update_started,
            )
            drain_metrics()
            drain_evaluations()
            tb.latest(
                "communication/learner/transition_get_wait_seconds_cumulative",
                learner_get_wait_seconds,
            )
            flush_tensorboard()
            if learner.transitions - last_checkpoint >= config.checkpoint_every:
                checkpoint_path = checkpoint_dir / f"dddqn_{learner.transitions}.pt"
                should_evaluate = learner.transitions - last_evaluation >= config.eval_every
                save_checkpoint(checkpoint_path, evaluate=should_evaluate)
                if should_evaluate:
                    last_evaluation = learner.transitions
                last_checkpoint = learner.transitions
            final_received = bool(gathered_round.is_final)
            expected_round += 1
        if learner.transitions < total:
            raise RuntimeError(
                f"final gathered round stopped at {learner.transitions}, below budget {total}"
            )
        final_path = checkpoint_dir / f"dddqn_{learner.transitions}.pt"
        should_evaluate = learner.transitions != last_evaluation
        save_checkpoint(final_path, evaluate=should_evaluate)
        drain_metrics()
        drain_evaluations()
        flush_tensorboard(force=True)
    finally:
        stop_event.set()
        gather.join(timeout=10)
        if gather.is_alive():
            gather.terminate()
            gather.join(timeout=2)
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
        gathered_round_queue.close()
        gather_error_queue.close()
        metric_queue.close()
        metric_queue.cancel_join_thread()
        for q in weight_queues:
            q.close()
            # The final asynchronous snapshot may intentionally remain unread
            # when actors stop at the transition budget.
            q.cancel_join_thread()
        evaluation_queue.close()
        evaluation_result_queue.close()
    return log_dir
