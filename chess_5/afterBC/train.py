"""Ape-X DDDQN training entry point for improving the white BC player."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .actor import actor_worker
from .common import (
    DEFAULT_BC_CHECKPOINT,
    DEFAULT_RUN_ROOT,
    EvaluationResult,
    GatherBatch,
    GatherPermit,
    RolloutSummary,
    WeightSnapshot,
    put_latest,
    validate_bc_checkpoint,
    validate_run_name,
)
from .evaluator import AsyncEvaluator, write_evaluation_json
from .gather import gather_worker
from .learner import DQNLearner
from .opponent import protocol_description


CheckpointRank = tuple[float, float, int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Train a white Gomoku agent against the frozen controlled-"
                     "stochastic BC_BEST environment."),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda",
                        help="Learner device; production defaults directly to cuda.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--board-size", type=int, default=9)

    parser.add_argument("--num-actors", type=int, default=8)
    parser.add_argument("--envs-per-actor", type=int, default=16)
    parser.add_argument(
        "--actor-torch-threads", type=int, default=1,
        help="CPU inference threads per actor process.",
    )
    parser.add_argument("--actor-batch-size", type=int, default=256)
    parser.add_argument("--learner-queue-size", type=int, default=2)
    parser.add_argument("--total-transitions", type=int, default=20_000_000)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--min-replay-size", type=int, default=50_000)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument(
        "--updates-per-transition", type=float, default=0.02,
        help=("Gradient updates credited per newly gathered transition; with the "
              "default 8x256 gather round this averages 40.96 updates per round."),
    )
    parser.add_argument("--target-update", type=int, default=2_500)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--weight-sync-updates", type=int, default=1_000)

    parser.add_argument("--log-interval", type=int, default=100_000)
    parser.add_argument("--checkpoint-interval", type=int, default=500_000)
    parser.add_argument(
        "--stochastic-eval-games", "--statistical-eval-games",
        dest="stochastic_eval_games", type=int, default=20,
    )
    parser.add_argument("--disable-evaluation", action="store_true")
    parser.add_argument("--disable-tensorboard", action="store_true")
    args = parser.parse_args(argv)
    if args.board_size != 9:
        parser.error("afterBC is tied to the 9x9 BC_BEST checkpoint")
    positive = (
        "num_actors", "envs_per_actor", "actor_torch_threads", "actor_batch_size",
        "learner_queue_size",
        "total_transitions", "n_step", "batch_size", "replay_size",
        "min_replay_size", "target_update", "weight_sync_updates", "log_interval",
        "checkpoint_interval", "stochastic_eval_games",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.per_alpha <= 1 or not 0 <= args.per_beta_start <= 1:
        parser.error("PER alpha and beta must be in [0, 1]")
    if args.updates_per_transition < 0:
        parser.error("--updates-per-transition cannot be negative")
    return args


def _json_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _write_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def tensorboard_log_dir(
    run_dir: Path,
    run_name: str,
    *,
    timestamp: str | None = None,
    process_id: int | None = None,
) -> Path:
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid() if process_id is None else int(process_id)
    return run_dir / "tensorboard" / f"{run_name}_{timestamp}_pid{process_id}"


class TrainingLogger:
    def __init__(self, log_dir: Path, *, enabled: bool) -> None:
        self.writer = SummaryWriter(str(log_dir)) if enabled else None
        self.cumulative = {
            "episodes": 0.0, "wins": 0.0, "losses": 0.0, "draws": 0.0,
            "returns": 0.0, "white_moves": 0.0, "collection_seconds": 0.0,
            "blocked_seconds": 0.0,
        }
        self.window = dict(self.cumulative)
        self.actor_epsilons: dict[int, float] = {}
        self.actor_policy_versions: dict[int, int] = {}
        self.actor_window_transitions: dict[int, float] = {}
        self.actor_window_seconds: dict[int, float] = {}
        self.last_log_time = time.perf_counter()
        self.last_log_step = 0
        self.last_log_updates = 0

    def consume(self, summary: RolloutSummary) -> None:
        values = {
            "episodes": summary.episodes,
            "wins": summary.white_wins,
            "losses": summary.white_losses,
            "draws": summary.draws,
            "returns": summary.return_sum,
            "white_moves": summary.white_move_sum,
            "collection_seconds": summary.collection_seconds,
            "blocked_seconds": summary.blocked_seconds,
        }
        for key, value in values.items():
            self.cumulative[key] += float(value)
            self.window[key] += float(value)
        self.actor_epsilons[summary.actor_id] = summary.epsilon
        self.actor_policy_versions[summary.actor_id] = summary.policy_version
        self.actor_window_transitions[summary.actor_id] = (
            self.actor_window_transitions.get(summary.actor_id, 0.0)
            + summary.transitions
        )
        self.actor_window_seconds[summary.actor_id] = (
            self.actor_window_seconds.get(summary.actor_id, 0.0)
            + summary.collection_seconds
        )

    def log(self, *, step: int, update_steps: int, replay_size: int,
            metrics: dict[str, float] | None) -> None:
        now = time.perf_counter()
        elapsed = max(1e-6, now - self.last_log_time)
        throughput = (step - self.last_log_step) / elapsed
        update_throughput = (update_steps - self.last_log_updates) / elapsed
        actor_rates = {
            actor_id: transitions / max(1e-6, self.actor_window_seconds[actor_id])
            for actor_id, transitions in self.actor_window_transitions.items()
        }
        actor_collection_throughput = sum(actor_rates.values())
        episodes = max(1.0, self.window["episodes"])
        win_rate = self.window["wins"] / episodes
        loss_rate = self.window["losses"] / episodes
        draw_rate = self.window["draws"] / episodes
        mean_return = self.window["returns"] / episodes
        mean_length = self.window["white_moves"] / episodes
        print(
            f"steps={step} updates={update_steps} replay={replay_size} "
            f"transitions/s={throughput:.1f} actor_collection/s={actor_collection_throughput:.1f} "
            f"updates/s={update_throughput:.2f} "
            f"W/L/D={win_rate:.3f}/{loss_rate:.3f}/{draw_rate:.3f} "
            f"return={mean_return:.3f} white_len={mean_length:.1f}",
            flush=True,
        )
        if self.writer is not None:
            writer = self.writer
            writer.add_scalar("Rollout/white_success_rate", win_rate, step)
            writer.add_scalar("Rollout/white_loss_rate", loss_rate, step)
            writer.add_scalar("Rollout/draw_rate", draw_rate, step)
            writer.add_scalar("Rollout/mean_return", mean_return, step)
            writer.add_scalar("Rollout/mean_white_moves", mean_length, step)
            writer.add_scalar("System/transitions_per_second", throughput, step)
            writer.add_scalar("Throughput/global_transitions_per_second", throughput, step)
            writer.add_scalar("Throughput/actor_collection_transitions_per_second",
                              actor_collection_throughput, step)
            writer.add_scalar("Throughput/learner_updates_per_second", update_throughput, step)
            writer.add_scalar("System/actor_blocked_seconds", self.window["blocked_seconds"], step)
            writer.add_scalar("Learner/replay_size", replay_size, step)
            writer.add_scalar("Learner/update_steps", update_steps, step)
            writer.add_scalar("Exploration/global_schedule_fraction", min(1.0, step / 1_000_000), step)
            for actor_id, epsilon in sorted(self.actor_epsilons.items()):
                writer.add_scalar(f"Exploration/actor_{actor_id}_epsilon", epsilon, step)
                lag = update_steps - self.actor_policy_versions.get(actor_id, update_steps)
                writer.add_scalar(f"System/actor_{actor_id}_policy_lag", lag, step)
                if actor_id in actor_rates:
                    writer.add_scalar(
                        f"Throughput/actor_{actor_id}_collection_transitions_per_second",
                        actor_rates[actor_id], step,
                    )
            if metrics:
                for key, value in metrics.items():
                    writer.add_scalar(f"Learner/{key}", value, step)
            writer.flush()
        self.window = {key: 0.0 for key in self.window}
        self.actor_window_transitions.clear()
        self.actor_window_seconds.clear()
        self.last_log_time = now
        self.last_log_step = step
        self.last_log_updates = update_steps

    def evaluation(self, result: dict[str, Any], step: int) -> None:
        if self.writer is None:
            return
        deterministic = result["deterministic"]
        stochastic = _stochastic_metrics(result)
        self.writer.add_scalar("Evaluation/deterministic_white_win",
                               float(deterministic["winner"] == "white"), step)
        self.writer.add_scalar("Evaluation/stochastic_success", float(result["success"]), step)
        for key in ("white_win_rate", "white_loss_rate", "draw_rate", "white_score_rate", "mean_moves"):
            self.writer.add_scalar(f"Evaluation/stochastic_{key}", stochastic[key], step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _drain_rollouts(
    rollout_queue: Any,
    logger: TrainingLogger,
    *,
    wait_for_actors: int = 0,
    timeout: float = 2.0,
) -> None:
    seen_actors: set[int] = set(logger.actor_window_transitions)
    deadline = time.monotonic() + timeout
    while True:
        try:
            if wait_for_actors and len(seen_actors) < wait_for_actors:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                summary = rollout_queue.get(timeout=min(0.1, remaining))
            else:
                summary = rollout_queue.get_nowait()
        except queue.Empty:
            if not wait_for_actors or time.monotonic() >= deadline:
                return
            continue
        if not isinstance(summary, RolloutSummary):
            raise TypeError(f"unexpected rollout log: {type(summary)!r}")
        logger.consume(summary)
        seen_actors.add(summary.actor_id)


def _raise_worker_error(error_queue: Any) -> None:
    try:
        error = error_queue.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"{error['worker']} failed:\n{error['traceback']}")


def _stochastic_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("stochastic", result.get("statistical"))
    if not isinstance(metrics, dict):
        raise KeyError("evaluation result has no stochastic metrics")
    return metrics


def _checkpoint_rank(result: dict[str, Any]) -> CheckpointRank:
    stochastic = _stochastic_metrics(result)
    return (
        float(stochastic["white_score_rate"]),
        float(stochastic["white_win_rate"]),
        int(result["deterministic"]["winner"] == "white"),
    )


def _copy_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _handle_evaluations(
    results: list[EvaluationResult],
    *,
    evaluation_dir: Path,
    best_path: Path,
    logger: TrainingLogger,
    best_rank: CheckpointRank | None,
) -> CheckpointRank | None:
    for message in results:
        if message.error is not None:
            print(f"WARNING: evaluation failed for {message.checkpoint}:\n{message.error}", file=sys.stderr)
            continue
        assert message.result is not None
        result = {**message.result, "step": message.step}
        write_evaluation_json(result, evaluation_dir / f"step_{message.step:012d}.json")
        logger.evaluation(result, message.step)
        rank = _checkpoint_rank(result)
        if best_rank is None or rank > best_rank:
            _copy_atomic(Path(message.checkpoint), best_path)
            best_rank = rank
        print(
            f"evaluation step={message.step} deterministic={result['deterministic']['winner']} "
            f"score={_stochastic_metrics(result)['white_score_rate']:.3f} "
            f"win={_stochastic_metrics(result)['white_win_rate']:.3f}",
            flush=True,
        )
    return best_rank


def _checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    return checkpoint_dir / f"step_{step:012d}.pt"


def _load_existing_best(
    evaluation_dir: Path,
) -> tuple[CheckpointRank, Path] | None:
    best: tuple[CheckpointRank, Path] | None = None
    if not evaluation_dir.is_dir():
        return None
    for path in sorted(evaluation_dir.glob("step_*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            rank = _checkpoint_rank(result)
            checkpoint = Path(result["checkpoint"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if checkpoint.is_file() and (best is None or rank > best[0]):
            best = (rank, checkpoint)
    return best


def run_training(args: argparse.Namespace) -> None:
    run_name = validate_run_name(args.run_name)
    validate_bc_checkpoint(args.bc_checkpoint)
    run_dir = args.run_root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    evaluation_dir = run_dir / "evaluations"
    latest_path = run_dir / "latest.pt"
    best_path = run_dir / "best.pt"
    tb_log_dir = tensorboard_log_dir(run_dir, run_name)
    config = _json_config(args)
    config["black_opponent_protocol"] = protocol_description()
    config["tensorboard_log_dir"] = None if args.disable_tensorboard else str(tb_log_dir)
    _write_config(config, run_dir / "config.json")
    logger = TrainingLogger(tb_log_dir, enabled=not args.disable_tensorboard)
    if not args.disable_tensorboard:
        print(f"TensorBoard log dir: {tb_log_dir}", flush=True)

    learner = DQNLearner(
        args.bc_checkpoint, device=args.device, board_size=args.board_size,
        lr=args.lr, gamma=args.gamma, batch_size=args.batch_size,
        replay_size=args.replay_size, min_replay_size=args.min_replay_size,
        per_alpha=args.per_alpha, per_beta_start=args.per_beta_start,
        target_update=args.target_update, grad_clip=args.grad_clip, seed=args.seed,
    )
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"--resume requested but checkpoint is missing: {latest_path}")
        learner.load_checkpoint(latest_path)
        print(f"Resumed learner from {latest_path} at step {learner.global_step}", flush=True)
    logger.last_log_step = learner.global_step
    logger.last_log_updates = learner.update_steps

    ctx = mp.get_context("spawn")
    evaluator = None if args.disable_evaluation else AsyncEvaluator(
        args.bc_checkpoint, board_size=args.board_size,
        stochastic_games=args.stochastic_eval_games, seed=args.seed + 70_000,
        context=ctx,
    )
    saved_steps: set[int] = set()
    existing_best = _load_existing_best(evaluation_dir) if args.resume else None
    best_rank: CheckpointRank | None = None
    if existing_best is not None:
        best_rank, existing_checkpoint = existing_best
        _copy_atomic(existing_checkpoint, best_path)

    def save_and_evaluate() -> Path:
        path = _checkpoint_path(checkpoint_dir, learner.global_step)
        if learner.global_step not in saved_steps:
            learner.save_checkpoint(path, config, latest=latest_path)
            saved_steps.add(learner.global_step)
            print(f"Saved checkpoint: {path}", flush=True)
            if evaluator is not None:
                evaluator.submit(path, learner.global_step)
        return path

    if not args.resume:
        save_and_evaluate()
    if learner.global_step >= args.total_transitions:
        save_and_evaluate()
        if evaluator is not None:
            best_rank = _handle_evaluations(
                evaluator.close(drain=True), evaluation_dir=evaluation_dir,
                best_path=best_path, logger=logger, best_rank=best_rank,
            )
        logger.close()
        return

    transition_queues = [ctx.Queue(maxsize=1) for _ in range(args.num_actors)]
    permit_queues = [ctx.Queue(maxsize=1) for _ in range(args.num_actors)]
    weight_queues = [ctx.Queue(maxsize=1) for _ in range(args.num_actors)]
    learner_queue = ctx.Queue(maxsize=args.learner_queue_size)
    rollout_queue = ctx.Queue()
    error_queue = ctx.Queue()
    stop_event = ctx.Event()
    initial_snapshot = WeightSnapshot(learner.update_steps, learner.snapshot())
    for permit_queue, weight_queue in zip(permit_queues, weight_queues):
        permit_queue.put(GatherPermit("continue", learner.global_step))
        weight_queue.put(initial_snapshot)

    actors = [
        ctx.Process(
            target=actor_worker,
            name=f"afterbc-actor-{actor_id}",
            args=(
                actor_id, args.num_actors, args.envs_per_actor, args.board_size,
                args.actor_batch_size, args.n_step, args.gamma,
                args.seed + actor_id * 100_003 + learner.global_step * 17,
                args.actor_torch_threads,
                args.bc_checkpoint,
                transition_queues[actor_id], permit_queues[actor_id],
                weight_queues[actor_id], rollout_queue, error_queue, stop_event,
            ),
        )
        for actor_id in range(args.num_actors)
    ]
    gather = ctx.Process(
        target=gather_worker,
        name="afterbc-gather",
        args=(transition_queues, permit_queues, learner_queue, error_queue, stop_event),
        kwargs={
            "actor_batch_size": args.actor_batch_size,
            "start_global_step": learner.global_step,
            "target_global_step": args.total_transitions,
        },
    )
    for process in actors:
        process.start()
    gather.start()

    next_log = ((learner.global_step // args.log_interval) + 1) * args.log_interval
    next_checkpoint = (
        (learner.global_step // args.checkpoint_interval) + 1
    ) * args.checkpoint_interval
    next_weight_sync = (
        (learner.update_steps // args.weight_sync_updates) + 1
    ) * args.weight_sync_updates
    latest_metrics: dict[str, float] | None = None
    first_training_log = True
    completed = False
    try:
        while True:
            _raise_worker_error(error_queue)
            _drain_rollouts(rollout_queue, logger)
            if evaluator is not None:
                best_rank = _handle_evaluations(
                    evaluator.poll(), evaluation_dir=evaluation_dir,
                    best_path=best_path, logger=logger, best_rank=best_rank,
                )
            try:
                batch = learner_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if not isinstance(batch, GatherBatch):
                raise TypeError(f"unexpected learner message: {type(batch)!r}")
            learner.add_packet(batch.packet, batch.global_step)
            if len(learner.replay) >= max(args.min_replay_size, args.batch_size):
                learner.update_credit += len(batch.packet) * args.updates_per_transition
                updates = int(learner.update_credit)
                learner.update_credit -= updates
                for update_index in range(updates):
                    latest_metrics = learner.train_step(total_training_steps=args.total_transitions)
                    if (update_index + 1) % 25 == 0:
                        _raise_worker_error(error_queue)
                        _drain_rollouts(rollout_queue, logger)
            if learner.update_steps >= next_weight_sync:
                snapshot = WeightSnapshot(learner.update_steps, learner.snapshot())
                for weight_queue in weight_queues:
                    put_latest(weight_queue, snapshot)
                next_weight_sync = (
                    (learner.update_steps // args.weight_sync_updates) + 1
                ) * args.weight_sync_updates
            if learner.global_step >= next_checkpoint:
                save_and_evaluate()
                while next_checkpoint <= learner.global_step:
                    next_checkpoint += args.checkpoint_interval
            if first_training_log or learner.global_step >= next_log:
                _drain_rollouts(
                    rollout_queue, logger, wait_for_actors=args.num_actors
                )
                logger.log(
                    step=learner.global_step, update_steps=learner.update_steps,
                    replay_size=len(learner.replay), metrics=latest_metrics,
                )
                first_training_log = False
                while next_log <= learner.global_step:
                    next_log += args.log_interval
            if batch.final:
                completed = True
                break

        _drain_rollouts(rollout_queue, logger)
        save_and_evaluate()
        for process in actors:
            process.join(timeout=30.0)
        gather.join(timeout=30.0)
        if evaluator is not None:
            best_rank = _handle_evaluations(
                evaluator.close(drain=True), evaluation_dir=evaluation_dir,
                best_path=best_path, logger=logger, best_rank=best_rank,
            )
    finally:
        stop_event.set()
        for permit_queue in permit_queues:
            put_latest(permit_queue, GatherPermit("stop", learner.global_step))
        for process in actors + [gather]:
            if process.is_alive():
                process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        if evaluator is not None and not evaluator.closed:
            evaluator.close(drain=completed)
        logger.close()
        for ipc_queue in (
            *transition_queues, *permit_queues, *weight_queues,
            learner_queue, rollout_queue, error_queue,
        ):
            ipc_queue.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
