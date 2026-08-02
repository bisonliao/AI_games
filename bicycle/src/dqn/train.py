"""Distributed DQN training CLI and single-GPU learner orchestration."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .actor import EpisodeMetric, ExperienceChunk, actor_process
from .checkpoint import load_checkpoint, save_checkpoint
from .config import DQNConfig, actor_epsilon
from .evaluate import EvaluationResult, evaluator_process
from .learner import DQNLearner, LearnerMetrics
from .replay import ReplayBuffer


class TrainingMetrics:
    """Central TensorBoard writer fed by actor, learner, and evaluator metrics."""

    def __init__(self, writer: SummaryWriter):
        self.writer = writer
        self.recent: deque[EpisodeMetric] = deque(maxlen=100)
        self.total_episodes = 0

    def add_episode(self, metric: EpisodeMetric, env_steps: int) -> None:
        self.recent.append(metric)
        self.total_episodes += 1
        self.writer.add_scalar("episode/return", metric.episode_return, env_steps)
        self.writer.add_scalar("episode/length", metric.episode_length, env_steps)
        self.writer.add_scalar("episode/progress_m", metric.progress_m, env_steps)
        self.writer.add_scalar("physics/roll_rms", metric.roll_rms, env_steps)
        self.writer.add_scalar("physics/max_abs_roll", metric.max_abs_roll, env_steps)
        self.writer.add_scalar("physics/lateral_drift_m", metric.lateral_drift_m, env_steps)
        self.writer.add_scalar("wind/episode_peak_n", metric.peak_wind_n, env_steps)
        self.writer.add_scalar("wind/gust_count", metric.gust_count, env_steps)
        self.writer.add_scalar(
            "wind/positive_force_fraction", metric.positive_wind_fraction, env_steps
        )
        self.writer.add_scalar(
            "wind/negative_force_fraction", metric.negative_wind_fraction, env_steps
        )
        self.writer.add_scalar(
            "physics/mean_abs_speed_error_mps",
            metric.mean_abs_speed_error_mps,
            env_steps,
        )
        self.writer.add_scalar(
            "control/reaction_wheel_saturation_fraction",
            metric.saturation_fraction,
            env_steps,
        )
        action_total = max(1, sum(metric.action_counts))
        for action, count in enumerate(metric.action_counts):
            self.writer.add_scalar(
                f"control/action_{action}_fraction", count / action_total, env_steps
            )
        self._write_business_rates(env_steps)

    def add_learner(self, metric: LearnerMetrics, env_steps: int) -> None:
        self.writer.add_scalar("dqn/td_loss", metric.loss, env_steps)
        self.writer.add_scalar("dqn/td_error_mean", metric.td_error_mean, env_steps)
        self.writer.add_scalar("dqn/td_error_max", metric.td_error_max, env_steps)
        self.writer.add_scalar("dqn/q_mean", metric.q_mean, env_steps)
        self.writer.add_scalar("dqn/target_mean", metric.target_mean, env_steps)
        self.writer.add_scalar("dqn/gradient_norm", metric.grad_norm, env_steps)

    def add_evaluation(self, result: EvaluationResult) -> None:
        step = result.env_steps
        self.writer.add_scalar("business/eval_success_rate_100", result.success_rate, step)
        self.writer.add_scalar("evaluation/fall_rate", result.fall_rate, step)
        self.writer.add_scalar("evaluation/timeout_rate", result.timeout_rate, step)
        self.writer.add_scalar("evaluation/mean_return", result.mean_return, step)
        self.writer.add_scalar("evaluation/mean_distance_m", result.mean_distance_m, step)
        self.writer.add_scalar("evaluation/mean_length", result.mean_length, step)
        self.writer.add_scalar("evaluation/roll_rms", result.roll_rms, step)

    def _write_business_rates(self, env_steps: int) -> None:
        if not self.recent:
            return
        size = len(self.recent)
        success = sum(item.outcome == "success" for item in self.recent) / size
        fall = sum(item.outcome == "fall" for item in self.recent) / size
        timeout = sum(item.outcome == "timeout" for item in self.recent) / size
        self.writer.add_scalar("business/train_success_rate_100", success, env_steps)
        self.writer.add_scalar("business/train_fall_rate_100", fall, env_steps)
        self.writer.add_scalar("business/train_timeout_rate_100", timeout, env_steps)


def train(args: argparse.Namespace) -> None:
    """Run the complete distributed training lifecycle.

    The parent process is the only learner and the only process allowed to use
    CUDA. It owns uniform replay, optimization, checkpoints, TensorBoard, and
    process supervision. Non-daemon actor processes each spawn an AsyncVectorEnv
    of PyBullet workers; a separate CPU evaluator consumes occasional policy
    snapshots. Bounded queues enforce backpressure instead of allowing actors to
    exhaust memory when collection outruns learning.

    Fresh runs use CLI hyperparameters. Resumed runs restore the DQN config from
    the checkpoint so network/replay semantics cannot silently change, while
    runtime topology (actor count, envs per actor, output directory, total steps)
    remains configurable from the new command.
    """
    # Phase 1: resolve immutable algorithm configuration and run destination.
    device = _select_device(args.device)
    resume_state = None
    if args.resume is not None:
        resume_state = load_checkpoint(args.resume, map_location=device)
        config = DQNConfig.from_dict(resume_state["config"])
    else:
        config = replace(
            DQNConfig(),
            replay_capacity=args.replay_capacity,
            replay_warmup=args.warmup,
            batch_size=args.batch_size,
            evaluation_episodes=args.evaluation_episodes,
            evaluation_interval_steps=args.evaluation_interval_steps,
            checkpoint_interval=args.checkpoint_interval,
        )
    run_dir = args.run_dir or create_run_dir(args.runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir={run_dir}")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "dqn": asdict(config),
                "actors": args.actors,
                "envs_per_actor": args.envs_per_actor,
                "seed": args.seed,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Phase 2: initialize learner-owned state and restore it before processes
    # receive their first policy snapshot.
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    learner = DQNLearner(config, device)
    replay = ReplayBuffer(config.replay_capacity, (config.observation_dim,))
    env_steps = 0
    best_success_rate = 0.0
    if resume_state is not None:
        learner.load_state_dict(resume_state["learner"])
        if "replay" in resume_state:
            replay.load_state_dict(resume_state["replay"])
        env_steps = int(resume_state["env_steps"])
        best_success_rate = float(resume_state.get("best_success_rate", 0.0))
        if "numpy_rng_state" in resume_state:
            rng.bit_generator.state = resume_state["numpy_rng_state"]
        if "torch_rng_state" in resume_state:
            torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_states" in resume_state:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_states"])

    # Phase 3: use spawn so actor creation is safe even after CUDA initialization.
    # Each actor is non-daemon, allowing it to spawn AsyncVectorEnv workers.
    context = mp.get_context("spawn")
    stop_event = context.Event()
    experience_queue = context.Queue(maxsize=args.actors * 4)
    metric_queue = context.Queue(maxsize=4096)
    weight_queues = [context.Queue(maxsize=1) for _ in range(args.actors)]
    evaluation_requests = context.Queue(maxsize=1)
    evaluation_results = context.Queue(maxsize=4)
    actors = [
        context.Process(
            target=actor_process,
            name=f"actor-{actor_id}",
            args=(
                actor_id,
                args.actors,
                args.envs_per_actor,
                actor_epsilon(actor_id, args.actors),
                config,
                args.seed,
                experience_queue,
                metric_queue,
                weight_queues[actor_id],
                stop_event,
            ),
        )
        for actor_id in range(args.actors)
    ]
    evaluator = context.Process(
        target=evaluator_process,
        name="evaluator",
        args=(
            config,
            evaluation_requests,
            evaluation_results,
            stop_event,
            config.evaluation_episodes,
        ),
    )
    # Start consumers before publishing the initial NumPy policy snapshot.
    for process in actors:
        process.start()
    evaluator.start()
    _broadcast_weights(weight_queues, learner.actor_state_dict(), learner.updates)

    # Only this process writes TensorBoard events, avoiding event-file corruption
    # from concurrent writers.
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), flush_secs=20)
    metrics = TrainingMetrics(writer)
    writer.add_text("run/config", json.dumps(asdict(config), indent=2), env_steps)
    for actor_id in range(args.actors):
        writer.add_scalar(
            f"actors/epsilon_{actor_id}", actor_epsilon(actor_id, args.actors), env_steps
        )
    last_report_steps = env_steps
    last_report_time = time.perf_counter()
    next_evaluation = max(
        config.evaluation_interval_steps,
        ((env_steps // config.evaluation_interval_steps) + 1)
        * config.evaluation_interval_steps,
    )
    next_checkpoint_update = (
        (learner.updates // config.checkpoint_interval) + 1
    ) * config.checkpoint_interval
    update_budget = 0.0
    last_broadcast_update = learner.updates
    interrupted = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        # Phase 4: ingest chunks, maintain the requested replay ratio, publish
        # weights, and schedule evaluation/checkpoint work.
        while env_steps < args.total_env_steps and not interrupted:
            try:
                chunk: ExperienceChunk = experience_queue.get(timeout=1.0)
            except queue.Empty:
                failed = [p for p in actors if not p.is_alive()]
                if failed:
                    raise RuntimeError(
                        "actor process exited unexpectedly: "
                        + ", ".join(f"{p.name}={p.exitcode}" for p in failed)
                    )
                if not evaluator.is_alive():
                    raise RuntimeError(
                        f"evaluator process exited unexpectedly: exitcode={evaluator.exitcode}"
                    )
                _drain_evaluations(evaluation_results, metrics)
                continue
            inserted = replay.extend(chunk.transitions)
            env_steps += chunk.environment_steps
            update_budget += inserted * config.replay_ratio / config.batch_size
            writer.add_scalar(
                f"actors/policy_lag_{chunk.actor_id}",
                max(0, learner.updates - chunk.policy_version),
                env_steps,
            )
            writer.add_scalar(
                f"actors/collection_steps_per_second_{chunk.actor_id}",
                chunk.environment_steps / max(chunk.collection_seconds, 1e-6),
                env_steps,
            )
            _drain_episode_metrics(metric_queue, metrics, env_steps)

            # update_budget is measured in learner minibatches. It decouples
            # asynchronous chunk sizes from the configured sampled/inserted ratio.
            while (
                len(replay) >= max(config.replay_warmup, config.batch_size)
                and update_budget >= 1.0
            ):
                learner_metric = learner.update(replay, rng)
                metrics.add_learner(learner_metric, env_steps)
                update_budget -= 1.0
                if learner.updates - last_broadcast_update >= config.actor_update_interval:
                    _broadcast_weights(
                        weight_queues, learner.actor_state_dict(), learner.updates
                    )
                    last_broadcast_update = learner.updates
                if learner.updates >= next_checkpoint_update:
                    save_checkpoint(
                        run_dir / "checkpoints/latest.pt",
                        learner,
                        config,
                        env_steps,
                        rng,
                        best_success_rate=best_success_rate,
                    )
                    if args.save_replay and learner.updates % (
                        config.checkpoint_interval * 10
                    ) == 0:
                        save_checkpoint(
                            run_dir / "checkpoints/full.pt",
                            learner,
                            config,
                            env_steps,
                            rng,
                            replay=replay,
                            best_success_rate=best_success_rate,
                        )
                    next_checkpoint_update += config.checkpoint_interval

            # Evaluation is asynchronous and the size-one request queue keeps only
            # the newest snapshot if the evaluator falls behind.
            if env_steps >= next_evaluation:
                _replace_queue_item(
                    evaluation_requests, (env_steps, learner.actor_state_dict())
                )
                next_evaluation += config.evaluation_interval_steps
            evaluation = _drain_evaluations(evaluation_results, metrics)
            if evaluation is not None and evaluation.success_rate > best_success_rate:
                best_success_rate = evaluation.success_rate
                save_checkpoint(
                    run_dir / "checkpoints/best.pt",
                    learner,
                    config,
                    env_steps,
                    rng,
                    best_success_rate=best_success_rate,
                )

            now = time.perf_counter()
            if now - last_report_time >= 5.0:
                elapsed = now - last_report_time
                throughput = (env_steps - last_report_steps) / elapsed
                writer.add_scalar("system/env_steps_per_second", throughput, env_steps)
                writer.add_scalar(
                    "system/physics_steps_per_second", throughput * 12, env_steps
                )
                writer.add_scalar("replay/size", len(replay), env_steps)
                writer.add_scalar("system/learner_updates", learner.updates, env_steps)
                writer.add_scalar("system/replay_update_budget", update_budget, env_steps)
                queue_size = _safe_qsize(experience_queue)
                if queue_size is not None:
                    writer.add_scalar("system/experience_queue_depth", queue_size, env_steps)
                if device.type == "cuda":
                    writer.add_scalar(
                        "system/gpu_allocated_bytes", torch.cuda.memory_allocated(device), env_steps
                    )
                    writer.add_scalar(
                        "system/gpu_reserved_bytes", torch.cuda.memory_reserved(device), env_steps
                    )
                print(
                    f"env_steps={env_steps:,} updates={learner.updates:,} "
                    f"replay={len(replay):,} throughput={throughput:,.0f} steps/s"
                )
                last_report_time = now
                last_report_steps = env_steps
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Phase 5: always stop children and write a resumable final checkpoint,
        # including replay by default even after SIGTERM or Ctrl+C.
        stop_event.set()
        _replace_queue_item(evaluation_requests, (None, None))
        for process in actors + [evaluator]:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        final_path = run_dir / "checkpoints/final.pt"
        save_checkpoint(
            final_path,
            learner,
            config,
            env_steps,
            rng,
            replay=replay if args.save_replay else None,
            best_success_rate=best_success_rate,
        )
        writer.flush()
        writer.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
    print(f"training stopped at env_steps={env_steps:,}; checkpoint={final_path}")


def _select_device(requested: str) -> torch.device:
    """Resolve learner device and reject accidental CPU production runs."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA learner requested but CUDA is unavailable; use --device cpu only for smoke tests"
            )
        return torch.device("cuda:0")
    if requested == "cpu":
        torch.set_num_threads(1)
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {requested}")


def create_run_dir(
    runs_root: Path,
    *,
    algorithm: str = "distributed-dqn",
    now: datetime | None = None,
    pid: int | None = None,
) -> Path:
    """Return a sortable, process-unique experiment directory.

    A numeric suffix is added only for the unlikely case that the same process
    requests more than one directory in the same second.
    """
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    process_id = os.getpid() if pid is None else pid
    base = runs_root / f"{timestamp}_{algorithm}_pid{process_id}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix:02d}")
        suffix += 1
    return candidate


def _broadcast_weights(
    queues: list[Any], state: dict[str, np.ndarray], version: int
) -> None:
    """Publish one policy version to every actor's size-one queue."""
    for destination in queues:
        _replace_queue_item(destination, (version, state))


def _replace_queue_item(destination: Any, item: Any) -> None:
    """Non-blockingly replace a stale item in a size-one/latest-value queue."""
    try:
        destination.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        destination.get_nowait()
    except queue.Empty:
        pass
    try:
        destination.put_nowait(item)
    except queue.Full:
        pass


def _drain_episode_metrics(
    source: Any, metrics: TrainingMetrics, env_steps: int
) -> None:
    """Write all currently available actor episode summaries."""
    while True:
        try:
            metric = source.get_nowait()
        except queue.Empty:
            return
        metrics.add_episode(metric, env_steps)


def _drain_evaluations(source: Any, metrics: TrainingMetrics) -> EvaluationResult | None:
    """Write completed evaluations and return the newest result, if any."""
    latest = None
    while True:
        try:
            latest = source.get_nowait()
        except queue.Empty:
            break
        metrics.add_evaluation(latest)
        print(
            f"evaluation env_steps={latest.env_steps:,} "
            f"success_rate={latest.success_rate:.3f} mean_distance={latest.mean_distance_m:.1f}m"
        )
    return latest


def _safe_qsize(source: Any) -> int | None:
    """Return queue depth on platforms that implement multiprocessing.qsize."""
    try:
        return source.qsize()
    except (NotImplementedError, OSError):
        return None


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI with units and effective defaults documented."""
    parser = argparse.ArgumentParser(
        description="Train distributed Double/Dueling DQN on BicycleBalance-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "exact output directory; when omitted, generate "
            "YYYYMMDD-HHMMSS_distributed-dqn_pidPID under --runs-root"
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="parent directory for automatically named run histories",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="learner device; actors and evaluator always remain on CPU",
    )
    parser.add_argument(
        "--actors",
        type=int,
        default=4,
        help="number of independent CPU actor controller processes",
    )
    parser.add_argument(
        "--envs-per-actor",
        type=int,
        default=4,
        help="PyBullet worker processes inside each actor's AsyncVectorEnv",
    )
    parser.add_argument(
        "--total-env-steps",
        type=int,
        default=5_000_000,
        help="global environment-step target, including all actors and envs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base seed used to derive learner, actor, and environment seeds",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "checkpoint to resume; restores saved DQN config, optimizer, counters, "
            "RNG, and replay when present"
        ),
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=1_000_000,
        help="maximum transitions in the uniform replay ring (fresh runs only)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50_000,
        help="replay transitions required before learner updates (fresh runs only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="uniform replay minibatch size (fresh runs only)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100_000,
        help="learner updates between lightweight latest checkpoints",
    )
    parser.add_argument(
        "--evaluation-interval-steps",
        type=int,
        default=500_000,
        help="global environment steps between asynchronous evaluations",
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=100,
        help="fixed-seed episodes per asynchronous evaluation",
    )
    parser.add_argument(
        "--save-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "include uniform replay in every tenth periodic checkpoint and the "
            "final checkpoint; use --no-save-replay for smaller files"
        ),
    )
    return parser


def main() -> None:
    """Validate topology arguments and enter the training lifecycle."""
    args = build_parser().parse_args()
    if args.actors <= 0 or args.envs_per_actor <= 0:
        raise SystemExit("actors and envs-per-actor must be positive")
    train(args)


if __name__ == "__main__":
    main()
