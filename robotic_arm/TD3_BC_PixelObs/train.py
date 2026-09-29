"""Train visual TD3+BC entirely from fixed expert episodes."""
from __future__ import annotations

import argparse
import math
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv

from .algorithm import OfflineTD3BC
from .bc_init import load_bc_actor, transfer_to_td3
from .config import DEFAULT_BC_MODEL, DEFAULT_DATASET, EnvConfig, run_directory, setup, tensorboard_directory
from .dataset import OfflineDataset
from .evaluator_worker import STOP, worker_main
from .visual import MultiViewCombinedExtractor


class SpaceOnlyEnv(gym.Env):
    """SB3 space provider that cannot accidentally interact with PyBullet."""

    def __init__(self, config: EnvConfig):
        self.observation_space = config.observation_space()
        self.action_space = config.action_space()

    def reset(self, *, seed=None, options=None):
        raise RuntimeError("offline training must not reset an environment")

    def step(self, action):
        raise RuntimeError("offline training must not step an environment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--bc-model", type=Path, default=DEFAULT_BC_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--total-transitions", type=int, default=5_000_000)
    parser.add_argument(
        "--total-updates", type=int, default=None,
        help="legacy alias; converted to sampled transitions using batch size",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--checkpoint-transitions", type=int, default=500_000,
        help="minimum sampled transitions between periodic checkpoints",
    )
    parser.add_argument(
        "--checkpoint-updates", type=int, default=None,
        help="legacy interval; effective interval is never below 500K transitions",
    )
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--log-transitions", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--final-eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument(
        "--bc-init", action="store_true",
        help="initialize the actor from bc_model.pt; random initialization is default",
    )
    parser.add_argument("--random-init", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--eval-randomize", action="store_true")
    parser.add_argument(
        "--alpha", type=float, default=2.5,
        help="TD3+BC Q-term scale; paper sweep uses 1, 2, 2.5, 3, 4",
    )
    parser.add_argument(
        "--augmentation-shift", type=int, default=4,
        help="consistent image shift in pixels; 0 disables DrQ augmentation",
    )
    parser.add_argument(
        "--no-augmentation", action="store_true",
        help="disable random image shifting; equivalent to --augmentation-shift 0",
    )
    return parser.parse_args()


def save_checkpoint(
    model: OfflineTD3BC,
    run_dir: Path,
    sampled: int,
    metadata: dict,
    name: str | None = None,
) -> Path:
    directory = run_dir / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    stem = name or f"td3_bc_{sampled:09d}_steps"
    target = directory / f"{stem}.zip"
    temporary = directory / f".{stem}.tmp"
    model.save(str(temporary))
    temporary.replace(target)
    target.with_suffix(".json").write_text(
        json.dumps({**metadata, "checkpoint": str(target)}, indent=2, default=str),
        encoding="utf-8",
    )
    return target


def prune_completed_checkpoints(run_dir: Path, completed: set[str], keep: int) -> None:
    periodic = sorted(
        (
            Path(path)
            for path in completed
            if Path(path).name.startswith("td3_bc_") and Path(path).exists()
        ),
        key=lambda path: path.stat().st_mtime,
    )
    for path in periodic[:-keep]:
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        completed.discard(str(path))


def drain_results(
    result_queue,
    writer: SummaryWriter,
    results_path: Path,
    run_dir: Path | None = None,
    completed_checkpoints: set[str] | None = None,
    keep_checkpoints: int = 2,
) -> int:
    count = 0
    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            break
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, default=str) + "\n")
        if completed_checkpoints is not None and result.get("checkpoint"):
            completed_checkpoints.add(str(result["checkpoint"]))
        step = int(result.get("sampled_transitions", 0))
        if result.get("ok"):
            for name in (
                "success_rate", "grasp_rate", "lift_rate", "place_rate",
                "release_rate", "premature_release_rate", "mean_return",
                "mean_episode_length",
            ):
                if name in result:
                    writer.add_scalar(f"eval/{name}", result[name], step)
        else:
            writer.add_text("eval/error", str(result.get("error")), step)
        writer.flush()
        count += 1
    if run_dir is not None and completed_checkpoints is not None:
        prune_completed_checkpoints(run_dir, completed_checkpoints, keep_checkpoints)
    return count


def main() -> None:
    args = parse_args()
    if args.total_transitions < 1 or args.batch_size < 1:
        raise SystemExit("total-transitions and batch-size must be positive")
    if args.total_updates is not None and args.total_updates < 1:
        raise SystemExit("total-updates must be positive")
    if args.total_updates is not None:
        args.total_transitions = args.total_updates * args.batch_size
    if args.batch_size > 256:
        raise SystemExit("batch-size above 256 is disabled for the 8GB GPU profile")
    if args.threads < 1 or args.keep_checkpoints < 1:
        raise SystemExit("threads and keep-checkpoints must be positive")
    if args.log_transitions < 10_000:
        raise SystemExit("--log-transitions must be at least 10000")
    if args.checkpoint_transitions < 500_000:
        raise SystemExit("periodic checkpoint interval must be at least 500000 transitions")
    if args.checkpoint_updates is not None and args.checkpoint_updates < 1:
        raise SystemExit("checkpoint-updates must be positive")
    if args.eval_episodes < 1 or args.final_eval_episodes < 1:
        raise SystemExit("checkpoint and evaluation values must be positive")
    if args.alpha <= 0.0:
        raise SystemExit("alpha must be positive")
    if args.augmentation_shift < 0:
        raise SystemExit("augmentation-shift must be non-negative")
    if args.no_augmentation:
        args.augmentation_shift = 0
    if args.bc_init and args.random_init:
        raise SystemExit("--bc-init and --random-init are mutually exclusive")
    device = setup(args.seed, args.device, threads=args.threads)
    config = EnvConfig()
    dataset = OfflineDataset(
        args.dataset,
        config,
        successful_only=True,
        seed=args.seed,
    )
    run_dir = run_directory(args.output)
    dataset.write_metadata(run_dir / "dataset_metadata.json")
    tb_dir = tensorboard_directory()
    writer = SummaryWriter(str(tb_dir))
    print(f"Run directory: {run_dir}")
    print(f"TensorBoard logs: {tb_dir}")
    print(f"Dataset: {len(dataset.episodes)} successful episodes, {len(dataset)} transitions")
    writer.add_scalar("train/augmentation_shift", args.augmentation_shift, 0)
    writer.flush()

    def factory():
        return SpaceOnlyEnv(config)

    # DummyVecEnv supplies spaces only. No reset/step call is made by this loop.
    spaces_env = DummyVecEnv([factory])
    policy_kwargs = {
        "net_arch": [256, 256],
        "features_extractor_class": MultiViewCombinedExtractor,
        "features_extractor_kwargs": {
            "n_views": 3,
            "frame_stack": config.frame_stack,
            "visual_head_version": 2,
        },
        "share_features_extractor": False,
    }
    model = OfflineTD3BC(
        "MultiInputPolicy",
        spaces_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        batch_size=args.batch_size,
        gamma=0.99,
        tau=0.005,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        alpha=args.alpha,
        bc_coef=1.0,
        augmentation_shift=args.augmentation_shift,
        seed=args.seed,
        device=device,
        verbose=0,
    )
    if args.bc_init:
        bc_actor, _ = load_bc_actor(
            args.bc_model,
            config,
            device,
        )
        error = transfer_to_td3(bc_actor, model.policy.actor, device)
        model.policy.actor_target.load_state_dict(model.policy.actor.state_dict())
        print(f"Transferred BC actor; max action error={error:.3e}")

    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    evaluator = context.Process(
        target=worker_main,
        args=(job_queue, result_queue, config.as_dict(), args.eval_device, 2),
        name="td3_bc_evaluator",
    )
    evaluator.start()
    results_path = run_dir / "eval" / "checkpoint_results.jsonl"
    last_log = 0
    last_checkpoint = 0
    started = time.perf_counter()
    submitted = 0
    completed_checkpoints: set[str] = set()

    def submit_checkpoint(episodes: int, sampled: int, name: str | None = None) -> None:
        nonlocal submitted
        metadata = {
            "sampled_transitions": sampled,
            "updates": sampled // args.batch_size,
            "dataset": dataset.metadata(),
            "env_config": config.as_dict(),
            "tensorboard_dir": str(tb_dir),
            "initialization": "bc" if args.bc_init else "random",
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "augmentation_shift": args.augmentation_shift,
            "augmentation_targets": 1,
            "augmentation_current": 1,
            "requested_total_transitions": args.total_transitions,
            "learner_threads": args.threads,
            "evaluator_threads": 2,
            "checkpoint_interval_transitions": max(
                args.checkpoint_transitions,
                (args.checkpoint_updates or 0) * args.batch_size,
            ),
        }
        checkpoint = save_checkpoint(model, run_dir, sampled, metadata, name=name)
        job_queue.put({
            "checkpoint": str(checkpoint),
            "sampled_transitions": sampled,
            "episodes": episodes,
            "seed_start": 10_000,
            "randomize": args.eval_randomize,
        })
        submitted += 1

    try:
        total_updates = math.ceil(args.total_transitions / args.batch_size)
        for update in range(1, total_updates + 1):
            metrics = model.offline_update(dataset, args.batch_size)
            sampled = update * args.batch_size
            if sampled - last_log >= args.log_transitions:
                elapsed = max(time.perf_counter() - started, 1e-6)
                writer.add_scalar("learner/sampled_transitions", sampled, sampled)
                writer.add_scalar("learner/updates", update, sampled)
                writer.add_scalar("learner/transitions_per_second", sampled / elapsed, sampled)
                writer.add_scalar(
                    "dataset/epochs", sampled / max(len(dataset), 1), sampled
                )
                for name, value in metrics.items():
                    if value == value:
                        writer.add_scalar(f"train/{name}", value, sampled)
                writer.flush()
                received = drain_results(
                    result_queue, writer, results_path, run_dir,
                    completed_checkpoints, args.keep_checkpoints,
                )
                submitted -= received
                last_log = sampled
            requested_interval = (
                args.checkpoint_updates * args.batch_size
                if args.checkpoint_updates is not None
                else 0
            )
            checkpoint_interval = max(args.checkpoint_transitions, requested_interval)
            if sampled - last_checkpoint >= checkpoint_interval:
                submit_checkpoint(args.eval_episodes, sampled)
                last_checkpoint = sampled
        final_sampled = total_updates * args.batch_size
        submit_checkpoint(args.final_eval_episodes, final_sampled, name="final_model")
    finally:
        # Wait for every queued checkpoint result before stopping the worker.
        wait_started = time.perf_counter()
        while submitted > 0:
            received = drain_results(
                result_queue, writer, results_path, run_dir,
                completed_checkpoints, args.keep_checkpoints,
            )
            submitted -= received
            if submitted:
                if not evaluator.is_alive():
                    raise RuntimeError("evaluator exited before all checkpoint results arrived")
                if time.perf_counter() - wait_started > 3600:
                    raise TimeoutError("timed out waiting for checkpoint evaluation")
                time.sleep(0.2)
        job_queue.put(STOP)
        evaluator.join(timeout=30.0)
        if evaluator.is_alive():
            evaluator.terminate()
            evaluator.join()
        writer.close()
        spaces_env.close()


if __name__ == "__main__":
    main()
