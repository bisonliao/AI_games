"""Train multi-view pixel-observation SAC with parallel PyBullet workers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .callbacks import (
    PixelTaskMetricsCallback,
    RolloutActionEntropyCallback,
    VisualHealthCallback,
)
from .async_eval import AsyncCpuEvalCallback
from .env import DEFAULT_CAMERA_SCALE, DEFAULT_FRAME_STACK, DEFAULT_IMAGE_SIZE
from .policy import MultiViewCombinedExtractor
from .timing import TimedSAC, TimingStats, TimingTensorBoardCallback, TimingVecEnv
from .utils import make_env_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="pick_place", choices=["reach", "pick_place"])
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-actors", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument("--camera-scale", type=float, default=DEFAULT_CAMERA_SCALE)
    parser.add_argument("--visual-head-version", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--share-features-extractor",
        action="store_true",
        help=(
            "share the actor/critic encoder; disabled by default so the "
            "critic TD loss trains its own visual encoder"
        ),
    )
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=-1,
        help=(
            "gradient updates after each rollout; -1 matches the number of "
            "new transitions (UTD approximately 1)"
        ),
    )
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument(
        "--time-log-freq",
        type=int,
        default=5_000,
        help="write aggregated time/* metrics to TensorBoard every N timesteps",
    )
    parser.add_argument(
        "--entropy-log-freq",
        type=int,
        default=5_000,
        help="estimate rollout/action_entropy every N timesteps",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("SAC_PixelObs/runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--start-method",
        choices=["forkserver", "spawn", "fork"],
        default="forkserver",
    )
    args = parser.parse_args()
    for name in (
        "n_actors", "image_size", "frame_stack", "max_episode_steps", "action_repeat",
        "buffer_size", "learning_starts", "batch_size", "train_freq",
        "eval_freq", "eval_episodes", "checkpoint_freq", "time_log_freq",
        "entropy_log_freq",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.gradient_steps == 0 or args.gradient_steps < -1:
        parser.error("--gradient-steps must be -1 or a positive integer")
    if args.camera_scale <= 0.0:
        parser.error("--camera-scale must be positive")
    if args.total_timesteps is None:
        args.total_timesteps = 200_000 if args.task == "reach" else 2_000_000
    return args


def build_run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"{args.task}_pixelobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("checkpoints", "best_model", "eval", "monitor", "tensorboard"):
        (run_dir / child).mkdir()
    return run_dir


def main() -> None:
    args = parse_args()
    run_dir = build_run_dir(args)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False, default=str)

    factories = [
        make_env_factory(
            task=args.task,
            rank=rank,
            seed=args.seed,
            image_size=args.image_size,
            frame_stack=args.frame_stack,
            max_episode_steps=args.max_episode_steps,
            action_repeat=args.action_repeat,
            camera_scale=args.camera_scale,
            monitor_dir=run_dir / "monitor",
        )
        for rank in range(args.n_actors)
    ]
    base_train_env = (
        DummyVecEnv(factories)
        if args.n_actors == 1
        else SubprocVecEnv(factories, start_method=args.start_method)
    )
    timing_stats = TimingStats()
    train_env = TimingVecEnv(base_train_env, timing_stats)
    try:
        model = TimedSAC(
            "MultiInputPolicy",
            train_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=(args.train_freq, "step"),
            gradient_steps=args.gradient_steps,
            ent_coef="auto",
            tensorboard_log=str(run_dir / "tensorboard"),
            policy_kwargs={
                "net_arch": [256, 256],
                "features_extractor_class": MultiViewCombinedExtractor,
                "features_extractor_kwargs": {
                    "n_views": 3,
                    "frame_stack": args.frame_stack,
                    "visual_head_version": args.visual_head_version,
                },
                # With SB3's shared extractor, the critic optimizer explicitly
                # excludes extractor parameters.  That made the visual branch
                # depend almost entirely on the actor loss and allowed a
                # proprioceptive blind-search shortcut. Separate extractors
                # let the TD loss train the critic's visual representation.
                "share_features_extractor": args.share_features_extractor,
            },
            verbose=1,
            seed=args.seed,
            device=args.device,
            timing_stats=timing_stats,
        )
        print(f"rollout policy device: {next(model.policy.actor.parameters()).device}")
        checkpoint = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_actors, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=f"{args.task}_pixel_sac",
            save_replay_buffer=False,
            verbose=2,
        )
        visual_health = VisualHealthCallback(
            check_freq=max(args.checkpoint_freq // args.n_actors, 1),
            verbose=1,
        )
        evaluation = AsyncCpuEvalCallback(
            # AsyncCpuEvalCallback compares against the model's global
            # num_timesteps, which already includes all parallel envs. Unlike
            # SB3's callback-call based EvalCallback, this must not be divided
            # by n_actors.
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            checkpoint_dir=run_dir / "eval" / "checkpoints",
            eval_dir=run_dir / "eval",
            task=args.task,
            seed=args.seed,
            image_size=args.image_size,
            frame_stack=args.frame_stack,
            max_episode_steps=args.max_episode_steps,
            action_repeat=args.action_repeat,
            camera_scale=args.camera_scale,
            verbose=1,
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(
                [
                    PixelTaskMetricsCallback(task=args.task),
                    RolloutActionEntropyCallback(log_freq=args.entropy_log_freq),
                    TimingTensorBoardCallback(
                        timing_stats,
                        log_freq=args.time_log_freq,
                    ),
                    checkpoint,
                    visual_health,
                    evaluation,
                ]
            ),
            log_interval=10,
            tb_log_name=f"{args.task}_pixel_sac",
            progress_bar=False,
        )
        model.save(str(run_dir / "final_model"))
        print(f"training artifacts: {run_dir}")
    finally:
        train_env.close()


if __name__ == "__main__":
    main()
