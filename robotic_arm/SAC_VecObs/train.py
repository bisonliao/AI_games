"""Train reach or pick-place with SB3 SAC and parallel PyBullet actors."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .callbacks import TaskMetricsCallback
from .utils import make_env_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="reach", choices=["reach", "pick_place"])
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
    )
    parser.add_argument("--n-actors", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="SB3/PyTorch device")
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=Path("SAC_VecObs/runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", type=Path, default=None, help="optional SB3 .zip checkpoint")
    parser.add_argument("--resume-vecnormalize", type=Path, default=None)
    parser.add_argument("--resume-replay-buffer", type=Path, default=None)
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument(
        "--start-method",
        choices=["forkserver", "spawn", "fork"],
        default="forkserver",
        help="multiprocessing method for actor processes",
    )
    args = parser.parse_args()
    if args.n_actors <= 0:
        parser.error("--n-actors must be positive")
    for name in (
        "max_episode_steps",
        "action_repeat",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "train_freq",
        "gradient_steps",
        "eval_freq",
        "eval_episodes",
        "checkpoint_freq",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.total_timesteps is None:
        args.total_timesteps = 1_000_000 if args.task == "reach" else 10_000_000
    return args


def build_run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    for child in ("checkpoints", "best_model", "eval", "monitor", "tensorboard"):
        (run_dir / child).mkdir()
    return run_dir


def main() -> None:
    args = parse_args()
    run_dir = build_run_dir(args)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            file,
            indent=2,
            ensure_ascii=False,
        )

    actor_factories = [
        make_env_factory(
            task=args.task,
            rank=rank,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            action_repeat=args.action_repeat,
            monitor_dir=run_dir / "monitor",
        )
        for rank in range(args.n_actors)
    ]
    if args.n_actors == 1:
        actor_env = DummyVecEnv(actor_factories)
    else:
        actor_env = SubprocVecEnv(actor_factories, start_method=args.start_method)

    if args.resume_vecnormalize is not None:
        train_env = VecNormalize.load(str(args.resume_vecnormalize), actor_env)
        train_env.training = True
        train_env.norm_reward = False
    else:
        train_env = VecNormalize(
            actor_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    eval_base = DummyVecEnv(
        [
            make_env_factory(
                task=args.task,
                rank=0,
                seed=args.seed + 100_000,
                max_episode_steps=args.max_episode_steps,
                action_repeat=args.action_repeat,
            )
        ]
    )
    eval_env = VecNormalize(
        eval_base,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    try:
        if args.resume is not None:
            model = SAC.load(
                str(args.resume),
                env=train_env,
                device=args.device,
                tensorboard_log=str(run_dir / "tensorboard"),
            )
            if args.resume_replay_buffer is not None:
                model.load_replay_buffer(str(args.resume_replay_buffer))
            reset_num_timesteps = False
        else:
            model = SAC(
                "MlpPolicy",
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
                policy_kwargs={"net_arch": [256, 256]},
                verbose=1,
                seed=args.seed,
                device=args.device,
            )
            reset_num_timesteps = True

        # Callback frequencies count vector-environment calls, whereas the CLI
        # expresses global transitions.  Divide by actor count accordingly.
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_actors, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=f"{args.task}_sac",
            save_replay_buffer=args.save_replay_buffer,
            save_vecnormalize=True,
            verbose=2,
        )
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(run_dir / "eval"),
            eval_freq=max(args.eval_freq // args.n_actors, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1,
        )
        callbacks = CallbackList(
            [TaskMetricsCallback(), checkpoint_callback, eval_callback]
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name=f"{args.task}_sac",
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        model.save(str(run_dir / "final_model"))
        train_env.save(str(run_dir / "vecnormalize.pkl"))
        if args.save_replay_buffer:
            model.save_replay_buffer(str(run_dir / "replay_buffer.pkl"))
        print(f"training artifacts: {run_dir}")
    finally:
        eval_env.close()
        train_env.close()


if __name__ == "__main__":
    main()
