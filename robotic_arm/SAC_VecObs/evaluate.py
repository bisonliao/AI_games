"""Evaluate a trained SB3 SAC policy, optionally in the PyBullet GUI."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .utils import make_env_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["reach", "pick_place"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=8)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.fps <= 0.0:
        parser.error("--fps must be positive")
    return args


def find_vecnormalize(checkpoint: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit
    candidates = [
        checkpoint.parent / "vecnormalize.pkl",
        checkpoint.parent.parent / "vecnormalize.pkl",
    ]
    stem = checkpoint.stem
    if stem.endswith("_steps"):
        prefix, steps, _ = stem.rsplit("_", 2)
        candidates.insert(0, checkpoint.parent / f"{prefix}_vecnormalize_{steps}_steps.pkl")
    return next((path for path in candidates if path.exists()), None)


def main() -> None:
    args = parse_args()
    env = DummyVecEnv(
        [
            make_env_factory(
                task=args.task,
                rank=0,
                seed=args.seed,
                max_episode_steps=args.max_episode_steps,
                action_repeat=args.action_repeat,
                render_mode="human" if args.gui else None,
            )
        ]
    )
    normalizer_path = find_vecnormalize(args.checkpoint, args.vecnormalize)
    if normalizer_path is not None:
        wrapped_env = VecNormalize.load(str(normalizer_path), env)
        wrapped_env.training = False
        wrapped_env.norm_reward = False
    else:
        print("warning: VecNormalize statistics not found; using raw observations")
        wrapped_env = env

    try:
        model = SAC.load(str(args.checkpoint), env=wrapped_env, device=args.device)
        observation = wrapped_env.reset()
        completed = 0
        episode_reward = 0.0
        while completed < args.episodes:
            frame_start = time.perf_counter()
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, rewards, dones, infos = wrapped_env.step(action)
            episode_reward += float(rewards[0])
            if dones[0]:
                info = infos[0]
                failure_reason = info.get("failure_reason", "")
                failure_text = f" failure_reason={failure_reason}" if failure_reason else ""
                completed += 1
                print(
                    f"episode={completed} success={info.get('success', False)} "
                    f"stage={info.get('stage', 'unknown')} "
                    f"steps={info.get('episode', {}).get('l', '?')} "
                    f"reward={episode_reward:.3f}{failure_text}"
                )
                episode_reward = 0.0
            if args.gui:
                remaining = (1.0 / args.fps) - (time.perf_counter() - frame_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        wrapped_env.close()


if __name__ == "__main__":
    main()
