"""Evaluate a multi-view pixel SAC checkpoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import DEFAULT_FRAME_STACK, DEFAULT_IMAGE_SIZE
from .policy import MultiViewCombinedExtractor
from .utils import make_env_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["reach", "pick_place"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument("--camera-scale", type=float, default=0.8)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--action-repeat", type=int, default=8)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()
    if args.episodes <= 0 or args.image_size <= 0 or args.frame_stack <= 0:
        parser.error("episodes, image-size, and frame-stack must be positive")
    if args.fps <= 0.0:
        parser.error("fps must be positive")

    env = DummyVecEnv([
        make_env_factory(
            task=args.task,
            rank=0,
            seed=args.seed,
            image_size=args.image_size,
            frame_stack=args.frame_stack,
            max_episode_steps=args.max_episode_steps,
            action_repeat=args.action_repeat,
            camera_scale=args.camera_scale,
            render_mode="human" if args.gui else None,
        )
    ])
    try:
        model = SAC.load(str(args.checkpoint), env=env, device=args.device)
        observation = env.reset()
        completed = 0
        reward = 0.0
        while completed < args.episodes:
            start = time.perf_counter()
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, rewards, dones, infos = env.step(action)
            reward += float(rewards[0])
            if dones[0]:
                info = infos[0]
                completed += 1
                reason = info.get("failure_reason", "")
                suffix = f" failure_reason={reason}" if reason else ""
                print(
                    f"episode={completed} success={info.get('success', False)} "
                    f"stage={info.get('stage', 'unknown')} "
                    f"steps={info.get('episode', {}).get('l', '?')} "
                    f"reward={reward:.3f}{suffix}"
                )
                reward = 0.0
            if args.gui:
                remaining = 1.0 / args.fps - (time.perf_counter() - start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        env.close()


if __name__ == "__main__":
    main()
