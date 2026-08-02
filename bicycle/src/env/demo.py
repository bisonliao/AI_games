"""Command-line random/no-op environment demo for GUI or headless inspection."""

from __future__ import annotations

import argparse
import time

from .bicycle_env import BicycleBalanceEnv


def main() -> None:
    """Run no-op episodes to inspect passive bicycle behavior."""
    parser = argparse.ArgumentParser(
        description="Run no-op BicycleBalance-v0 episodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable the PyBullet GUI and run as fast as possible",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="number of episodes to run"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the first consecutive episode"
    )
    args = parser.parse_args()
    render_mode = None if args.headless else "human"
    env = BicycleBalanceEnv(render_mode=render_mode)
    try:
        for episode in range(args.episodes):
            _, _ = env.reset(seed=args.seed + episode)
            done = False
            total_reward = 0.0
            while not done:
                _, reward, terminated, truncated, info = env.step(1)
                total_reward += reward
                done = terminated or truncated
                if render_mode == "human":
                    time.sleep(1.0 / env.config.control_hz)
            print(
                f"episode={episode} outcome={info['outcome']} "
                f"distance={info['progress_m']:.2f} return={total_reward:.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
