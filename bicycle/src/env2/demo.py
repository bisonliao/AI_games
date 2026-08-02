"""Command-line steering-environment demo for GUI or headless inspection."""

from __future__ import annotations

import argparse
import time

from .baseline import pd_action
from .bicycle_env import BicycleSteeringEnv


def main() -> None:
    """Run scripted steering episodes so the balancing mechanism is visible."""
    parser = argparse.ArgumentParser(
        description="Run scripted BicycleSteeringBalance-v0 episodes",
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
    env = BicycleSteeringEnv(render_mode=None if args.headless else "human")
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            while True:
                observation, reward, terminated, truncated, info = env.step(
                    pd_action(observation)
                )
                total_reward += reward
                if not args.headless:
                    time.sleep(1.0 / env.config.control_hz)
                if terminated or truncated:
                    break
            print(
                f"episode={episode} outcome={info['outcome']} "
                f"distance={info['progress_m']:.2f} return={total_reward:.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
