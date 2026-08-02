"""Bang-bang roll-PD baseline for the steering-only bicycle task."""

from __future__ import annotations

import argparse
import math

from .bicycle_env import BicycleSteeringEnv


def pd_action(observation) -> int:
    """Steer toward the predicted fall direction using roll and roll rate."""
    roll = math.atan2(float(observation[0]), float(observation[1]))
    roll_rate = float(observation[2]) * 10.0
    command = roll + 0.30 * roll_rate
    if command > 0.002:
        return 2  # Positive roll is a fall toward -Y, so turn right.
    if command < -0.002:
        return 1
    return 0


def run_gate(episodes: int = 100, seed_start: int = 0) -> tuple[float, float]:
    """Return steering-PD and no-action success rates on identical seeds."""
    rates: list[float] = []
    for controller in (pd_action, lambda _observation: 0):
        successes = 0
        env = BicycleSteeringEnv()
        try:
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed_start + episode)
                while True:
                    observation, _, terminated, truncated, info = env.step(
                        controller(observation)
                    )
                    if terminated or truncated:
                        break
                successes += int(info["success"])
        finally:
            env.close()
        rates.append(successes / episodes)
    return rates[0], rates[1]


def main() -> None:
    """Print the scripted and passive success rates for a chosen seed set."""
    parser = argparse.ArgumentParser(
        description="Measure steering-task scripted and passive success rates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="episodes per baseline controller"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first seed in the consecutive evaluation set",
    )
    args = parser.parse_args()
    pd_rate, idle_rate = run_gate(args.episodes, args.seed_start)
    print(f"pd_success_rate={pd_rate:.3f} idle_success_rate={idle_rate:.3f}")


if __name__ == "__main__":
    main()
