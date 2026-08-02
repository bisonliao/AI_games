"""Bang-bang roll PD baseline and 100-seed environment difficulty gate."""

from __future__ import annotations

import argparse
import math

from .bicycle_env import BicycleBalanceEnv


def pd_action(observation) -> int:
    """Map exact roll/roll-rate observations to negative, zero, or positive torque."""
    roll = math.atan2(float(observation[0]), float(observation[1]))
    roll_rate = float(observation[2]) * 10.0
    command = roll + 0.35 * roll_rate
    if command > 0.01:
        return 2
    if command < -0.01:
        return 0
    return 1


def run_gate(episodes: int = 100, seed_start: int = 0) -> tuple[float, float]:
    """Return success rates for the PD controller and the no-op policy."""
    pd_successes = 0
    idle_successes = 0
    for controller, counter_name in ((pd_action, "pd"), (lambda _obs: 1, "idle")):
        successes = 0
        env = BicycleBalanceEnv()
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
        if counter_name == "pd":
            pd_successes = successes
        else:
            idle_successes = successes
    return pd_successes / episodes, idle_successes / episodes


def main() -> None:
    """Run both baselines and fail when the configured difficulty gates regress."""
    parser = argparse.ArgumentParser(
        description="Run bicycle task solvability and non-triviality gates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="episodes per baseline controller"
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first seed in the consecutive gate seed set",
    )
    args = parser.parse_args()
    pd_rate, idle_rate = run_gate(args.episodes, args.seed_start)
    print(f"pd_success_rate={pd_rate:.3f} idle_success_rate={idle_rate:.3f}")
    if pd_rate < 0.95 or idle_rate >= 0.10:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
