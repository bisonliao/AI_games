from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DQN.combined import evaluate_combined
from DQN.runtime import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen high/low BasicMath policies through raw actions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--high-checkpoint", type=Path, required=True)
    parser.add_argument("--low-checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_001)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Maximum raw policy decisions per second; 0 means unthrottled",
    )
    args = parser.parse_args()

    high_model, high_checkpoint = load_checkpoint(args.high_checkpoint)
    low_model, low_checkpoint = load_checkpoint(args.low_checkpoint)
    if high_checkpoint.get("task") != "high":
        raise ValueError("--high-checkpoint is not a high-level checkpoint")
    if low_checkpoint.get("task") != "low":
        raise ValueError("--low-checkpoint is not a low-level checkpoint")

    metrics = evaluate_combined(
        high_model,
        low_model,
        episodes=args.episodes,
        seed=args.seed,
        gui=args.gui,
        fps=args.fps,
    )
    print(f"high_checkpoint={args.high_checkpoint}")
    print(f"low_checkpoint={args.low_checkpoint}")
    for name, value in metrics.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
