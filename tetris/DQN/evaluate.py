from __future__ import annotations

import argparse
from pathlib import Path

from .evaluator import evaluate_checkpoint


def evaluate(
    checkpoint: str | Path,
    episodes: int = 5,
    max_steps: int = 100_000,
    device: str = "cpu",
    *,
    seed: int = 0,
    render: bool = False,
    render_fps: int = 10,
) -> dict[str, float | int | str]:
    return evaluate_checkpoint(
        checkpoint,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        device=device,
        render=render,
        render_fps=render_fps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="show a Pygame window during evaluation")
    parser.add_argument("--render-fps", type=int, default=10)
    args = parser.parse_args()
    print(
        evaluate(
            args.checkpoint,
            args.episodes,
            args.max_steps,
            args.device,
            seed=args.seed,
            render=args.render,
            render_fps=args.render_fps,
        )
    )


if __name__ == "__main__":
    main()
