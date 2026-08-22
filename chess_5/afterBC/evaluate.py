"""Standalone CLI for afterBC checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import DEFAULT_BC_CHECKPOINT, validate_bc_checkpoint
from .evaluator import evaluate_checkpoint, write_evaluation_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an afterBC white DQN checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument(
        "--stochastic-games", "--statistical-games",
        dest="stochastic_games", type=int, default=128,
    )
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.stochastic_games < 1:
        parser.error("--stochastic-games must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_bc_checkpoint(args.bc_checkpoint)
    result = evaluate_checkpoint(
        args.checkpoint, args.bc_checkpoint,
        stochastic_games=args.stochastic_games, seed=args.seed,
    )
    if args.output is not None:
        write_evaluation_json(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
