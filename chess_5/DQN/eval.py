#!/usr/bin/env python3
"""离线评测入口：按路径或run-name加载checkpoint，分别执黑、执白对战启发式机器人并输出结果。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .evaluator import evaluate_checkpoint
    from .run_paths import named_directory, validate_run_name
except ImportError:
    from evaluator import evaluate_checkpoint
    from run_paths import named_directory, validate_run_name


DEFAULT_HISTORY_DIR = Path(__file__).resolve().parent / "history"


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    """Sort by the trailing checkpoint sequence number, then by filename."""
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def checkpoints_for_run(history_dir: Path, run_name: str) -> List[Path]:
    run_directory = named_directory(history_dir, validate_run_name(run_name)).resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Run history directory does not exist: {run_directory}")
    checkpoints = sorted(run_directory.glob("*.pt"), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found in {run_directory}")
    return checkpoints


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DQN checkpoint(s) as white against a heuristic agent playing black."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to one checkpoint; absolute paths outside the project are supported.",
    )
    source.add_argument(
        "--run-name",
        help="Evaluate every .pt checkpoint under <history-dir>/<run-name> in order.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help=f"Checkpoint history root (default: {DEFAULT_HISTORY_DIR}).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=16,
        help="Number of games for each side per checkpoint (default: 16).",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=5,
        help="Board size used by the checkpoints (default: 5).",
    )
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error("--games must be at least 1")
    if args.board_size < 5:
        parser.error("--board-size must be at least 5")
    return args


def format_result(role: str, result: Dict[str, Any]) -> str:
    games = int(result["games"])
    wins = int(result["wins"])
    losses = int(result["losses"])
    draws = int(result["draws"])
    win_rate = wins / games if games else 0.0
    return (
        f"  DQN执{role}: W={wins} L={losses} D={draws} "
        f"胜率={win_rate:.2%} 耗时={float(result['duration_seconds']):.2f}s"
    )


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        checkpoints = [checkpoint]
    else:
        checkpoints = checkpoints_for_run(args.history_dir, args.run_name)

    print(
        f"评测设置：每个 checkpoint 分别执黑、执白与启发式机器人对弈；"
        f"棋盘={args.board_size}x{args.board_size}，每种执棋方 {args.games} 局，CPU only。"
    )

    failures = 0
    for index, checkpoint in enumerate(checkpoints, start=1):
        if len(checkpoints) > 1:
            print(f"[{index}/{len(checkpoints)}] ", end="", flush=True)
        print(checkpoint)
        try:
            black_result = evaluate_checkpoint(
                checkpoint,
                board_size=args.board_size,
                num_games=args.games,
                seed=args.seed,
                agent_player=1,
            )
            white_result = evaluate_checkpoint(
                checkpoint,
                board_size=args.board_size,
                num_games=args.games,
                seed=args.seed,
                agent_player=-1,
            )
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {exc}")
            continue
        print(format_result("黑", black_result))
        print(format_result("白", white_result))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n评测已中断。", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
