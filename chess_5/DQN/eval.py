#!/usr/bin/env python3
"""DQN 离线评测命令行入口。

支持三种互斥的使用方式：

1. ``--checkpoint``：加载一个 checkpoint，分别执黑、执白对战启发式机器人。
2. ``--run-name``：依次加载指定 run 下的全部 checkpoint，每个 checkpoint
   分别执黑、执白对战启发式机器人。
3. ``--checkpoint-a`` 与 ``--checkpoint-b``：加载两个 checkpoint 直接对弈，
   先由 A 执黑、B 执白，再由 B 执黑、A 执白。

每种执棋组合默认并行评测 16 局，可通过 ``--games`` 调整。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .agent import DQNAgent, slice_obs
    from .evaluator import evaluate_checkpoint
    from .run_paths import named_directory, validate_run_name
except ImportError:
    from agent import DQNAgent, slice_obs
    from evaluator import evaluate_checkpoint
    from run_paths import named_directory, validate_run_name

from env import make_vector_env


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
        description="离线评测 DQN checkpoint 的棋力。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方式（以下三种模式互斥）：

  1. 单个 checkpoint 对战启发式机器人
     指定 --checkpoint。该 checkpoint 先执黑评测，再执白评测。

       python DQN/eval.py --checkpoint PATH --board-size 9

  2. 一个 run 的全部 checkpoint 对战启发式机器人
     指定 --run-name，可用 --history-dir 指定 checkpoint 历史目录根路径。
     每个 checkpoint 都分别执黑、执白评测。

       python DQN/eval.py --run-name RUN_NAME --history-dir DQN/history --board-size 9

  3. 两个 checkpoint 直接对弈
     --checkpoint-a 和 --checkpoint-b 必须同时指定。第一轮 A 执黑、B 执白，
     第二轮 B 执黑、A 执白，并分别输出 A、B 的胜率。

       python DQN/eval.py --checkpoint-a A.pt --checkpoint-b B.pt --board-size 9

公共规则：
  --games 表示每一种执棋组合的对局数，默认 16；不是两轮的总局数。
  三种模式均只使用 CPU，checkpoint 中的 DQN 使用 greedy 策略。
""",
    )
    source = parser.add_mutually_exclusive_group()
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
        "--checkpoint-a",
        type=Path,
        help="First checkpoint (A) in checkpoint-versus-checkpoint mode.",
    )
    parser.add_argument(
        "--checkpoint-b",
        type=Path,
        help="Second checkpoint (B) in checkpoint-versus-checkpoint mode.",
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
    pair_requested = args.checkpoint_a is not None or args.checkpoint_b is not None
    if pair_requested and (args.checkpoint_a is None or args.checkpoint_b is None):
        parser.error("--checkpoint-a and --checkpoint-b must be specified together")
    source_count = int(args.checkpoint is not None) + int(args.run_name is not None) + int(pair_requested)
    if source_count != 1:
        parser.error(
            "specify exactly one source: --checkpoint, --run-name, or both "
            "--checkpoint-a and --checkpoint-b"
        )
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


def _terminal_winner(infos: Dict[str, Any], index: int) -> int:
    """Read a completed game's winner from a SameStep vector environment."""
    mask = infos.get("_final_info")
    if mask is None or not mask[index]:
        raise RuntimeError("Completed evaluation game is missing final_info")
    final_infos = infos.get("final_info")
    if isinstance(final_infos, dict):
        winner_mask = final_infos.get("_winner")
        if "winner" in final_infos and (winner_mask is None or winner_mask[index]):
            return int(final_infos["winner"][index])
    elif final_infos is not None and final_infos[index] is not None:
        return int(final_infos[index].get("winner", 0))
    return 0


def evaluate_checkpoint_pair(
    checkpoint_a: Path,
    checkpoint_b: Path,
    *,
    board_size: int,
    num_games: int,
    seed: int,
    a_player: int,
) -> Dict[str, Any]:
    """Play two greedy checkpoint policies and return results by A/B identity."""
    if num_games < 1:
        raise ValueError("num_games must be at least 1")
    if a_player not in (1, -1):
        raise ValueError("a_player must be 1 (black) or -1 (white)")

    agent_a = DQNAgent(
        board_size, replay_size=1, min_replay_size=1, batch_size=1,
        device="cpu", seed=seed,
    )
    agent_b = DQNAgent(
        board_size, replay_size=1, min_replay_size=1, batch_size=1,
        device="cpu", seed=seed + 1,
    )
    metadata_a = agent_a.load_checkpoint(checkpoint_a, load_optimizer=False)
    metadata_b = agent_b.load_checkpoint(checkpoint_b, load_optimizer=False)
    for label, metadata in (("A", metadata_a), ("B", metadata_b)):
        checkpoint_board_size = int(metadata.get("board_size", board_size))
        if checkpoint_board_size != board_size:
            raise ValueError(
                f"Checkpoint {label} board size {checkpoint_board_size} "
                f"does not match {board_size}"
            )

    # make_vector_env固定使用SameStep自动重置：终局step返回的next_obs已经是
    # 下一局初始观测，上一局胜者保存在infos["final_info"]中。较早结束的槽位
    # 会继续新对局，因此用finished保证每个槽位只统计第一局。
    envs = make_vector_env(
        num_games, board_size=board_size, asynchronous=False, seed=seed,
    )
    a_wins = b_wins = draws = 0
    finished = np.zeros(num_games, dtype=np.bool_)
    started = time.perf_counter()
    try:
        obs, _ = envs.reset(seed=[seed + index for index in range(num_games)])
        while not np.all(finished):
            players = obs["current_player"].reshape(-1)
            actions = np.zeros(num_games, dtype=np.int64)
            a_indices = np.flatnonzero(players == a_player)
            b_indices = np.flatnonzero(players == -a_player)
            if len(a_indices):
                boards, current, masks = slice_obs(obs, a_indices)
                actions[a_indices] = agent_a.select_actions(
                    boards, current, masks, epsilon=0.0,
                )
            if len(b_indices):
                boards, current, masks = slice_obs(obs, b_indices)
                actions[b_indices] = agent_b.select_actions(
                    boards, current, masks, epsilon=0.0,
                )
            next_obs, _, terminated, truncated, infos = envs.step(actions)
            done = np.logical_or(terminated, truncated)
            for index in np.flatnonzero(done & ~finished):
                winner = _terminal_winner(infos, int(index))
                if winner == a_player:
                    a_wins += 1
                elif winner == -a_player:
                    b_wins += 1
                else:
                    draws += 1
                finished[index] = True
            obs = next_obs
    finally:
        envs.close()

    return {
        "games": num_games,
        "a_player": a_player,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "duration_seconds": time.perf_counter() - started,
    }


def format_pair_result(result: Dict[str, Any]) -> str:
    games = int(result["games"])
    a_wins = int(result["a_wins"])
    b_wins = int(result["b_wins"])
    draws = int(result["draws"])
    a_role, b_role = ("黑", "白") if int(result["a_player"]) == 1 else ("白", "黑")
    return (
        f"  A执{a_role} / B执{b_role}: "
        f"A胜={a_wins} ({a_wins / games:.2%}) "
        f"B胜={b_wins} ({b_wins / games:.2%}) "
        f"和={draws} ({draws / games:.2%}) "
        f"耗时={float(result['duration_seconds']):.2f}s"
    )


def _resolved_checkpoint(path: Path, label: str) -> Path:
    checkpoint = path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint {label} does not exist: {checkpoint}")
    return checkpoint


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoint_a is not None:
        checkpoint_a = _resolved_checkpoint(args.checkpoint_a, "A")
        checkpoint_b = _resolved_checkpoint(args.checkpoint_b, "B")
        print(
            f"双 checkpoint 对弈：棋盘={args.board_size}x{args.board_size}，"
            f"每种执棋方 {args.games} 局，CPU only。"
        )
        print(f"A: {checkpoint_a}")
        print(f"B: {checkpoint_b}")
        first = evaluate_checkpoint_pair(
            checkpoint_a, checkpoint_b, board_size=args.board_size,
            num_games=args.games, seed=args.seed, a_player=1,
        )
        print(format_pair_result(first))
        second = evaluate_checkpoint_pair(
            checkpoint_a, checkpoint_b, board_size=args.board_size,
            num_games=args.games, seed=args.seed, a_player=-1,
        )
        print(format_pair_result(second))
        return 0

    if args.checkpoint is not None:
        checkpoint = _resolved_checkpoint(args.checkpoint, "")
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
