"""人机对弈入口：渲染棋盘，让人类执黑连续对战DQN checkpoint或启发式机器人。"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import GomokuEnv

try:
    from .agent import DQNAgent
    from .heuristic_agent import HeuristicAgent
    from .run_paths import checkpoint_filename, named_directory, validate_run_name
except ImportError:
    from agent import DQNAgent
    from heuristic_agent import HeuristicAgent
    from run_paths import checkpoint_filename, named_directory, validate_run_name


DEFAULT_HISTORY_DIR = Path(__file__).resolve().parent / "history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Gomoku as black against a white DQN or heuristic agent."
    )
    parser.add_argument("--opponent", choices=("dqn", "heuristic"), default="dqn")
    parser.add_argument(
        "--run-name", default=None,
        help=(
            "Training name required for a DQN opponent. "
            "Pass an empty string to use legacy checkpoints directly under --history-dir."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint filename or sequence number under --history-dir. "
            "Defaults to the checkpoint with the largest sequence number."
        ),
    )
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    if args.opponent == "dqn" and args.run_name is None:
        parser.error("--run-name is required when --opponent=dqn")
    return args


def checkpoint_sort_key(path: Path) -> Any:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def history_checkpoints(history_dir: Path) -> List[Path]:
    if not history_dir.is_dir():
        return []
    return sorted(history_dir.glob("*.pt"), key=checkpoint_sort_key)


def resolve_checkpoint(history_dir: Path, run_name: str, requested: str | None) -> Path:
    legacy_layout = run_name == ""
    if legacy_layout:
        history_dir = history_dir.expanduser().resolve()
    else:
        run_name = validate_run_name(run_name)
        history_dir = named_directory(history_dir, run_name).resolve()
    if requested is None:
        checkpoints = history_checkpoints(history_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No .pt checkpoints found in {history_dir}")
        return checkpoints[-1]

    value = Path(requested).expanduser()
    if value.is_absolute():
        candidate = value
    elif requested.isdigit():
        filename = (
            f"{int(requested):06d}.pt"
            if legacy_layout
            else checkpoint_filename(run_name, int(requested))
        )
        candidate = history_dir / filename
    else:
        candidate = history_dir / value
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")
    return candidate


def build_agent(board_size: int, device: str, checkpoint: Path) -> DQNAgent:
    # Playing does not use replay or optimization, so keep their allocations tiny.
    agent = DQNAgent(
        board_size,
        replay_size=1,
        min_replay_size=1,
        batch_size=1,
        device=device,
    )
    metadata = agent.load_checkpoint(checkpoint, load_optimizer=False)
    checkpoint_board_size = int(metadata.get("board_size", board_size))
    if checkpoint_board_size != board_size:
        raise ValueError(
            f"Checkpoint board size is {checkpoint_board_size}x{checkpoint_board_size}, "
            f"but --board-size is {board_size}x{board_size}."
        )
    return agent


def wait_after_game() -> bool:
    """Return True to start another game, or False when the window is closed."""
    import pygame
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            return False
        if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
            return True


def result_text(winner: int, opponent_name: str = "DQN agent") -> str:
    if winner == 1:
        return "黑棋获胜，你赢了！"
    if winner == -1:
        return f"白棋获胜，{opponent_name} 赢了。"
    return "和棋。"


def main() -> None:
    args = parse_args()
    if args.opponent == "heuristic":
        checkpoint = None
        agent: Any = HeuristicAgent(seed=0)
        opponent_name = "启发式机器人"
    else:
        checkpoint = resolve_checkpoint(args.history_dir, args.run_name, args.checkpoint)
        agent = build_agent(args.board_size, args.device, checkpoint)
        opponent_name = "DQN agent"
    env = GomokuEnv(
        board_size=args.board_size,
        render_mode="human",
        starting_player="black",
        illegal_action_mode="raise",
    )

    if checkpoint is not None:
        print(f"Loaded checkpoint: {checkpoint}")
    print(f"你执黑棋，点击棋盘上的交叉点落子；{opponent_name}执白棋。")
    try:
        game_number = 1
        while True:
            print(f"\n第 {game_number} 局")
            obs, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                current_player = int(obs["current_player"].item())
                if current_player == 1:
                    action = env.wait_for_human_action()
                else:
                    action = int(
                        agent.select_actions(
                            obs["board"][None, ...],
                            np.array([-1], dtype=np.int8),
                            obs["action_mask"][None, ...],
                            epsilon=0.0,
                        )[0]
                    )
                obs, _, terminated, truncated, info = env.step(action)

            print(result_text(int(info["winner"]), opponent_name))
            print("点击棋盘或按任意键开始下一局；关闭窗口退出。")
            if not wait_after_game():
                break
            game_number += 1
    except KeyboardInterrupt:
        print("\n游戏已退出。")
    finally:
        env.close()


if __name__ == "__main__":
    main()
