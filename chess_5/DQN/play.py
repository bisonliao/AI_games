"""五子棋人机对弈命令行入口。

人类固定执黑，程序渲染棋盘并让白方使用以下两种互斥对手之一：

1. ``--opponent dqn``：加载指定训练 run 的 DQN checkpoint。必须同时提供
   ``--run-name``；可用 ``--checkpoint`` 选择序号或文件名，省略时加载该 run
   序号最大的 checkpoint。
2. ``--opponent heuristic``：使用启发式机器人，不需要 run 或 checkpoint。

DQN 默认使用开局受控采样、随后转为 greedy 的策略，也可通过
``--dqn-policy greedy`` 在整局中使用确定性 greedy 策略。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import GomokuEnv

try:
    from .agent import DQNAgent, encode_boards
    from .heuristic_agent import HeuristicAgent
    from .run_paths import checkpoint_filename, named_directory, validate_run_name
except ImportError:
    from agent import DQNAgent, encode_boards
    from heuristic_agent import HeuristicAgent
    from run_paths import checkpoint_filename, named_directory, validate_run_name


DEFAULT_HISTORY_DIR = Path(__file__).resolve().parent / "history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="五子棋人机对弈：你执黑棋，DQN 或启发式机器人执白棋。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方式：

  1. 对战某个 run 的最新 DQN checkpoint
     --opponent 默认为 dqn，因此可以省略；--run-name 必须指定。

       python DQN/play.py --run-name RUN_NAME --board-size 9

  2. 对战某个 run 的指定 DQN checkpoint
     --checkpoint 可写 checkpoint 序号或该 run 目录下的完整文件名。

       python DQN/play.py --run-name RUN_NAME --checkpoint 3 --board-size 9
       python DQN/play.py --run-name RUN_NAME --checkpoint RUN_NAME_000003.pt --board-size 9

  3. 对战启发式机器人
     该模式不加载 checkpoint，不需要 --run-name、--checkpoint 或 --device。

       python DQN/play.py --opponent heuristic --board-size 9

  4. 使用旧版平铺 checkpoint 目录
     当 checkpoint 直接位于 --history-dir、没有 run 子目录时，传入空 run-name。

       python DQN/play.py --run-name "" --history-dir DQN/history --checkpoint 3

DQN 策略：
  --dqn-policy controlled（默认）仅在每局开局阶段从 Q 值靠前的合法动作中
  受控采样；--opening-top-k、--opening-temperature、--stochastic-agent-moves
  和 --play-seed 只影响该模式。遇到一步必杀或必堵局面时改用网络 greedy。

  --dqn-policy greedy 在整局中选择 Q 值最大的合法动作，适合复现确定性对局。
""",
    )
    parser.add_argument(
        "--opponent", choices=("dqn", "heuristic"), default="dqn",
        help="白方对手类型：DQN checkpoint（默认）或启发式机器人。",
    )
    parser.add_argument(
        "--run-name", default=None,
        help=(
            "DQN对手所属的训练run名称；DQN模式必填。传空字符串表示直接从"
            "--history-dir读取旧版平铺checkpoint。"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "run目录下的checkpoint序号或文件名；省略时使用序号最大的checkpoint。"
        ),
    )
    parser.add_argument(
        "--history-dir", type=Path, default=DEFAULT_HISTORY_DIR,
        help=f"Checkpoint历史目录根路径（默认：{DEFAULT_HISTORY_DIR}）。",
    )
    parser.add_argument("--board-size", type=int, default=5, help="棋盘边长，默认5。")
    parser.add_argument(
        "--device", type=str, default="auto",
        help="DQN推理设备，默认auto（CUDA可用时优先使用CUDA）。",
    )
    parser.add_argument(
        "--dqn-policy",
        choices=("controlled", "greedy"),
        default="controlled",
        help="DQN人机对弈策略：默认在开局做受控采样；greedy用于复现确定性对局。",
    )
    parser.add_argument(
        "--opening-top-k",
        type=int,
        default=3,
        help="受控采样只考虑Q值最高的前K个合法动作，默认3。",
    )
    parser.add_argument(
        "--opening-temperature",
        type=float,
        default=0.10,
        help="开局Softmax初始温度，随后随DQN落子次数线性衰减，默认0.10。",
    )
    parser.add_argument(
        "--stochastic-agent-moves",
        type=int,
        default=6,
        help="每局前多少次DQN落子允许受控采样，之后恢复greedy，默认6。",
    )
    parser.add_argument(
        "--play-seed",
        type=int,
        default=0,
        help="仅控制人机对弈受控采样的随机种子，默认0。",
    )
    args = parser.parse_args()
    if args.opponent == "dqn" and args.run_name is None:
        parser.error("--run-name is required when --opponent=dqn")
    if args.opening_top_k < 1:
        parser.error("--opening-top-k must be at least 1")
    if args.opening_temperature < 0.0:
        parser.error("--opening-temperature cannot be negative")
    if args.stochastic_agent_moves < 0:
        parser.error("--stochastic-agent-moves cannot be negative")
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


def dqn_action_values(
    agent: DQNAgent,
    board: np.ndarray,
    current_player: int,
) -> np.ndarray:
    """Return one board's raw DQN Q values without changing the network mode."""
    states = encode_boards(board[None, ...], np.asarray([current_player], dtype=np.int8))
    was_training = agent.online_net.training
    agent.online_net.eval()
    with torch.inference_mode():
        q_values = agent.online_net(torch.as_tensor(states, device=agent.device))[0]
    if was_training:
        agent.online_net.train()
    return q_values.cpu().numpy()


def _has_five_from(
    board: np.ndarray,
    row: int,
    col: int,
    player: int,
) -> bool:
    """Check whether the newly placed stone completes five in a row."""
    size = board.shape[0]
    for row_step, col_step in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for direction in (-1, 1):
            next_row = row + direction * row_step
            next_col = col + direction * col_step
            while (
                0 <= next_row < size
                and 0 <= next_col < size
                and board[next_row, next_col] == player
            ):
                count += 1
                next_row += direction * row_step
                next_col += direction * col_step
        if count >= 5:
            return True
    return False


def has_immediate_win_or_block(
    board: np.ndarray,
    current_player: int,
    action_mask: np.ndarray,
) -> bool:
    """Return whether either side has an immediate winning move.

    A current-player win is a 必杀 position; an opponent win is a 必堵
    position.  Detection only disables sampling and never chooses the move.
    """
    position = np.asarray(board, dtype=np.int8)
    if position.ndim != 2 or position.shape[0] != position.shape[1]:
        raise ValueError("board must be a square 2-D array")
    mask = np.asarray(action_mask, dtype=bool).reshape(-1)
    if mask.size != position.size:
        raise ValueError("action_mask size must match the board")
    legal = np.flatnonzero(mask & (position.reshape(-1) == 0))
    size = position.shape[0]
    candidate = position.copy()
    for player in (int(current_player), -int(current_player)):
        for action in legal:
            row, col = divmod(int(action), size)
            candidate[row, col] = player
            is_win = _has_five_from(candidate, row, col, player)
            candidate[row, col] = 0
            if is_win:
                return True
    return False


def controlled_dqn_action(
    q_values: np.ndarray,
    action_mask: np.ndarray,
    agent_move_index: int,
    rng: np.random.Generator,
    *,
    top_k: int = 3,
    initial_temperature: float = 0.10,
    stochastic_moves: int = 6,
) -> int:
    """Sample among the DQN's best legal moves, decaying to greedy by move count.

    This function deliberately contains no handcrafted win/block rules: every
    candidate and preference comes from the checkpoint's own Q values.
    """
    values = np.asarray(q_values, dtype=np.float64).reshape(-1)
    mask = np.asarray(action_mask, dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError("q_values and action_mask must have the same flattened shape")
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise RuntimeError("No legal actions available")
    if not np.all(np.isfinite(values[legal])):
        raise RuntimeError("DQN produced non-finite Q values for legal actions")

    ranked = legal[np.argsort(-values[legal], kind="stable")]
    greedy_action = int(ranked[0])
    if (
        top_k <= 1
        or initial_temperature <= 0.0
        or stochastic_moves <= 0
        or agent_move_index >= stochastic_moves
    ):
        return greedy_action

    # Both temperature and candidate count shrink as this agent makes moves.
    # At stochastic_moves the caller receives a completely greedy policy.
    decay = 1.0 - max(0, agent_move_index) / float(stochastic_moves)
    temperature = initial_temperature * decay
    active_k = 1 + int(np.ceil((min(top_k, legal.size) - 1) * decay))
    candidates = ranked[:active_k]
    logits = (values[candidates] - values[candidates].max()) / temperature
    weights = np.exp(logits)
    probabilities = weights / weights.sum()
    return int(rng.choice(candidates, p=probabilities))


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
    play_rng = np.random.default_rng(args.play_seed)
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
        if args.dqn_policy == "controlled":
            print(
                "DQN受控随机："
                f"前{args.stochastic_agent_moves}次落子，"
                f"Top-{args.opening_top_k}，"
                f"初始温度{args.opening_temperature:g}，随后线性衰减至greedy；"
                "出现一步必杀或必堵局面时强制使用神经网络greedy动作。"
            )
        else:
            print("DQN使用纯greedy策略，重复局面将选择相同动作。")
    print(f"你执黑棋，点击棋盘上的交叉点落子；{opponent_name}执白棋。")
    try:
        game_number = 1
        while True:
            print(f"\n第 {game_number} 局")
            obs, _ = env.reset()
            terminated = truncated = False
            dqn_move_index = 0
            while not (terminated or truncated):
                current_player = int(obs["current_player"].item())
                if current_player == 1:
                    action = env.wait_for_human_action()
                elif args.opponent == "dqn":
                    q_values = dqn_action_values(agent, obs["board"], current_player)
                    force_greedy = (
                        args.dqn_policy == "greedy"
                        or has_immediate_win_or_block(
                            obs["board"], current_player, obs["action_mask"]
                        )
                    )
                    action = controlled_dqn_action(
                        q_values,
                        obs["action_mask"],
                        dqn_move_index,
                        play_rng,
                        top_k=args.opening_top_k,
                        initial_temperature=args.opening_temperature,
                        stochastic_moves=(
                            0 if force_greedy else args.stochastic_agent_moves
                        ),
                    )
                    dqn_move_index += 1
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
