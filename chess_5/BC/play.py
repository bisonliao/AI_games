"""Human-versus-BC/heuristic graphical Gomoku entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BC.agent import BCAgent
from BC.checkpoints import resolve_checkpoint
from BC.heuristic_agent import HeuristicAgent
from BC.sampling import (has_immediate_win_or_block, ranked_legal_actions,
                         rank_softmax_action)
from env import GomokuEnv


DEFAULT_CHECKPOINT_ROOT = Path(__file__).resolve().parent / "checkpoints"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="五子棋人机对弈：你执黑棋，BC 或启发式机器人执白棋。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python BC/play.py --run-name RUN_NAME --board-size 5
  python BC/play.py --run-name RUN_NAME --stage 02_bc_v1 --checkpoint-name latest.pt
  python BC/play.py --checkpoint BC/checkpoints/RUN_NAME/05_bc_v2/best.pt
  python BC/play.py --opponent heuristic --board-size 9
""",
    )
    parser.add_argument("--opponent", choices=("bc", "heuristic"), default="bc")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="Direct BC checkpoint path.")
    source.add_argument("--run-name", help="Pipeline run under --checkpoint-root.")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--stage", help="Stage such as 02_bc_v1 or 05_bc_v2; latest stage by default.")
    parser.add_argument("--checkpoint-name", choices=("best.pt", "latest.pt"), default="best.pt")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bc-policy", choices=("controlled", "greedy"), default="controlled")
    parser.add_argument("--opening-top-k", type=int, default=4)
    parser.add_argument("--opening-temperature", type=float, default=1.0)
    parser.add_argument("--stochastic-agent-moves", type=int, default=6)
    parser.add_argument("--play-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.opponent == "bc" and args.checkpoint is None and args.run_name is None:
        parser.error("BC opponent requires --checkpoint or --run-name")
    if args.opponent == "heuristic" and (args.checkpoint is not None or args.run_name is not None):
        parser.error("heuristic opponent does not use a checkpoint")
    if args.opening_top_k < 1 or args.opening_temperature < 0 or args.stochastic_agent_moves < 0:
        parser.error("opening parameters must be non-negative and top-k at least 1")
    return args


def controlled_bc_action(logits: np.ndarray, action_mask: np.ndarray, move_index: int,
                         rng: np.random.Generator, *, top_k: int = 4,
                         temperature: float = 1.0, stochastic_moves: int = 6,
                         force_greedy: bool = False) -> int:
    ranked = ranked_legal_actions(logits, action_mask)
    return rank_softmax_action(ranked, move_index, rng, top_k=top_k,
                               temperature=temperature, stochastic_moves=stochastic_moves,
                               force_greedy=force_greedy)


def wait_after_game() -> bool:
    import pygame
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            return False
        if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
            return True


def result_text(winner: int, opponent_name: str = "BC agent") -> str:
    if winner == 1:
        return "黑棋获胜，你赢了！"
    if winner == -1:
        return f"白棋获胜，{opponent_name} 赢了。"
    return "和棋。"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    checkpoint = None
    if args.opponent == "heuristic":
        agent: Any = HeuristicAgent(seed=args.play_seed)
        opponent_name = "启发式机器人"
    else:
        checkpoint = resolve_checkpoint(args.checkpoint_root, direct=args.checkpoint,
                                        run_name=args.run_name, stage=args.stage,
                                        checkpoint_name=args.checkpoint_name)
        agent = BCAgent(args.board_size, device=args.device)
        agent.load_checkpoint(checkpoint); agent.net.eval()
        opponent_name = "BC agent"
        print(f"Loaded checkpoint: {checkpoint}")
    rng = np.random.default_rng(args.play_seed)
    env = GomokuEnv(board_size=args.board_size, render_mode="human",
                    starting_player="black", illegal_action_mode="raise")
    print(f"你执黑棋，点击交叉点落子；{opponent_name}执白棋。")
    try:
        game = 1
        while True:
            print(f"\n第 {game} 局")
            obs, _ = env.reset(); done = False; bc_moves = 0
            while not done:
                player = int(obs["current_player"][0])
                if player == 1:
                    action = env.wait_for_human_action()
                elif args.opponent == "heuristic":
                    decision = agent.ranked_decision(obs["board"], player, obs["action_mask"],
                                                     args.opening_top_k)
                    action = rank_softmax_action(
                        decision.actions, int(np.count_nonzero(obs["board"] == player)), rng,
                        top_k=args.opening_top_k, temperature=args.opening_temperature,
                        stochastic_moves=args.stochastic_agent_moves,
                        force_greedy=decision.tactical)
                else:
                    logits = agent.action_logits(obs["board"], [player])[0]
                    action = controlled_bc_action(
                        logits, obs["action_mask"], bc_moves, rng,
                        top_k=args.opening_top_k,
                        temperature=(0.0 if args.bc_policy == "greedy" else args.opening_temperature),
                        stochastic_moves=args.stochastic_agent_moves,
                        force_greedy=has_immediate_win_or_block(
                            obs["board"], player, obs["action_mask"]),
                    )
                    bc_moves += 1
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            print(result_text(int(info["winner"]), opponent_name))
            print("点击棋盘或按任意键开始下一局；关闭窗口退出。")
            if not wait_after_game():
                break
            game += 1
    except KeyboardInterrupt:
        print("\n游戏已退出。")
    finally:
        env.close()


if __name__ == "__main__":
    main()
