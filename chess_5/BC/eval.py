"""Deterministic two-color evaluation of a BC checkpoint against the expert."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BC.agent import BCAgent
from DQN.heuristic_agent import HeuristicAgent
from env import GomokuEnv


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _play(task: dict[str, Any]) -> dict[str, int]:
    import torch
    torch.set_num_threads(1)
    agent = BCAgent(task["board_size"], device="cpu")
    agent.load_checkpoint(Path(task["checkpoint"])); agent.net.eval()
    heuristic = HeuristicAgent(seed=task["seed"], max_candidates=task["max_candidates"])
    wins = losses = draws = moves = illegal = 0
    for game in range(task["games"]):
        env = GomokuEnv(board_size=task["board_size"], starting_player="black", illegal_action_mode="raise")
        obs, _ = env.reset(seed=task["seed"] + game); done = False
        while not done:
            player = int(obs["current_player"][0])
            actor = agent if player == task["agent_player"] else heuristic
            action = int(actor.select_actions(obs["board"], [player], obs["action_mask"])[0])
            if not obs["action_mask"][action]: illegal += 1
            obs, _, terminated, truncated, info = env.step(action)
            moves += 1; done = terminated or truncated
        winner = int(info["winner"])
        if winner == task["agent_player"]: wins += 1
        elif winner == -task["agent_player"]: losses += 1
        else: draws += 1
        env.close()
    return {"wins": wins, "losses": losses, "draws": draws, "moves": moves,
            "illegal_actions": illegal, "games": task["games"]}


def evaluate(checkpoint: Path, board_size: int, games: int, seed: int,
             workers: int, max_candidates: int, agent_player: int) -> dict[str, Any]:
    workers = min(workers, games)
    counts = [games // workers + (i < games % workers) for i in range(workers)]
    tasks = [{"checkpoint": str(checkpoint), "board_size": board_size, "games": count,
              "seed": seed + i * 1_000_003, "max_candidates": max_candidates,
              "agent_player": agent_player} for i, count in enumerate(counts)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_play, tasks))
    result = {key: sum(part[key] for part in parts)
              for key in ("wins", "losses", "draws", "moves", "illegal_actions", "games")}
    n = result["games"]; scores = result["wins"] + 0.5 * result["draws"]; mean = scores / n
    variance = ((result["wins"] * (1 - mean) ** 2 + result["draws"] * (0.5 - mean) ** 2
                 + result["losses"] * mean ** 2) / max(1, n - 1))
    margin = 1.959963984540054 * math.sqrt(variance / n)
    result.update({"agent_player": "black" if agent_player == 1 else "white",
                   "win_rate": result["wins"] / n, "loss_rate": result["losses"] / n,
                   "draw_rate": result["draws"] / n, "score_rate": mean,
                   "win_rate_ci95": _wilson(result["wins"], n),
                   "score_rate_ci95": (max(0.0, mean - margin), min(1.0, mean + margin)),
                   "average_game_length": result["moves"] / n})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BC against the heuristic expert.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--games-per-color", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tb-dir", type=Path, default=None,
                        help="Exact TensorBoard directory for this pipeline step.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games_per_color < 1 or args.workers < 1:
        raise ValueError("games and workers must be positive")
    results = [evaluate(args.checkpoint, args.board_size, args.games_per_color,
                        args.seed, args.workers, args.max_candidates, color) for color in (1, -1)]
    output = {"checkpoint": str(args.checkpoint.resolve()), "board_size": args.board_size,
              "seed": args.seed, "games_per_color": args.games_per_color, "results": results,
              "passes_45_55": all(0.45 <= item["score_rate"] <= 0.55 for item in results)}
    rendered = json.dumps(output, ensure_ascii=False, indent=2); print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered + "\n")
    if args.tb_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(args.tb_dir))
        writer.add_text("Pipeline/config", json.dumps({
            "checkpoint": str(args.checkpoint.resolve()), "board_size": args.board_size,
            "games_per_color": args.games_per_color, "seed": args.seed,
        }, ensure_ascii=False, indent=2), 0)
        for result in results:
            color = result["agent_player"].capitalize()
            for metric in ("win_rate", "loss_rate", "draw_rate", "score_rate",
                           "average_game_length", "illegal_actions"):
                writer.add_scalar(f"{color}/{metric}", result[metric], 0)
        writer.add_scalar("Evaluation/passes_45_55", float(output["passes_45_55"]), 0)
        writer.close()


if __name__ == "__main__":
    main()
