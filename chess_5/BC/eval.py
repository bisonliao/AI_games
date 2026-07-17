"""BC checkpoint evaluation against the local expert or another BC policy."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BC.agent import BCAgent
from BC.checkpoints import checkpoints_for_run, resolve_checkpoint
from BC.heuristic_agent import HeuristicAgent
from BC.sampling import rank_softmax_action
from env import GomokuEnv


DEFAULT_CHECKPOINT_ROOT = Path(__file__).resolve().parent / "checkpoints"


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _summarize(counts: dict[str, int], agent_player: int, seconds: float) -> dict[str, Any]:
    n = counts["games"]
    mean = (counts["wins"] + 0.5 * counts["draws"]) / n
    variance = ((counts["wins"] * (1 - mean) ** 2 + counts["draws"] * (0.5 - mean) ** 2
                 + counts["losses"] * mean ** 2) / max(1, n - 1))
    margin = 1.959963984540054 * math.sqrt(variance / n)
    decisive = counts["wins"] + counts["losses"]
    return {**counts, "agent_player": "black" if agent_player == 1 else "white",
            "win_rate": counts["wins"] / n, "loss_rate": counts["losses"] / n,
            "draw_rate": counts["draws"] / n, "score_rate": mean,
            "win_rate_ci95": _wilson(counts["wins"], n),
            "score_rate_ci95": (max(0.0, mean - margin), min(1.0, mean + margin)),
            "decisive_game_rate": decisive / n,
            "decisive_win_rate": counts["wins"] / decisive if decisive else None,
            "average_game_length": counts["moves"] / n, "duration_seconds": seconds}


def passes_45_55(result: dict[str, Any], min_decisive_rate: float = 0.20) -> bool:
    return 0.45 <= result["score_rate"] <= 0.55 and \
        result["decisive_game_rate"] >= min_decisive_rate


def _expert_worker(task: dict[str, Any]) -> dict[str, int]:
    import torch
    torch.set_num_threads(1)
    agent = BCAgent(task["board_size"], device="cpu")
    agent.load_checkpoint(Path(task["checkpoint"])); agent.net.eval()
    expert = HeuristicAgent(seed=task["seed"], max_candidates=task["max_candidates"])
    expert_rng = np.random.default_rng(task["seed"] + 991)
    decision_cache: dict[tuple[bytes, int], Any] = {}
    wins = losses = draws = moves = illegal = 0
    for game in range(task["games"]):
        env = GomokuEnv(board_size=task["board_size"], starting_player="black",
                        illegal_action_mode="raise")
        obs, _ = env.reset(seed=task["seed"] + game); done = False
        while not done:
            player = int(obs["current_player"][0])
            if player == task["agent_player"]:
                action = int(agent.select_actions(obs["board"], [player], obs["action_mask"])[0])
            else:
                key = (obs["board"].tobytes(), player)
                decision = decision_cache.get(key)
                if decision is None:
                    decision = expert.ranked_decision(obs["board"], player, obs["action_mask"],
                                                      task["expert_top_k"])
                    decision_cache[key] = decision
                action = rank_softmax_action(
                    decision.actions, int(np.count_nonzero(obs["board"] == player)),
                    expert_rng, top_k=task["expert_top_k"],
                    temperature=task["expert_temperature"],
                    stochastic_moves=task["expert_stochastic_moves"],
                    force_greedy=decision.tactical)
            if not obs["action_mask"][action]:
                illegal += 1
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
             workers: int, max_candidates: int, agent_player: int,
             expert_top_k: int = 4, expert_temperature: float = 1.5,
             expert_stochastic_moves: int = 6) -> dict[str, Any]:
    """Evaluate one greedy BC policy against the BC-local heuristic expert."""
    if games < 1 or workers < 1 or agent_player not in (1, -1):
        raise ValueError("invalid evaluation games, workers or player")
    workers = min(workers, games)
    counts = [games // workers + (i < games % workers) for i in range(workers)]
    tasks = [{"checkpoint": str(checkpoint), "board_size": board_size, "games": count,
              "seed": seed + i * 1_000_003, "max_candidates": max_candidates,
              "expert_top_k": expert_top_k, "expert_temperature": expert_temperature,
              "expert_stochastic_moves": expert_stochastic_moves,
              "agent_player": agent_player} for i, count in enumerate(counts)]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_expert_worker, tasks))
    totals = {key: sum(part[key] for part in parts)
              for key in ("wins", "losses", "draws", "moves", "illegal_actions", "games")}
    return _summarize(totals, agent_player, time.perf_counter() - started)


def _pair_worker(task: dict[str, Any]) -> dict[str, int]:
    import torch
    torch.set_num_threads(1)
    a = BCAgent(task["board_size"], device="cpu"); a.load_checkpoint(Path(task["checkpoint_a"]))
    b = BCAgent(task["board_size"], device="cpu"); b.load_checkpoint(Path(task["checkpoint_b"]))
    a.net.eval(); b.net.eval()
    a_wins = b_wins = draws = moves = 0
    for game in range(task["games"]):
        env = GomokuEnv(board_size=task["board_size"], starting_player="black",
                        illegal_action_mode="raise")
        obs, _ = env.reset(seed=task["seed"] + game); done = False
        while not done:
            player = int(obs["current_player"][0])
            actor = a if player == task["a_player"] else b
            action = int(actor.select_actions(obs["board"], [player], obs["action_mask"])[0])
            obs, _, terminated, truncated, info = env.step(action)
            moves += 1; done = terminated or truncated
        winner = int(info["winner"])
        if winner == task["a_player"]: a_wins += 1
        elif winner == -task["a_player"]: b_wins += 1
        else: draws += 1
        env.close()
    return {"a_wins": a_wins, "b_wins": b_wins, "draws": draws,
            "moves": moves, "games": task["games"]}


def evaluate_checkpoint_pair(checkpoint_a: Path, checkpoint_b: Path, *, board_size: int,
                             games: int, seed: int, workers: int, a_player: int) -> dict[str, Any]:
    if games < 1 or workers < 1 or a_player not in (1, -1):
        raise ValueError("invalid pair evaluation games, workers or player")
    workers = min(workers, games)
    counts = [games // workers + (i < games % workers) for i in range(workers)]
    tasks = [{"checkpoint_a": str(checkpoint_a), "checkpoint_b": str(checkpoint_b),
              "board_size": board_size, "games": count, "seed": seed + i * 1_000_003,
              "a_player": a_player} for i, count in enumerate(counts)]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_pair_worker, tasks))
    totals = {key: sum(part[key] for part in parts)
              for key in ("a_wins", "b_wins", "draws", "moves", "games")}
    return {**totals, "a_player": "black" if a_player == 1 else "white",
            "a_win_rate": totals["a_wins"] / games, "b_win_rate": totals["b_wins"] / games,
            "draw_rate": totals["draws"] / games, "average_game_length": totals["moves"] / games,
            "duration_seconds": time.perf_counter() - started}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评测 BC checkpoint：对启发式、整个 run 或两个 BC 模型互评。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python BC/eval.py --checkpoint PATH --board-size 5
  python BC/eval.py --run-name RUN_NAME --checkpoint-kind best --board-size 5
  python BC/eval.py --checkpoint-a A.pt --checkpoint-b B.pt --board-size 5
""",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run-name")
    parser.add_argument("--checkpoint-a", type=Path)
    parser.add_argument("--checkpoint-b", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpoint-kind", choices=("best", "latest", "all"), default="best")
    parser.add_argument("--games", "--games-per-color", dest="games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--expert-top-k", type=int, default=4)
    parser.add_argument("--expert-temperature", type=float, default=1.5)
    parser.add_argument("--expert-stochastic-moves", type=int, default=6)
    parser.add_argument("--min-decisive-rate", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tb-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    pair = args.checkpoint_a is not None or args.checkpoint_b is not None
    if pair and (args.checkpoint_a is None or args.checkpoint_b is None):
        parser.error("--checkpoint-a and --checkpoint-b must be specified together")
    if int(args.checkpoint is not None) + int(args.run_name is not None) + int(pair) != 1:
        parser.error("specify exactly one source: --checkpoint, --run-name, or checkpoint A/B")
    if (args.games < 1 or args.workers < 1 or not 5 <= args.board_size <= 9
            or args.expert_top_k < 1 or args.expert_temperature < 0
            or not 0 <= args.min_decisive_rate <= 1):
        parser.error("games/workers must be positive and board-size must be 5..9")
    return args


def _write_tb(directory: Path, output: dict[str, Any]) -> None:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(directory))
    writer.add_text("Evaluation/result", json.dumps(output, ensure_ascii=False, indent=2), 0)
    for checkpoint_index, item in enumerate(output.get("evaluations", [])):
        for result in item["results"]:
            color = result["agent_player"].capitalize()
            for metric in ("win_rate", "loss_rate", "draw_rate", "score_rate",
                           "decisive_game_rate", "average_game_length", "illegal_actions"):
                writer.add_scalar(f"{color}/{metric}", result[metric], checkpoint_index)
    if output.get("mode") == "checkpoint_pair":
        for result in output["results"]:
            role = "A_black" if result["a_player"] == "black" else "A_white"
            for metric in ("a_win_rate", "b_win_rate", "draw_rate", "average_game_length"):
                writer.add_scalar(f"Pair/{role}/{metric}", result[metric], 0)
    writer.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.checkpoint_a is not None:
        a = resolve_checkpoint(args.checkpoint_root, direct=args.checkpoint_a)
        b = resolve_checkpoint(args.checkpoint_root, direct=args.checkpoint_b)
        results = [evaluate_checkpoint_pair(a, b, board_size=args.board_size, games=args.games,
                                            seed=args.seed, workers=args.workers, a_player=color)
                   for color in (1, -1)]
        output: dict[str, Any] = {"mode": "checkpoint_pair", "checkpoint_a": str(a),
                                  "checkpoint_b": str(b), "board_size": args.board_size,
                                  "games_per_color": args.games, "results": results}
    else:
        checkpoints = ([resolve_checkpoint(args.checkpoint_root, direct=args.checkpoint)]
                       if args.checkpoint is not None else
                       checkpoints_for_run(args.checkpoint_root, args.run_name, args.checkpoint_kind))
        evaluations = []
        for index, checkpoint in enumerate(checkpoints, 1):
            print(f"[{index}/{len(checkpoints)}] {checkpoint}", flush=True)
            results = [evaluate(checkpoint, args.board_size, args.games, args.seed,
                                args.workers, args.max_candidates, color,
                                args.expert_top_k, args.expert_temperature,
                                args.expert_stochastic_moves) for color in (1, -1)]
            evaluations.append({"checkpoint": str(checkpoint), "results": results,
                                "passes_45_55": all(passes_45_55(result, args.min_decisive_rate)
                                                    for result in results)})
        output = {"mode": "heuristic", "board_size": args.board_size,
                  "games_per_color": args.games, "seed": args.seed, "evaluations": evaluations,
                  "passes_45_55": all(item["passes_45_55"] for item in evaluations)}
        # Preserve the former single-checkpoint top-level result shape used by pipeline consumers.
        if len(evaluations) == 1:
            output.update({"checkpoint": evaluations[0]["checkpoint"],
                           "results": evaluations[0]["results"],
                           "passes_45_55": evaluations[0]["passes_45_55"]})
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(rendered + "\n")
    if args.tb_dir:
        _write_tb(args.tb_dir, output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n评测已中断。", file=sys.stderr); raise SystemExit(130)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr); raise SystemExit(2)
