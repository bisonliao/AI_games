"""Build a frozen challenge bank and evaluate BC checkpoints against it."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BC.agent import BCAgent
from BC.heuristic_agent import HeuristicAgent
from BC.oracle import REASON_TO_ID, SOURCE_TO_ID, oracle_identity, validate_oracle_identity
from gomoku_env import GomokuEnv


V3_FIELDS = (
    "boards", "players", "actions", "candidate_actions", "candidate_counts",
    "tactical", "reasons", "plies", "sources", "rounds", "behavior_actions",
    "canonical_keys", "games", "trajectory_groups",
)


def build_bank(data_dir: Path, output: Path, *, seed: int = 0,
               ood_states: int = 20_000, win_states: int = 2_000,
               block_states: int = 2_000, fork_states: int = 1_000,
               prefix_count: int = 500) -> dict[str, Any]:
    metadata = json.loads((data_dir / "metadata.json").read_text())
    if int(metadata.get("format_version", 0)) < 3:
        raise ValueError("challenge banks require v3 datasets")
    collected = {name: [] for name in V3_FIELDS}
    prefixes: list[np.ndarray] = []
    selected_keys: set[bytes] = set()
    tactical_counts = {"win": 0, "block": 0, "fork": 0}
    ood_count = 0
    for shard_name in metadata["shards"]:
        with np.load(data_dir / shard_name) as data:
            groups = np.asarray(data["trajectory_groups"])
            # Challenge canonical keys are explicitly excluded by the trainer,
            # so the bank may draw from every generated trajectory.
            val = np.ones(len(groups), dtype=bool)
            game_ids = np.asarray(data["games"])
            starts = np.r_[0, np.flatnonzero(game_ids[1:] != game_ids[:-1]) + 1]
            ends = np.r_[starts[1:], len(game_ids)]
            for start, end in zip(starts, ends):
                if (len(prefixes) < prefix_count and val[start] and end - start >= 4
                        and int(data["sources"][start]) != SOURCE_TO_ID["expert_selfplay"]):
                    length = min(16, max(4, 4 + (len(prefixes) * 13) % 13), end - start)
                    prefixes.append(np.asarray(data["behavior_actions"][start:start + length],
                                               dtype=np.int16))
            for index in np.flatnonzero(val):
                key = bytes(data["canonical_keys"][index])
                reason = int(data["reasons"][index])
                source = int(data["sources"][index])
                category = ("win" if reason == REASON_TO_ID["win"] else
                            "block" if reason == REASON_TO_ID["block"] else
                            "fork" if reason in (REASON_TO_ID["own_fork"],
                                                 REASON_TO_ID["block_fork"]) else None)
                want_tactical = bool(
                    category == "win" and tactical_counts["win"] < win_states
                    or category == "block" and tactical_counts["block"] < block_states
                    or category == "fork" and tactical_counts["fork"] < fork_states
                )
                want_ood = (key not in selected_keys
                            and source != SOURCE_TO_ID["expert_selfplay"]
                            and ood_count < ood_states)
                if not want_tactical and not want_ood:
                    continue
                for name in V3_FIELDS:
                    collected[name].append(np.asarray(data[name][index]).copy())
                selected_keys.add(key)
                if want_ood:
                    ood_count += 1
                if want_tactical and category is not None:
                    tactical_counts[category] += 1
    if (ood_count < ood_states or len(prefixes) < prefix_count
            or tactical_counts["win"] < win_states
            or tactical_counts["block"] < block_states
            or tactical_counts["fork"] < fork_states):
        raise RuntimeError(
            f"bootstrap data cannot fill challenge bank: ood={ood_count}/{ood_states}, "
            f"prefixes={len(prefixes)}/{prefix_count}, tactical={tactical_counts}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix_array = np.full((prefix_count, 16), -1, dtype=np.int16)
    prefix_lengths = np.zeros(prefix_count, dtype=np.int8)
    for index, prefix in enumerate(prefixes[:prefix_count]):
        prefix_array[index, :len(prefix)] = prefix
        prefix_lengths[index] = len(prefix)
    arrays = {name: np.asarray(values) for name, values in collected.items()}
    arrays.update({"prefix_actions": prefix_array, "prefix_lengths": prefix_lengths})
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays); os.replace(temporary, output)
    result = {
        "format_version": 1, "source_data": str(data_dir.resolve()),
        "samples": len(arrays["actions"]), "ood_states": ood_count,
        "tactical_counts": tactical_counts, "prefixes": prefix_count,
        "oracle": metadata["oracle"],
    }
    output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _agreement(agent: BCAgent, bank: Path, *, limit: int = 0) -> dict[str, Any]:
    paths = [bank]
    if bank.is_dir():
        metadata = json.loads((bank / "metadata.json").read_text())
        paths = [bank / name for name in metadata["shards"]]
    fields: dict[str, list[np.ndarray]] = {name: [] for name in
                                           ("boards", "players", "actions",
                                            "candidate_actions", "reasons", "sources")}
    remaining = limit if limit > 0 else None
    for path in paths:
        with np.load(path) as data:
            count = len(data["actions"]) if remaining is None else min(remaining, len(data["actions"]))
            for name in fields:
                fields[name].append(np.asarray(data[name][:count]))
        if remaining is not None:
            remaining -= count
            if remaining <= 0:
                break
    boards = np.concatenate(fields["boards"]); players = np.concatenate(fields["players"])
    targets = np.concatenate(fields["actions"])
    candidates = np.concatenate(fields["candidate_actions"])
    reasons = np.concatenate(fields["reasons"])
    sources = np.concatenate(fields["sources"])
    count = len(targets); masks = boards.reshape(count, -1) == 0
    predictions = []
    for start in range(0, count, 2048):
        predictions.append(agent.select_actions(boards[start:start + 2048],
                                                players[start:start + 2048],
                                                masks[start:start + 2048]))
    predictions = np.concatenate(predictions)
    top1 = predictions == targets
    top4 = ((predictions[:, None] == candidates) & (candidates >= 0)).any(1)
    win_block = np.isin(reasons, [REASON_TO_ID["win"], REASON_TO_ID["block"]])
    forks = np.isin(reasons, [REASON_TO_ID["own_fork"], REASON_TO_ID["block_fork"]])
    ood = sources != SOURCE_TO_ID["expert_selfplay"]
    return {
        "samples": count, "illegal_actions": int(np.count_nonzero(~masks[np.arange(count), predictions])),
        "top1_accuracy": float(top1.mean()) if count else 0.0,
        "top4_accuracy": float(top4.mean()) if count else 0.0,
        "ood_top1_accuracy": float(top1[ood].mean()) if ood.any() else 0.0,
        "ood_top4_accuracy": float(top4[ood].mean()) if ood.any() else 0.0,
        "ood_samples": int(ood.sum()),
        "win_block_accuracy": float(top1[win_block].mean()) if win_block.any() else 0.0,
        "fork_top4_accuracy": float(top4[forks].mean()) if forks.any() else 0.0,
        "win_block_samples": int(win_block.sum()), "fork_samples": int(forks.sum()),
    }


def _match_worker(task: dict[str, Any]) -> dict[str, int]:
    import torch
    torch.set_num_threads(1)
    agent = BCAgent(task["board_size"], device="cpu")
    agent.load_checkpoint(Path(task["checkpoint"])); agent.net.eval()
    expert = HeuristicAgent(seed=task["seed"], max_candidates=task["max_candidates"])
    totals = {color: {name: 0 for name in ("wins", "losses", "draws", "games")}
              for color in ("black", "white")}
    moves = illegal = 0
    for prefix, length in zip(task["prefixes"], task["lengths"]):
        for agent_player in (1, -1):
            env = GomokuEnv(board_size=task["board_size"], starting_player="black",
                            illegal_action_mode="raise")
            obs, _ = env.reset(seed=task["seed"]); done = False
            for action in prefix[:int(length)]:
                obs, _, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    done = True; break
            while not done:
                player = int(obs["current_player"][0])
                if player == agent_player:
                    action = int(agent.select_actions(obs["board"], [player],
                                                      obs["action_mask"])[0])
                else:
                    action = expert.ranked_decision(
                        obs["board"], player, obs["action_mask"], top_k=1
                    ).actions[0]
                if not obs["action_mask"][action]:
                    illegal += 1
                obs, _, terminated, truncated, info = env.step(action)
                moves += 1; done = terminated or truncated
            winner = int(info["winner"])
            color = "black" if agent_player == 1 else "white"
            totals[color]["wins"] += int(winner == agent_player)
            totals[color]["losses"] += int(winner == -agent_player)
            totals[color]["draws"] += int(winner == 0)
            totals[color]["games"] += 1; env.close()
    return {"colors": totals, "moves": moves, "illegal_actions": illegal}


def _matches(checkpoint: Path, bank: Path, board_size: int, workers: int,
             max_candidates: int, seed: int) -> dict[str, Any]:
    with np.load(bank) as data:
        prefixes = data["prefix_actions"]; lengths = data["prefix_lengths"]
    workers = min(workers, len(prefixes))
    chunks = np.array_split(np.arange(len(prefixes)), workers)
    tasks = [{"checkpoint": str(checkpoint), "board_size": board_size,
              "prefixes": prefixes[chunk], "lengths": lengths[chunk],
              "max_candidates": max_candidates, "seed": seed + i * 1_000_003}
             for i, chunk in enumerate(chunks)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_match_worker, tasks))
    colors = {}
    for color in ("black", "white"):
        totals = {key: sum(part["colors"][color][key] for part in parts)
                  for key in ("wins", "losses", "draws", "games")}
        score = (totals["wins"] + 0.5 * totals["draws"]) / max(1, totals["games"])
        variance = ((totals["wins"] * (1 - score) ** 2
                     + totals["draws"] * (0.5 - score) ** 2
                     + totals["losses"] * score ** 2) / max(1, totals["games"] - 1))
        margin = 1.959963984540054 * math.sqrt(variance / max(1, totals["games"]))
        colors[color] = {**totals, "score_rate": score,
                         "score_rate_ci95": [max(0.0, score - margin),
                                             min(1.0, score + margin)],
                         "decisive_game_rate":
                             (totals["wins"] + totals["losses"]) / max(1, totals["games"])}
    return {"colors": colors, "moves": sum(part["moves"] for part in parts),
            "illegal_actions": sum(part["illegal_actions"] for part in parts)}


def evaluate_composite(checkpoint: Path, bank: Path, *, board_size: int = 9,
                       audit_data: Path | None = None, workers: int = 16,
                       max_candidates: int = 12, seed: int = 20_000,
                       skip_matches: bool = False) -> dict[str, Any]:
    metadata = json.loads(bank.with_suffix(".json").read_text())
    validate_oracle_identity(metadata.get("oracle"),
                             oracle_identity(max_candidates=max_candidates, top_k=4))
    agent = BCAgent(board_size, device="cpu"); checkpoint_data = agent.load_checkpoint(checkpoint)
    if checkpoint_data.get("oracle") is not None:
        validate_oracle_identity(checkpoint_data["oracle"], metadata["oracle"])
    agreement = _agreement(agent, bank)
    audit = _agreement(agent, audit_data, limit=20_000) if audit_data is not None else None
    matches = (None if skip_matches else
               _matches(checkpoint, bank, board_size, workers, max_candidates, seed))
    passed = (
        agreement["illegal_actions"] == 0
        and agreement["ood_top1_accuracy"] >= 0.85
        and agreement["ood_top4_accuracy"] >= 0.97
        and agreement["win_block_accuracy"] >= 0.995
        and agreement["fork_top4_accuracy"] >= 0.98
        and (audit is None or audit["top4_accuracy"] >= 0.95)
        and (matches is None or (matches["illegal_actions"] == 0 and all(
            result["score_rate"] >= 0.45
            and result["score_rate_ci95"][0] >= 0.40
            and result["decisive_game_rate"] >= 0.20
            for result in matches["colors"].values()
        )))
    )
    return {"checkpoint": str(checkpoint), "challenge_bank": str(bank),
            "agreement": agreement, "rollout_audit": audit,
            "matches": matches, "passed": bool(passed)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/evaluate the fixed BC challenge bank.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--data-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--ood-states", type=int, default=20_000)
    build.add_argument("--win-states", type=int, default=2_000)
    build.add_argument("--block-states", type=int, default=2_000)
    build.add_argument("--fork-states", type=int, default=1_000)
    build.add_argument("--prefix-count", type=int, default=500)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--bank", type=Path, required=True)
    evaluate.add_argument("--audit-data", type=Path, default=None)
    evaluate.add_argument("--board-size", type=int, default=9)
    evaluate.add_argument("--workers", type=int, default=16)
    evaluate.add_argument("--max-candidates", type=int, default=12)
    evaluate.add_argument("--seed", type=int, default=20_000)
    evaluate.add_argument("--skip-matches", action="store_true")
    evaluate.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result = build_bank(args.data_dir, args.output, seed=args.seed,
                            ood_states=args.ood_states, win_states=args.win_states,
                            block_states=args.block_states, fork_states=args.fork_states,
                            prefix_count=args.prefix_count)
    else:
        result = evaluate_composite(
            args.checkpoint, args.bank, board_size=args.board_size,
            audit_data=args.audit_data, workers=args.workers,
            max_candidates=args.max_candidates, seed=args.seed,
            skip_matches=args.skip_matches,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
