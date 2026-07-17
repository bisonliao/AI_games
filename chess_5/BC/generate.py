"""Generate immutable expert-self-play or offline aggregation datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BC.agent import BCAgent
from BC.cache import ExpertCache
from BC.diversity import analyze_shards
from env import GomokuEnv


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    import torch
    torch.set_num_threads(1)
    worker = int(task["worker"])
    board_size = int(task["board_size"])
    seed = int(task["seed"]) + worker * 1_000_003
    output = Path(task["output"])
    shard = output / f"shard-{worker:05d}.npz"
    if shard.exists():
        with np.load(shard) as old:
            return {"shard": shard.name, "samples": int(len(old["actions"])),
                    "games": int(task["games"]), "hits": 0, "misses": 0, "expert_queries": 0,
                    "seconds": 0.0, "resumed": True}

    cache_path = Path(task["cache_dir"]) / f"cache-{worker:03d}.sqlite3"
    policy = None
    if task["mode"] == "aggregate":
        policy = BCAgent(board_size, device="cpu")
        policy.load_checkpoint(Path(task["checkpoint"]))
        policy.net.eval()

    boards: list[np.ndarray] = []
    players: list[int] = []
    actions: list[int] = []
    game_ids: list[int] = []
    started = time.perf_counter()
    with ExpertCache(cache_path, board_size, int(task["max_candidates"]), seed,
                     int(task["cache_labels_per_state"])) as expert:
        for local_game in range(int(task["games"])):
            game_id = (worker << 32) | local_game
            env = GomokuEnv(board_size=board_size, starting_player="black", illegal_action_mode="raise")
            obs, _ = env.reset(seed=seed + local_game)
            learner_player = 1 if (worker + local_game) % 2 == 0 else -1
            done = False
            while not done:
                board = obs["board"].copy()
                player = int(obs["current_player"][0])
                expert_action = expert.label(board, player)
                boards.append(board); players.append(player)
                actions.append(expert_action); game_ids.append(game_id)
                actual_action = expert_action
                if policy is not None and player == learner_player:
                    actual_action = int(policy.select_actions(
                        board, [player], obs["action_mask"]
                    )[0])
                obs, _, terminated, truncated, _ = env.step(actual_action)
                done = terminated or truncated
            env.close()
        hits, misses, expert_queries = expert.hits, expert.misses, expert.expert_queries

    labeling_seconds = time.perf_counter() - started
    arrays = {
        "boards": np.asarray(boards, dtype=np.int8),
        "players": np.asarray(players, dtype=np.int8),
        "actions": np.asarray(actions, dtype=np.int16),
        "games": np.asarray(game_ids, dtype=np.int64),
    }
    temporary = shard.with_suffix(".tmp.npz")
    write_started = time.perf_counter()
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, shard)
    write_seconds = time.perf_counter() - write_started
    return {"shard": shard.name, "samples": len(actions), "games": int(task["games"]),
            "hits": hits, "misses": misses, "expert_queries": expert_queries,
            "seconds": time.perf_counter() - started,
            "labeling_seconds": labeling_seconds, "write_seconds": write_seconds,
            "queries_per_second": expert_queries / max(1e-9, labeling_seconds),
            "samples_per_second": len(actions) / max(1e-9, labeling_seconds),
            "resumed": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline Gomoku expert labels.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("expert", "aggregate"), default="expert")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Frozen BC checkpoint; required in aggregate mode.")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--cache-labels-per-state", type=int, default=4,
                        help="Expert tie samples cached per canonical state (default: 4).")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--tb-dir", type=Path, default=None,
                        help="Exact TensorBoard directory for this pipeline step.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.board_size < 5 or args.board_size > 9:
        raise ValueError("--board-size must be between 5 and 9")
    if args.games < 1 or args.workers < 1 or args.cache_labels_per_state < 1:
        raise ValueError("games, workers and cache labels must be positive")
    if args.mode == "aggregate" and args.checkpoint is None:
        raise ValueError("--checkpoint is required in aggregate mode")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        previous_checkpoint = previous.get("checkpoint")
        requested_checkpoint = str(args.checkpoint.resolve()) if args.checkpoint else None
        identity = (previous.get("board_size"), previous.get("mode"), previous.get("seed"),
                    previous.get("games"), previous.get("max_candidates"),
                    previous.get("cache_labels_per_state"), previous.get("workers"),
                    previous_checkpoint)
        if identity != (args.board_size, args.mode, args.seed, args.games, args.max_candidates,
                        args.cache_labels_per_state,
                        min(args.workers, args.games), requested_checkpoint):
            raise ValueError("existing dataset metadata is incompatible")
        if previous.get("status") == "complete":
            raise FileExistsError("dataset version is already complete and immutable")

    workers = min(args.workers, args.games)
    counts = [args.games // workers + (i < args.games % workers) for i in range(workers)]
    cache_dir = args.cache_dir or args.output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "format_version": 1, "status": "running", "mode": args.mode,
        "board_size": args.board_size, "seed": args.seed, "games": args.games,
        "workers": workers, "max_candidates": args.max_candidates,
        "cache_labels_per_state": args.cache_labels_per_state,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "shards": [], "created_at": time.time(),
    }
    atomic_json(metadata_path, metadata)
    tasks = [dict(vars(args), worker=i, games=counts[i], output=str(args.output),
                  cache_dir=str(cache_dir)) for i in range(workers)]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(f"{result['shard']}: games={result['games']} samples={result['samples']} "
                  f"cache={result['hits']}/{result['hits'] + result['misses']} "
                  f"seconds={result['seconds']:.1f} q/s={result.get('queries_per_second', 0):.1f} "
                  f"write={result.get('write_seconds', 0):.2f}s", flush=True)
    results.sort(key=lambda item: item["shard"])
    shard_paths = [args.output / result["shard"] for result in results]
    diversity = analyze_shards(shard_paths)
    atomic_json(args.output / "diversity.json", diversity)
    print(
        "diversity: effective_trajectory_ratio="
        f"{diversity['canonical_effective_trajectory_ratio']:.3f} "
        "dominant_trajectory_fraction="
        f"{diversity['dominant_canonical_trajectory_fraction']:.3f} "
        f"state_unique_ratio={diversity['canonical_state_unique_ratio']:.3f} "
        f"analysis_seconds={diversity['analysis_seconds']:.2f}",
        flush=True,
    )
    metadata.update({
        "status": "complete", "shards": [r["shard"] for r in results],
        "samples": sum(r["samples"] for r in results),
        "cache_hits": sum(r["hits"] for r in results),
        "cache_misses": sum(r["misses"] for r in results), "completed_at": time.time(),
        "expert_queries": sum(r.get("expert_queries", 0) for r in results),
        "diversity_file": "diversity.json", "diversity": diversity,
        "worker_results": results,
    })
    atomic_json(metadata_path, metadata)
    queries = metadata["cache_hits"] + metadata["cache_misses"]
    hit_rate = metadata["cache_hits"] / max(1, queries)
    print(f"complete: samples={metadata['samples']} cache_hit_rate={hit_rate:.3f}")
    if args.tb_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(args.tb_dir))
        writer.add_text("Pipeline/config", json.dumps({
            "mode": args.mode, "board_size": args.board_size, "games": args.games,
            "workers": workers, "seed": args.seed, "output": str(args.output),
        }, ensure_ascii=False, indent=2), 0)
        writer.add_scalar("Data/games", args.games, 0)
        writer.add_scalar("Data/samples", metadata["samples"], 0)
        writer.add_scalar("Data/cache_hit_rate", hit_rate, 0)
        writer.add_scalar("Data/expert_queries", metadata["expert_queries"], 0)
        writer.add_scalar("Diversity/canonical_effective_trajectory_ratio",
                          diversity["canonical_effective_trajectory_ratio"], 0)
        writer.add_scalar("Diversity/dominant_canonical_trajectory_fraction",
                          diversity["dominant_canonical_trajectory_fraction"], 0)
        writer.add_scalar("Diversity/canonical_state_unique_ratio",
                          diversity["canonical_state_unique_ratio"], 0)
        writer.add_text("Diversity/details", json.dumps(diversity, ensure_ascii=False, indent=2), 0)
        for index, result in enumerate(results):
            writer.add_scalar("Workers/samples_per_second", result.get("samples_per_second", 0), index)
            writer.add_scalar("Workers/expert_queries_per_second", result.get("queries_per_second", 0), index)
            writer.add_scalar("Workers/write_seconds", result.get("write_seconds", 0), index)
        writer.close()


if __name__ == "__main__":
    main()
