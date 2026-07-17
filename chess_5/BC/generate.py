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
from BC.diversity import analyze_shards, assess_diversity, canonical_trajectory_hash
from BC.sampling import has_immediate_win_or_block, ranked_legal_actions, rank_softmax_action
from env import GomokuEnv


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    import torch
    torch.set_num_threads(1)
    worker = int(task["worker"]); board_size = int(task["board_size"])
    seed = int(task["seed"]) + worker * 1_000_003
    output = Path(task["output"]); shard = output / f"shard-{worker:05d}.npz"
    if shard.exists():
        with np.load(shard) as old:
            if "trajectory_groups" not in old:
                raise ValueError(f"old-format partial shard cannot be resumed: {shard}")
            return {"shard": shard.name, "samples": int(len(old["actions"])),
                    "games": int(task["games"]), "hits": 0, "misses": 0,
                    "expert_queries": 0, "seconds": 0.0, "resumed": True}

    policy = None
    if task["mode"] == "aggregate":
        policy = BCAgent(board_size, device="cpu")
        policy.load_checkpoint(Path(task["checkpoint"])); policy.net.eval()
    policy_rng = np.random.default_rng(seed + 424_242)
    boards: list[np.ndarray] = []; players: list[int] = []; actions: list[int] = []
    game_ids: list[int] = []; trajectory_groups: list[bytes] = []
    cache_path = Path(task["cache_dir"]) / f"cache-{worker:03d}.sqlite3"
    started = time.perf_counter()
    with ExpertCache(cache_path, board_size, int(task["max_candidates"]), seed,
                     top_k=int(task.get("expert_top_k", 4)),
                     temperature=float(task.get("expert_temperature", 1.5)),
                     stochastic_moves=int(task.get("expert_stochastic_moves", 6))) as expert:
        for local_game in range(int(task["games"])):
            game_id = (worker << 32) | local_game
            env = GomokuEnv(board_size=board_size, starting_player="black",
                            illegal_action_mode="raise")
            obs, _ = env.reset(seed=seed + local_game)
            learner_player = 1 if (worker + local_game) % 2 == 0 else -1
            start = len(boards); done = False
            while not done:
                board = obs["board"].copy(); player = int(obs["current_player"][0])
                expert_action = expert.label(board, player)
                boards.append(board); players.append(player); actions.append(expert_action)
                game_ids.append(game_id); actual_action = expert_action
                if policy is not None and player == learner_player:
                    logits = policy.action_logits(board, [player])[0]
                    ranked = ranked_legal_actions(logits, obs["action_mask"])
                    move_index = int(np.count_nonzero(board == player))
                    actual_action = rank_softmax_action(
                        ranked, move_index, policy_rng,
                        top_k=int(task.get("bc_top_k", 4)),
                        temperature=float(task.get("bc_temperature", 1.0)),
                        stochastic_moves=int(task.get("bc_stochastic_moves", 6)),
                        force_greedy=has_immediate_win_or_block(
                            board, player, obs["action_mask"]),
                    )
                obs, _, terminated, truncated, _ = env.step(actual_action)
                done = terminated or truncated
            env.close()
            group = canonical_trajectory_hash(np.asarray(boards[start:]),
                                              np.asarray(players[start:]))
            trajectory_groups.extend([group] * (len(boards) - start))
        hits, misses, expert_queries = expert.hits, expert.misses, expert.expert_queries

    labeling_seconds = time.perf_counter() - started
    arrays = {"boards": np.asarray(boards, dtype=np.int8),
              "players": np.asarray(players, dtype=np.int8),
              "actions": np.asarray(actions, dtype=np.int16),
              "games": np.asarray(game_ids, dtype=np.int64),
              "trajectory_groups": np.asarray(trajectory_groups, dtype="S32")}
    temporary = shard.with_suffix(".tmp.npz"); write_started = time.perf_counter()
    np.savez_compressed(temporary, **arrays); os.replace(temporary, shard)
    write_seconds = time.perf_counter() - write_started
    return {"shard": shard.name, "samples": len(actions), "games": int(task["games"]),
            "hits": hits, "misses": misses, "expert_queries": expert_queries,
            "seconds": time.perf_counter() - started, "labeling_seconds": labeling_seconds,
            "write_seconds": write_seconds,
            "queries_per_second": expert_queries / max(1e-9, labeling_seconds),
            "samples_per_second": len(actions) / max(1e-9, labeling_seconds),
            "resumed": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline Gomoku expert labels.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("expert", "aggregate"), default="expert")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--expert-top-k", "--cache-labels-per-state", dest="expert_top_k",
                        type=int, default=4,
                        help="Ranked expert candidates; old cache option is a deprecated alias.")
    parser.add_argument("--expert-temperature", type=float, default=1.5)
    parser.add_argument("--expert-stochastic-moves", type=int, default=6)
    parser.add_argument("--bc-top-k", type=int, default=4)
    parser.add_argument("--bc-temperature", type=float, default=1.0)
    parser.add_argument("--bc-stochastic-moves", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--tb-dir", type=Path, default=None)
    parser.add_argument("--quality-gate", action="store_true")
    parser.add_argument("--quality-min-games", type=int, default=100)
    parser.add_argument("--min-effective-trajectory-ratio", type=float, default=0.01)
    parser.add_argument("--max-dominant-trajectory-fraction", type=float, default=0.50)
    parser.add_argument("--min-state-unique-ratio", type=float, default=0.001)
    return parser.parse_args()


def _identity(args: argparse.Namespace, workers: int) -> dict[str, Any]:
    return {"format_version": 2, "mode": args.mode, "board_size": args.board_size,
            "seed": args.seed, "games": args.games, "workers": workers,
            "max_candidates": args.max_candidates, "expert_top_k": args.expert_top_k,
            "expert_temperature": args.expert_temperature,
            "expert_stochastic_moves": args.expert_stochastic_moves,
            "bc_top_k": args.bc_top_k, "bc_temperature": args.bc_temperature,
            "bc_stochastic_moves": args.bc_stochastic_moves,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None}


def _write_tensorboard(args: argparse.Namespace, metadata: dict[str, Any], diversity: dict[str, Any],
                       quality: dict[str, Any], results: list[dict[str, Any]]) -> None:
    if args.tb_dir is None:
        return
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(args.tb_dir))
    writer.add_text("Pipeline/config", json.dumps(_identity(args, metadata["workers"]),
                                                   ensure_ascii=False, indent=2), 0)
    writer.add_scalar("Data/games", args.games, 0); writer.add_scalar("Data/samples", metadata["samples"], 0)
    writer.add_scalar("Data/cache_hit_rate", metadata["cache_hit_rate"], 0)
    writer.add_scalar("Data/expert_queries", metadata["expert_queries"], 0)
    for tag, key in (("canonical_effective_trajectory_ratio", "canonical_effective_trajectory_ratio"),
                     ("dominant_canonical_trajectory_fraction", "dominant_canonical_trajectory_fraction"),
                     ("canonical_state_unique_ratio", "canonical_state_unique_ratio")):
        writer.add_scalar(f"Diversity/{tag}", diversity[key], 0)
    writer.add_scalar("Diversity/quality_gate_passed", float(quality["passed"]), 0)
    writer.add_text("Diversity/details", json.dumps({**diversity, "quality": quality},
                                                     ensure_ascii=False, indent=2), 0)
    for index, result in enumerate(results):
        writer.add_scalar("Workers/samples_per_second", result.get("samples_per_second", 0), index)
        writer.add_scalar("Workers/expert_queries_per_second", result.get("queries_per_second", 0), index)
        writer.add_scalar("Workers/write_seconds", result.get("write_seconds", 0), index)
    writer.close()


def main() -> None:
    args = parse_args()
    if not 5 <= args.board_size <= 9 or args.games < 1 or args.workers < 1:
        raise ValueError("board size must be 5..9 and games/workers positive")
    if min(args.expert_top_k, args.bc_top_k) < 1 or min(args.expert_temperature, args.bc_temperature) < 0:
        raise ValueError("top-k must be positive and temperatures non-negative")
    if args.mode == "aggregate" and args.checkpoint is None:
        raise ValueError("--checkpoint is required in aggregate mode")
    args.output.mkdir(parents=True, exist_ok=True); metadata_path = args.output / "metadata.json"
    workers = min(args.workers, args.games); identity = _identity(args, workers)
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("existing dataset metadata is incompatible")
        if previous.get("status") == "complete":
            raise FileExistsError("dataset version is already complete and immutable")
    cache_dir = args.cache_dir or args.output / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {**identity, "status": "running", "shards": [],
                                "created_at": time.time()}
    atomic_json(metadata_path, metadata)
    counts = [args.games // workers + (i < args.games % workers) for i in range(workers)]
    tasks = [dict(vars(args), worker=i, games=counts[i], output=str(args.output),
                  cache_dir=str(cache_dir)) for i in range(workers)]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(f"{result['shard']}: games={result['games']} samples={result['samples']} "
                  f"cache={result['hits']}/{result['hits'] + result['misses']} "
                  f"seconds={result['seconds']:.1f} q/s={result.get('queries_per_second', 0):.1f}", flush=True)
    results.sort(key=lambda item: item["shard"])
    shards = [args.output / result["shard"] for result in results]
    diversity = analyze_shards(shards)
    quality = assess_diversity(diversity, hard_min_games=args.quality_min_games,
                               min_effective_ratio=args.min_effective_trajectory_ratio,
                               max_dominant_fraction=args.max_dominant_trajectory_fraction,
                               min_state_unique_ratio=args.min_state_unique_ratio)
    diversity["quality"] = quality; atomic_json(args.output / "diversity.json", diversity)
    for warning in quality["warnings"]:
        print(f"!!!!!!!!!!!!!!!! 数据多样性警告：{warning} !!!!!!!!!!!!!!!!", flush=True)
    print("!!!!!!!!!!!!!!!! 数据多样性量化结果 !!!!!!!!!!!!!!!!", flush=True)
    print(f"!!!!!!!!!!!!!!!! 有效轨迹比例："
          f"{diversity['canonical_effective_trajectory_ratio']:.2%}（越高越好） !!!!!!!!!!!!!!!!",
          flush=True)
    print(f"!!!!!!!!!!!!!!!! 最大单一轨迹占比："
          f"{diversity['dominant_canonical_trajectory_fraction']:.2%}（越低越好） !!!!!!!!!!!!!!!!",
          flush=True)
    print(f"!!!!!!!!!!!!!!!! 独特状态比例："
          f"{diversity['canonical_state_unique_ratio']:.2%}（越高越好） !!!!!!!!!!!!!!!!",
          flush=True)
    if not quality["passed"]:
        conclusion = "不可接受，将拒绝进入训练" if args.quality_gate else "不可接受"
    elif quality["warnings"]:
        conclusion = "可接受，但多样性仍有警告，需要关注"
    else:
        conclusion = "可接受，多样性良好"
    print(f"!!!!!!!!!!!!!!!! 数据质量结论：{conclusion} !!!!!!!!!!!!!!!!", flush=True)
    if quality["failures"]:
        print("!!!!!!!!!!!!!!!! 不通过原因：" + " | ".join(quality["failures"])
              + " !!!!!!!!!!!!!!!!", flush=True)
    hits = sum(r["hits"] for r in results); misses = sum(r["misses"] for r in results)
    metadata.update({"status": "complete" if quality["passed"] or not args.quality_gate else "rejected",
                     "shards": [r["shard"] for r in results],
                     "samples": sum(r["samples"] for r in results), "cache_hits": hits,
                     "cache_misses": misses, "cache_hit_rate": hits / max(1, hits + misses),
                     "expert_queries": sum(r["expert_queries"] for r in results),
                     "diversity_file": "diversity.json", "diversity": diversity,
                     "quality": quality, "worker_results": results, "completed_at": time.time()})
    atomic_json(metadata_path, metadata); _write_tensorboard(args, metadata, diversity, quality, results)
    if args.quality_gate and not quality["passed"]:
        raise RuntimeError("dataset rejected by diversity quality gate: " + "; ".join(quality["failures"]))
    print(f"complete: samples={metadata['samples']} cache_hit_rate={metadata['cache_hit_rate']:.3f} "
          f"expert_queries={metadata['expert_queries']}")


if __name__ == "__main__":
    main()
