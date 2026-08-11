"""生成不可变的 Bootstrap 或 DAgger 监督数据集。

每个状态同时保存两种动作：``actions`` / ``candidate_actions`` 是冻结专家给出的
监督标签；``behavior_actions`` 是真正落到环境中、负责推进棋局的动作。DAgger 用
当前 Agent 的 behavior action 制造它自己容易访问的后续状态，但训练标签始终来自
专家，Agent 的错误动作不会被当作正确答案学习。

生成任务按 worker 拆成独立 NPZ shard 和 SQLite 专家缓存。全部 shard 完成后，
主进程汇总新增状态、战术覆盖和数据多样性，再原子提交 metadata。
"""

from __future__ import annotations

import argparse
import json
import math
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
from BC.cache import ExpertCache, merge_caches
from BC.diversity import analyze_shards, assess_diversity, canonical_trajectory_hash
from BC.oracle import (DATA_FORMAT_VERSION, DEFAULT_ORACLE_TOP_K, REASON_TO_ID,
                       SOURCE_TO_ID, encode_decision, oracle_identity)
from BC.sampling import has_immediate_win_or_block, ranked_legal_actions, rank_softmax_action
from BC.symmetry import canonicalize
from gomoku_env import GomokuEnv


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """先写临时文件再替换目标，保证恢复时只会读到完整 JSON。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _local_random_action(board: np.ndarray, action_mask: np.ndarray,
                         rng: np.random.Generator) -> int:
    """在棋子附近选择局部随机合法动作，避免制造完全不自然的棋局。

    空棋盘优先中央区域；已有棋子时优先选择距任意棋子不超过两格的位置。
    """
    size = board.shape[0]
    legal = np.flatnonzero(np.asarray(action_mask, dtype=bool).reshape(-1))
    occupied = np.argwhere(board != 0)
    if occupied.size == 0:
        margin = max(0, (size - 5) // 2)
        central = [int(action) for action in legal
                   if margin <= int(action) // size < size - margin
                   and margin <= int(action) % size < size - margin]
        legal = np.asarray(central or legal.tolist(), dtype=np.int64)
    else:
        nearby = []
        for action in legal:
            row, col = divmod(int(action), size)
            if np.min(np.max(np.abs(occupied - np.asarray([row, col])), axis=1)) <= 2:
                nearby.append(int(action))
        if nearby:
            legal = np.asarray(nearby, dtype=np.int64)
    return int(rng.choice(legal))


def _policy_action(policy: BCAgent, board: np.ndarray, player: int,
                   action_mask: np.ndarray, rng: np.random.Generator,
                   task: dict[str, Any]) -> int:
    """按 BC logits 排序做开局受控采样，必胜或必堵局面强制 greedy。"""
    logits = policy.action_logits(board, [player])[0]
    ranked = ranked_legal_actions(logits, action_mask)
    move_index = int(np.count_nonzero(board == player))
    return rank_softmax_action(
        ranked, move_index, rng, top_k=int(task.get("bc_top_k", 4)),
        temperature=float(task.get("bc_temperature", 1.0)),
        stochastic_moves=int(task.get("bc_stochastic_moves", 8)),
        force_greedy=has_immediate_win_or_block(board, player, action_mask),
    )


def _behavior_source(mode: str, worker: int, local_game: int) -> str:
    """用确定性 slot 分配每盘棋的行为来源。

    Bootstrap：50% 专家自博弈、25% 扰动开局、25% epsilon 专家。
    DAgger：40% 当前策略对专家、30% 自博弈、20% 对历史策略、10% epsilon 策略。
    分配只依赖 worker 和局号，所以中断恢复不会改变已有棋局的来源。
    """
    slot = (worker * 997 + local_game) % (4 if mode in ("expert", "bootstrap") else 10)
    if mode in ("expert", "bootstrap"):
        return ("expert_selfplay" if slot < 2 else
                "perturbed_opening" if slot == 2 else "epsilon_expert")
    return ("policy_expert" if slot < 4 else
            "policy_selfplay" if slot < 7 else
            "policy_history" if slot < 9 else "epsilon_policy")


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    """生成一个独立 shard；函数内的逐局循环是 DAgger 数据收集核心。"""
    import torch
    torch.set_num_threads(1)
    worker = int(task["worker"]); board_size = int(task["board_size"])
    seed = int(task["seed"]) + worker * 1_000_003
    output = Path(task["output"]); shard = output / f"shard-{worker:05d}.npz"
    # shard 是 worker 的提交边界。已存在就复用，避免恢复时重跑已完成棋局。
    if shard.exists():
        with np.load(shard) as old:
            if "trajectory_groups" not in old or (
                    int(task.get("data_format_version", DATA_FORMAT_VERSION)) >= 3
                    and "candidate_actions" not in old):
                raise ValueError(f"old-format partial shard cannot be resumed: {shard}")
            return {"shard": shard.name, "samples": int(len(old["actions"])),
                    "games": int(task["games"]), "hits": 0, "misses": 0,
                    "expert_queries": 0, "seconds": 0.0, "resumed": True}

    mode = str(task["mode"])
    is_dagger = mode in ("aggregate", "dagger")
    policy = None
    # Bootstrap 没有可用策略；只有 DAgger/aggregate 才加载行为策略。
    if is_dagger:
        policy = BCAgent(board_size, device="cpu")
        policy.load_checkpoint(Path(task["checkpoint"])); policy.net.eval()
    history_paths = [Path(path) for path in task.get("history_checkpoints", [])]
    history_agents: dict[Path, BCAgent] = {}
    policy_rng = np.random.default_rng(seed + 424_242)
    # 以下列表按下标严格对齐：每个下标描述一次落子前状态的全部监督信息。
    boards: list[np.ndarray] = []; players: list[int] = []; actions: list[int] = []
    game_ids: list[int] = []; trajectory_groups: list[bytes] = []
    candidate_actions: list[np.ndarray] = []; candidate_counts: list[int] = []
    tacticals: list[bool] = []; reasons: list[int] = []; plies: list[int] = []
    sources: list[int] = []; rounds: list[int] = []; behavior_actions: list[int] = []
    canonical_keys: list[bytes] = []
    cache_path = Path(task["cache_dir"]) / f"cache-{worker:03d}.sqlite3"
    started = time.perf_counter()
    with ExpertCache(cache_path, board_size, int(task["max_candidates"]), seed,
                     top_k=int(task.get("expert_top_k", 4)),
                     temperature=float(task.get("expert_temperature", 1.5)),
                     stochastic_moves=int(task.get("expert_stochastic_moves", 6)),
                     shared_path=(Path(task["shared_cache"])
                                  if task.get("shared_cache") else None)) as expert:
        for local_game in range(int(task["games"])):
            game_id = (worker << 32) | local_game
            env = GomokuEnv(board_size=board_size, starting_player="black",
                            illegal_action_mode="raise")
            obs, _ = env.reset(seed=seed + local_game)
            # policy_expert / policy_history 中交替当前策略的颜色，避免只覆盖单边状态。
            learner_player = 1 if (worker + local_game) % 2 == 0 else -1
            source = str(task.get("behavior_source") or
                         _behavior_source(mode, worker, local_game))
            random_prefix = int(policy_rng.integers(2, 11)) if source == "perturbed_opening" else 0
            history = None
            # 历史对手按局抽取，并在 worker 内复用模型，避免反复加载 checkpoint。
            if source == "policy_history" and history_paths:
                history_path = history_paths[int(policy_rng.integers(len(history_paths)))]
                history = history_agents.get(history_path)
                if history is None:
                    history = BCAgent(board_size, device="cpu")
                    history.load_checkpoint(history_path); history.net.eval()
                    history_agents[history_path] = history
            start = len(boards); done = False
            while not done:
                board = obs["board"].copy(); player = int(obs["current_player"][0])
                # 无论实际由谁落子，都先查询同一个冻结专家并保存监督答案。
                decision = expert.decision(board, player)
                encoded, candidate_count, reason, tactical = encode_decision(
                    decision, DEFAULT_ORACLE_TOP_K
                )
                expert_action = int(decision.actions[0])
                actual_action = expert_action
                move_index = int(np.count_nonzero(board == player))
                # actual_action 只决定后续状态来自哪个分布，不会覆盖 expert_action 标签。
                if source == "perturbed_opening" and int(np.count_nonzero(board)) < random_prefix:
                    actual_action = _local_random_action(board, obs["action_mask"], policy_rng)
                elif source == "epsilon_expert":
                    if not tactical and policy_rng.random() < 0.15:
                        actual_action = _local_random_action(board, obs["action_mask"], policy_rng)
                    else:
                        actual_action = rank_softmax_action(
                            decision.actions, move_index, policy_rng,
                            top_k=int(task.get("expert_top_k", 4)),
                            temperature=float(task.get("expert_temperature", 1.5)),
                            stochastic_moves=int(task.get("expert_stochastic_moves", 6)),
                            force_greedy=tactical,
                        )
                elif source == "expert_selfplay":
                    actual_action = rank_softmax_action(
                        decision.actions, move_index, policy_rng,
                        top_k=int(task.get("expert_top_k", 4)),
                        temperature=float(task.get("expert_temperature", 1.5)),
                        stochastic_moves=int(task.get("expert_stochastic_moves", 6)),
                        force_greedy=tactical,
                    )
                elif source == "policy_expert":
                    if player == learner_player:
                        actual_action = _policy_action(
                            policy, board, player, obs["action_mask"], policy_rng, task
                        )
                elif source == "policy_selfplay":
                    actual_action = _policy_action(
                        policy, board, player, obs["action_mask"], policy_rng, task
                    )
                elif source == "policy_history":
                    actor = policy if player == learner_player or history is None else history
                    actual_action = _policy_action(
                        actor, board, player, obs["action_mask"], policy_rng, task
                    )
                elif source == "epsilon_policy":
                    if not tactical and policy_rng.random() < 0.10:
                        actual_action = _local_random_action(board, obs["action_mask"], policy_rng)
                    else:
                        actual_action = _policy_action(
                            policy, board, player, obs["action_mask"], policy_rng, task
                        )
                # 保存专家答案、实际行为和审计字段。canonical key 将旋转/镜像等价
                # 状态合并，用于跨轮去重、覆盖统计和 challenge 隔离。
                boards.append(board); players.append(player); actions.append(expert_action)
                candidate_actions.append(encoded); candidate_counts.append(candidate_count)
                tacticals.append(tactical); reasons.append(reason)
                plies.append(int(np.count_nonzero(board))); sources.append(SOURCE_TO_ID[source])
                rounds.append(int(task.get("dagger_round", 0)))
                behavior_actions.append(actual_action)
                canonical_keys.append(canonicalize(board, player)[0])
                game_ids.append(game_id)
                obs, _, terminated, truncated, _ = env.step(actual_action)
                done = terminated or truncated
            env.close()
            # 同一盘棋共享轨迹 hash；dataset.py 按整条轨迹切分训练/验证集，
            # 防止相邻状态泄漏到两个 split。
            group = canonical_trajectory_hash(np.asarray(boards[start:]),
                                              np.asarray(players[start:]))
            trajectory_groups.extend([group] * (len(boards) - start))
        hits, misses, expert_queries = expert.hits, expert.misses, expert.expert_queries

    labeling_seconds = time.perf_counter() - started
    # v3 shard 字段契约。紧凑整数类型可显著降低大规模数据的磁盘占用。
    arrays = {"boards": np.asarray(boards, dtype=np.int8),
              "players": np.asarray(players, dtype=np.int8),
              "actions": np.asarray(actions, dtype=np.int16),
              "candidate_actions": np.asarray(candidate_actions, dtype=np.int16),
              "candidate_counts": np.asarray(candidate_counts, dtype=np.int8),
              "tactical": np.asarray(tacticals, dtype=np.bool_),
              "reasons": np.asarray(reasons, dtype=np.int8),
              "plies": np.asarray(plies, dtype=np.int16),
              "sources": np.asarray(sources, dtype=np.int8),
              "rounds": np.asarray(rounds, dtype=np.int8),
              "behavior_actions": np.asarray(behavior_actions, dtype=np.int16),
              "canonical_keys": np.asarray(canonical_keys, dtype="S32"),
              "games": np.asarray(game_ids, dtype=np.int64),
              "trajectory_groups": np.asarray(trajectory_groups, dtype="S32")}
    # shard 也采用原子提交，metadata 不会引用半写状态的 NPZ。
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
    """解析单轮生成规模、覆盖目标、专家身份和受控采样参数。"""
    parser = argparse.ArgumentParser(description="Generate offline Gomoku expert labels.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("expert", "aggregate", "bootstrap", "dagger"),
                        default="expert")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--min-games", type=int, default=0)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--target-new-states", type=int, default=0)
    parser.add_argument("--target-win-states", type=int, default=0)
    parser.add_argument("--target-block-states", type=int, default=0)
    parser.add_argument("--target-fork-states", type=int, default=0)
    parser.add_argument("--batch-games", type=int, default=250)
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
    parser.add_argument("--bc-stochastic-moves", type=int, default=8)
    parser.add_argument("--dagger-round", type=int, default=0)
    parser.add_argument("--history-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--previous-data-dir", type=Path, action="append", default=[])
    parser.add_argument("--shared-cache", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--tb-dir", type=Path, default=None)
    parser.add_argument("--quality-gate", action="store_true")
    parser.add_argument("--quality-min-games", type=int, default=100)
    parser.add_argument("--min-effective-trajectory-ratio", type=float, default=0.01)
    parser.add_argument("--max-dominant-trajectory-fraction", type=float, default=0.50)
    parser.add_argument("--min-state-unique-ratio", type=float, default=0.001)
    return parser.parse_args()


def _identity(args: argparse.Namespace, workers: int) -> dict[str, Any]:
    """构造数据集不可变身份，恢复时用它拒绝混入不同配置。"""
    return {"format_version": DATA_FORMAT_VERSION, "mode": args.mode,
            "board_size": args.board_size,
            "seed": args.seed, "games": args.games, "workers": workers,
            "min_games": args.min_games, "max_games": args.max_games,
            "target_new_states": args.target_new_states,
            "target_win_states": args.target_win_states,
            "target_block_states": args.target_block_states,
            "target_fork_states": args.target_fork_states,
            "batch_games": args.batch_games, "dagger_round": args.dagger_round,
            "max_candidates": args.max_candidates, "expert_top_k": args.expert_top_k,
            "expert_temperature": args.expert_temperature,
            "expert_stochastic_moves": args.expert_stochastic_moves,
            "bc_top_k": args.bc_top_k, "bc_temperature": args.bc_temperature,
            "bc_stochastic_moves": args.bc_stochastic_moves,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "history_checkpoints": [str(path.resolve()) for path in args.history_checkpoint],
            "previous_data_dirs": [str(path.resolve()) for path in args.previous_data_dir],
            "oracle": oracle_identity(max_candidates=args.max_candidates,
                                      top_k=args.expert_top_k)}


def _load_canonical_keys(data_dirs: list[Path]) -> set[bytes]:
    """读取旧轮次状态，用于判断本轮真正新增了多少 canonical 状态。"""
    keys: set[bytes] = set()
    for root in data_dirs:
        metadata = json.loads((Path(root) / "metadata.json").read_text())
        for name in metadata.get("shards", []):
            with np.load(Path(root) / name) as data:
                if "canonical_keys" in data:
                    plies = data["plies"] if "plies" in data else \
                        np.count_nonzero(data["boards"], axis=(1, 2))
                    keys.update(bytes(key) for key in data["canonical_keys"][plies >= 2])
    return keys


def _shard_new_keys(paths: list[Path], previous: set[bytes]) -> set[bytes]:
    """返回当前 shard 中历史未出现的状态；忽略高度重复的前两手开局。"""
    keys: set[bytes] = set()
    for path in paths:
        with np.load(path) as data:
            plies = data["plies"] if "plies" in data else \
                np.count_nonzero(data["boards"], axis=(1, 2))
            keys.update(bytes(key) for key in data["canonical_keys"][plies >= 2]
                        if bytes(key) not in previous)
    return keys


def _shard_coverage(paths: list[Path], previous: set[bytes]) -> dict[str, int]:
    """汇总新增状态以及 win/block/fork 战术样本覆盖。"""
    new_keys: set[bytes] = set()
    counts = {"win": 0, "block": 0, "fork": 0}
    for path in paths:
        with np.load(path) as data:
            plies = data["plies"]
            new_keys.update(bytes(key) for key in data["canonical_keys"][plies >= 2]
                            if bytes(key) not in previous)
            reasons = np.asarray(data["reasons"])
            counts["win"] += int(np.count_nonzero(reasons == REASON_TO_ID["win"]))
            counts["block"] += int(np.count_nonzero(reasons == REASON_TO_ID["block"]))
            counts["fork"] += int(np.count_nonzero(
                (reasons == REASON_TO_ID["own_fork"])
                | (reasons == REASON_TO_ID["block_fork"])
            ))
    return {"new_canonical_states": len(new_keys), **counts}


def _write_tensorboard(args: argparse.Namespace, metadata: dict[str, Any], diversity: dict[str, Any],
                       quality: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """记录数据吞吐、缓存命中、覆盖和多样性，供定位生成阶段问题。"""
    if args.tb_dir is None:
        return
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=str(args.tb_dir))
    writer.add_text("Pipeline/config", json.dumps(_identity(args, metadata["workers"]),
                                                   ensure_ascii=False, indent=2), 0)
    writer.add_scalar("Data/games", args.games, 0); writer.add_scalar("Data/samples", metadata["samples"], 0)
    writer.add_scalar("Data/cache_hit_rate", metadata["cache_hit_rate"], 0)
    writer.add_scalar("Data/expert_queries", metadata["expert_queries"], 0)
    writer.add_scalar("Data/new_canonical_states", metadata.get("new_canonical_states", 0), 0)
    for tag, key in (("canonical_effective_trajectory_ratio", "canonical_effective_trajectory_ratio"),
                     ("dominant_canonical_trajectory_fraction", "dominant_canonical_trajectory_fraction"),
                     ("canonical_state_unique_ratio", "canonical_state_unique_ratio")):
        writer.add_scalar(f"Diversity/{tag}", diversity[key], 0)
    writer.add_scalar("Diversity/quality_gate_passed", float(quality["passed"]), 0)
    writer.add_scalar("Diversity/canonical_state_duplicate_fraction",
                      diversity.get("canonical_state_duplicate_fraction", 0), 0)
    writer.add_scalar("Diversity/maximum_state_visit_fraction",
                      diversity.get("maximum_state_visit_fraction", 0), 0)
    writer.add_scalar("Diversity/top1_action_entropy",
                      diversity.get("top1_action_entropy", 0), 0)
    for phase, values in diversity.get("phase_coverage", {}).items():
        for metric in ("samples", "canonical_unique_count", "canonical_effective_count",
                       "canonical_entropy"):
            writer.add_scalar(f"PhaseCoverage/{phase}/{metric}", values[metric], 0)
    for group, values in (("Sources", diversity.get("source_counts", {})),
                          ("Reasons", diversity.get("reason_counts", {})),
                          ("Players", diversity.get("player_counts", {}))):
        for name, value in values.items():
            writer.add_scalar(f"{group}/{name}", value, 0)
    writer.add_text("Diversity/details", json.dumps({**diversity, "quality": quality},
                                                     ensure_ascii=False, indent=2), 0)
    for index, result in enumerate(results):
        writer.add_scalar("Workers/samples_per_second", result.get("samples_per_second", 0), index)
        writer.add_scalar("Workers/expert_queries_per_second", result.get("queries_per_second", 0), index)
        writer.add_scalar("Workers/write_seconds", result.get("write_seconds", 0), index)
    writer.close()


def main() -> None:
    """分批启动 worker，达到覆盖目标后汇总并冻结本轮数据集。"""
    args = parse_args()
    if (not 5 <= args.board_size <= 9 or args.games < 1 or args.workers < 1
            or args.batch_games < 1):
        raise ValueError("board size must be 5..9 and games/workers positive")
    if min(args.expert_top_k, args.bc_top_k) < 1 or min(args.expert_temperature, args.bc_temperature) < 0:
        raise ValueError("top-k must be positive and temperatures non-negative")
    # DAgger 必须有行为策略；Bootstrap 则不依赖 checkpoint。
    if args.mode in ("aggregate", "dagger") and args.checkpoint is None:
        raise ValueError("--checkpoint is required in aggregate mode")
    # metadata 是恢复契约：已有目录只能使用完全相同的 identity 继续生成；
    # status=complete 后数据集视为不可变，禁止覆盖。
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
    # target_new_states 只统计此前所有轮次都没见过的 canonical 状态。
    previous_keys = _load_canonical_keys(args.previous_data_dir)
    adaptive = args.target_new_states > 0
    min_games = (args.min_games or (2_000 if args.mode == "bootstrap" else 1_000)) \
        if adaptive else 0
    max_games = args.max_games or args.games
    total_goal = max_games if adaptive else args.games
    if adaptive and min_games > total_goal:
        raise ValueError("--min-games cannot exceed the generation game limit")
    results: list[dict[str, Any]] = []
    # adaptive 模式按 wave 分批生成；每个 wave 后检查是否已满足全部覆盖目标。
    with ProcessPoolExecutor(max_workers=workers) as pool:
        generated_games = 0; task_index = 0
        while generated_games < total_goal:
            wave = []
            for _ in range(workers):
                if generated_games >= total_goal:
                    break
                adaptive_task_games = max(1, math.ceil(args.batch_games / workers))
                count = (min(adaptive_task_games, total_goal - generated_games)
                         if adaptive else
                         args.games // workers + (task_index < args.games % workers))
                if count <= 0:
                    break
                task = dict(vars(args), worker=task_index, games=count,
                            output=str(args.output), cache_dir=str(cache_dir),
                            history_checkpoints=[str(path) for path in args.history_checkpoint],
                            shared_cache=str(args.shared_cache) if args.shared_cache else None,
                            data_format_version=DATA_FORMAT_VERSION)
                wave.append(pool.submit(_worker, task))
                generated_games += count; task_index += 1
                if not adaptive and task_index >= workers:
                    break
            for future in as_completed(wave):
                result = future.result(); results.append(result)
                print(f"{result['shard']}: games={result['games']} samples={result['samples']} "
                      f"cache={result['hits']}/{result['hits'] + result['misses']} "
                      f"seconds={result['seconds']:.1f} q/s={result.get('queries_per_second', 0):.1f}", flush=True)
            if adaptive and generated_games >= min_games:
                paths = [args.output / item["shard"] for item in results]
                coverage = _shard_coverage(paths, previous_keys)
                print(f"coverage: games={generated_games} "
                      f"new={coverage['new_canonical_states']}/{args.target_new_states} "
                      f"win={coverage['win']}/{args.target_win_states} "
                      f"block={coverage['block']}/{args.target_block_states} "
                      f"fork={coverage['fork']}/{args.target_fork_states}", flush=True)
                if (coverage["new_canonical_states"] >= args.target_new_states
                        and coverage["win"] >= args.target_win_states
                        and coverage["block"] >= args.target_block_states
                        and coverage["fork"] >= args.target_fork_states):
                    break
            if not adaptive:
                break
    results.sort(key=lambda item: item["shard"])
    shards = [args.output / result["shard"] for result in results]
    coverage = _shard_coverage(shards, previous_keys)
    new_state_count = coverage["new_canonical_states"]
    # 合并 worker 缓存供下一轮只读复用；随后做跨 worker 的全局多样性分析。
    shared_output = args.output / "cache" / "shared.sqlite3"
    merge_caches(shared_output, sorted(cache_dir.glob("cache-*.sqlite3")))
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
                     "generated_games": sum(r["games"] for r in results),
                     "samples": sum(r["samples"] for r in results), "cache_hits": hits,
                     "cache_misses": misses, "cache_hit_rate": hits / max(1, hits + misses),
                     "expert_queries": sum(r["expert_queries"] for r in results),
                     "new_canonical_states": new_state_count,
                     "coverage_counts": coverage,
                     "coverage_stalled": bool(
                         args.target_new_states and (
                             new_state_count < args.target_new_states
                             or coverage["win"] < args.target_win_states
                             or coverage["block"] < args.target_block_states
                             or coverage["fork"] < args.target_fork_states
                         )),
                     "shared_cache": str(shared_output.relative_to(args.output)),
                     "diversity_file": "diversity.json", "diversity": diversity,
                     "quality": quality, "worker_results": results, "completed_at": time.time()})
    atomic_json(metadata_path, metadata); _write_tensorboard(args, metadata, diversity, quality, results)
    if args.quality_gate and not quality["passed"]:
        raise RuntimeError("dataset rejected by diversity quality gate: " + "; ".join(quality["failures"]))
    print(f"complete: samples={metadata['samples']} cache_hit_rate={metadata['cache_hit_rate']:.3f} "
          f"expert_queries={metadata['expert_queries']}")


if __name__ == "__main__":
    main()
