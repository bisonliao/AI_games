"""在数据生成后统计对称去重的轨迹、状态、阶段和来源多样性。"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .symmetry import transform_board
from .oracle import ID_TO_REASON, ID_TO_SOURCE


def _game_ranges(game_ids: np.ndarray) -> Iterable[tuple[int, int]]:
    """按连续 game id 产出每盘棋的 [start, end)；生成器保证同局状态连续。"""
    if len(game_ids) == 0:
        return
    starts = np.r_[0, np.flatnonzero(game_ids[1:] != game_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(game_ids)]
    yield from zip(starts.tolist(), ends.tolist())


def canonical_trajectory_hash(boards: np.ndarray, players: np.ndarray) -> bytes:
    """对整条棋局的 8 种全局对称分别 hash，取最小值作为轨迹身份。"""
    hashers = [hashlib.sha256() for _ in range(8)]
    for board, player in zip(boards, players):
        player_byte = bytes((int(player) + 1,))
        for transform, hasher in enumerate(hashers):
            hasher.update(player_byte)
            hasher.update(transform_board(board, transform).tobytes())
    return min(hasher.digest() for hasher in hashers)


def assess_diversity(metrics: dict[str, Any], *, hard_min_games: int = 100,
                     min_effective_ratio: float = 0.01,
                     max_dominant_fraction: float = 0.50,
                     min_state_unique_ratio: float = 0.001) -> dict[str, Any]:
    """将量化指标分成建议警告和会阻止训练的硬失败。"""
    warnings: list[str] = []
    failures: list[str] = []
    checks = (
        (metrics["canonical_effective_trajectory_ratio"], 0.05, "below",
         "有效轨迹比例"),
        (metrics["dominant_canonical_trajectory_fraction"], 0.20, "above",
         "最大单一轨迹占比"),
        (metrics["canonical_state_unique_ratio"], 0.005, "below", "独特状态比例"),
    )
    for value, threshold, direction, name in checks:
        bad = value < threshold if direction == "below" else value > threshold
        if bad:
            relation = "低于" if direction == "below" else "高于"
            warnings.append(f"{name} {value:.2%}，{relation}建议线 {threshold:.2%}")
    if metrics["games"] >= hard_min_games:
        if "phase_coverage" in metrics:
            # v3 使用自适应的绝对覆盖目标。数据规模增长后会反复访问有限状态分布，
            # unique ratio 自然下降，因此比例只做诊断，不再直接作为拒绝条件。
            if metrics["dominant_canonical_trajectory_fraction"] > max_dominant_fraction:
                failures.append(f"最大单一轨迹占比高于硬门槛 {max_dominant_fraction:.2%}")
            if metrics.get("maximum_state_visit_fraction", 0.0) > 0.10:
                failures.append("单一 canonical 状态占比高于硬门槛 10.00%")
            player_counts = metrics.get("player_counts", {})
            total_players = sum(player_counts.values())
            if total_players and min(player_counts.get("black", 0),
                                     player_counts.get("white", 0)) / total_players < 0.40:
                failures.append("黑白样本比例失衡，较少一方低于 40.00%")
        else:
            if metrics["canonical_effective_trajectory_ratio"] < min_effective_ratio:
                failures.append(f"有效轨迹比例低于硬门槛 {min_effective_ratio:.2%}")
            if metrics["dominant_canonical_trajectory_fraction"] > max_dominant_fraction:
                failures.append(f"最大单一轨迹占比高于硬门槛 {max_dominant_fraction:.2%}")
            if metrics["canonical_state_unique_ratio"] < min_state_unique_ratio:
                failures.append(f"独特状态比例低于硬门槛 {min_state_unique_ratio:.2%}")
    return {"passed": not failures, "warnings": warnings, "failures": failures,
            "thresholds": {"hard_min_games": hard_min_games,
                           "min_effective_trajectory_ratio": min_effective_ratio,
                           "max_dominant_trajectory_fraction": max_dominant_fraction,
                           "min_state_unique_ratio": min_state_unique_ratio}}


def analyze_shards(shards: Iterable[Path]) -> dict[str, Any]:
    """单次扫描全部 shard，计算对称感知的覆盖与分布指标。

    每个棋盘的 8 种变换同时用于完整轨迹 hash 和 canonical 状态计数；还按开局、
    中盘、残局分别统计有效状态数，并审计黑白、行为来源、标签原因和动作熵。
    """
    started = time.perf_counter()
    trajectory_counts: Counter[bytes] = Counter()
    canonical_states: set[bytes] = set()
    state_counts: Counter[bytes] = Counter()
    phase_state_counts: dict[str, Counter[bytes]] = {
        "opening_0_15": Counter(), "middle_16_35": Counter(), "late_36_plus": Counter()
    }
    player_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    candidate_total = 0
    samples = games = 0

    for path in shards:
        with np.load(path) as data:
            boards = np.asarray(data["boards"], dtype=np.int8)
            players = np.asarray(data["players"], dtype=np.int8)
            game_ids = np.asarray(data["games"], dtype=np.int64)
            stored_groups = data["trajectory_groups"] if "trajectory_groups" in data else None
            samples += len(boards)
            plies = (np.asarray(data["plies"], dtype=np.int16) if "plies" in data else
                     np.count_nonzero(boards, axis=(1, 2)).astype(np.int16))
            sources = data["sources"] if "sources" in data else np.zeros(len(boards), np.int8)
            reasons = data["reasons"] if "reasons" in data else \
                np.full(len(boards), 5, np.int8)
            candidate_counts = data["candidate_counts"] if "candidate_counts" in data else \
                np.ones(len(boards), np.int8)
            actions = data["actions"]
            for start, end in _game_ranges(game_ids):
                for offset, (board, player) in enumerate(zip(boards[start:end], players[start:end]),
                                                          start=start):
                    variants = [transform_board(board, transform).tobytes() for transform in range(8)]
                    player_byte = bytes((int(player) + 1,))
                    key = hashlib.sha256(player_byte + min(variants)).digest()
                    canonical_states.add(key); state_counts[key] += 1
                    phase = ("opening_0_15" if int(plies[offset]) <= 15 else
                             "middle_16_35" if int(plies[offset]) <= 35 else "late_36_plus")
                    phase_state_counts[phase][key] += 1
                    player_counts["black" if int(player) == 1 else "white"] += 1
                    source_counts[ID_TO_SOURCE.get(int(sources[offset]), "unknown")] += 1
                    reason_counts[ID_TO_REASON.get(int(reasons[offset]), "unknown")] += 1
                    action_counts[int(actions[offset])] += 1
                    candidate_total += int(candidate_counts[offset])
                group = (bytes(stored_groups[start]) if stored_groups is not None else
                         canonical_trajectory_hash(boards[start:end], players[start:end]))
                trajectory_counts[group] += 1
                games += 1

    # exp(Shannon entropy) 是“有效轨迹数”：大量重复轨迹只贡献接近 1 的有效数量。
    if games:
        probabilities = np.asarray(list(trajectory_counts.values()), dtype=np.float64) / games
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        effective_count = float(math.exp(entropy))
        dominant_fraction = float(max(trajectory_counts.values()) / games)
    else:
        entropy = effective_count = dominant_fraction = 0.0

    phase_metrics = {}
    for name, counts in phase_state_counts.items():
        count = sum(counts.values())
        if count:
            probabilities = np.asarray(list(counts.values()), dtype=np.float64) / count
            phase_entropy = float(-np.sum(probabilities * np.log(probabilities)))
        else:
            phase_entropy = 0.0
        phase_metrics[name] = {
            "samples": count, "canonical_unique_count": len(counts),
            "canonical_effective_count": float(math.exp(phase_entropy)) if count else 0.0,
            "canonical_entropy": phase_entropy,
        }
    if samples:
        action_probabilities = np.asarray(list(action_counts.values()), dtype=np.float64) / samples
        action_entropy = float(-np.sum(action_probabilities * np.log(action_probabilities)))
    else:
        action_entropy = 0.0
    return {
        "format_version": 1,
        "games": games,
        "samples": samples,
        "canonical_trajectory_unique_count": len(trajectory_counts),
        "canonical_trajectory_entropy": entropy,
        "canonical_effective_trajectory_count": effective_count,
        "canonical_effective_trajectory_ratio": effective_count / max(1, games),
        "dominant_canonical_trajectory_fraction": dominant_fraction,
        "canonical_state_unique_count": len(canonical_states),
        "canonical_state_unique_ratio": len(canonical_states) / max(1, samples),
        "canonical_state_duplicate_fraction": 1.0 - len(canonical_states) / max(1, samples),
        "maximum_state_visit_fraction": max(state_counts.values(), default=0) / max(1, samples),
        "phase_coverage": phase_metrics,
        "player_counts": dict(player_counts),
        "source_counts": dict(source_counts),
        "reason_counts": dict(reason_counts),
        "top1_action_entropy": action_entropy,
        "average_candidate_count": candidate_total / max(1, samples),
        "analysis_seconds": time.perf_counter() - started,
    }
