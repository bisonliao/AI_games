"""Post-generation diversity metrics for sharded Gomoku datasets."""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .symmetry import transform_board


def _game_ranges(game_ids: np.ndarray) -> Iterable[tuple[int, int]]:
    """Yield contiguous [start, end) ranges; generated games are shard-contiguous."""
    if len(game_ids) == 0:
        return
    starts = np.r_[0, np.flatnonzero(game_ids[1:] != game_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(game_ids)]
    yield from zip(starts.tolist(), ends.tolist())


def canonical_trajectory_hash(boards: np.ndarray, players: np.ndarray) -> bytes:
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
    warnings: list[str] = []
    failures: list[str] = []
    checks = (
        (metrics["canonical_effective_trajectory_ratio"], 0.05, "below",
         "effective trajectory ratio"),
        (metrics["dominant_canonical_trajectory_fraction"], 0.20, "above",
         "dominant trajectory fraction"),
        (metrics["canonical_state_unique_ratio"], 0.005, "below", "state unique ratio"),
    )
    for value, threshold, direction, name in checks:
        bad = value < threshold if direction == "below" else value > threshold
        if bad:
            warnings.append(f"{name} {value:.6f} is {direction} warning threshold {threshold:.6f}")
    if metrics["games"] >= hard_min_games:
        if metrics["canonical_effective_trajectory_ratio"] < min_effective_ratio:
            failures.append("canonical effective trajectory ratio below hard minimum")
        if metrics["dominant_canonical_trajectory_fraction"] > max_dominant_fraction:
            failures.append("dominant canonical trajectory fraction above hard maximum")
        if metrics["canonical_state_unique_ratio"] < min_state_unique_ratio:
            failures.append("canonical state unique ratio below hard minimum")
    return {"passed": not failures, "warnings": warnings, "failures": failures,
            "thresholds": {"hard_min_games": hard_min_games,
                           "min_effective_trajectory_ratio": min_effective_ratio,
                           "max_dominant_trajectory_fraction": max_dominant_fraction,
                           "min_state_unique_ratio": min_state_unique_ratio}}


def analyze_shards(shards: Iterable[Path]) -> dict[str, Any]:
    """Compute three symmetry-aware coverage indicators in one dataset scan.

    Every board is transformed eight ways once. The same transformed bytes feed
    both the complete-trajectory hash and the canonical-state set.
    """
    started = time.perf_counter()
    trajectory_counts: Counter[bytes] = Counter()
    canonical_states: set[bytes] = set()
    samples = games = 0

    for path in shards:
        with np.load(path) as data:
            boards = np.asarray(data["boards"], dtype=np.int8)
            players = np.asarray(data["players"], dtype=np.int8)
            game_ids = np.asarray(data["games"], dtype=np.int64)
            stored_groups = data["trajectory_groups"] if "trajectory_groups" in data else None
            samples += len(boards)
            for start, end in _game_ranges(game_ids):
                for board, player in zip(boards[start:end], players[start:end]):
                    variants = [transform_board(board, transform).tobytes() for transform in range(8)]
                    player_byte = bytes((int(player) + 1,))
                    canonical_states.add(hashlib.sha256(player_byte + min(variants)).digest())
                group = (bytes(stored_groups[start]) if stored_groups is not None else
                         canonical_trajectory_hash(boards[start:end], players[start:end]))
                trajectory_counts[group] += 1
                games += 1

    if games:
        probabilities = np.asarray(list(trajectory_counts.values()), dtype=np.float64) / games
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        effective_count = float(math.exp(entropy))
        dominant_fraction = float(max(trajectory_counts.values()) / games)
    else:
        entropy = effective_count = dominant_fraction = 0.0

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
        "analysis_seconds": time.perf_counter() - started,
    }
