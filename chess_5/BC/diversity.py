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
            samples += len(boards)
            for start, end in _game_ranges(game_ids):
                hashers = [hashlib.sha256() for _ in range(8)]
                for board, player in zip(boards[start:end], players[start:end]):
                    variants = [transform_board(board, transform).tobytes() for transform in range(8)]
                    player_byte = bytes((int(player) + 1,))
                    canonical_states.add(hashlib.sha256(player_byte + min(variants)).digest())
                    for hasher, raw in zip(hashers, variants):
                        hasher.update(player_byte)
                        hasher.update(raw)
                # Taking the minimum over all global board transforms makes the
                # trajectory identity invariant to rotations and reflections.
                trajectory_counts[min(hasher.digest() for hasher in hashers)] += 1
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
