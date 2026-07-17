"""Dihedral board transforms and canonical expert-query keys."""

from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np


def transform_board(board: np.ndarray, transform: int) -> np.ndarray:
    if transform not in range(8):
        raise ValueError("transform must be in [0, 7]")
    result = np.rot90(np.asarray(board), transform % 4)
    if transform >= 4:
        result = np.fliplr(result)
    return np.ascontiguousarray(result)


def transform_action(action: int, size: int, transform: int) -> int:
    marker = np.zeros((size, size), dtype=np.uint8)
    marker.flat[int(action)] = 1
    return int(np.flatnonzero(transform_board(marker, transform))[0])


def inverse_action(action: int, size: int, transform: int) -> int:
    marker = np.zeros((size, size), dtype=np.uint8)
    marker.flat[int(action)] = 1
    target = np.flatnonzero(marker)[0]
    for candidate in range(size * size):
        if transform_action(candidate, size, transform) == target:
            return candidate
    raise ValueError("action has no inverse")


def canonicalize(board: np.ndarray, player: int) -> Tuple[bytes, np.ndarray, int]:
    variants = [(transform_board(board, t).tobytes(), t) for t in range(8)]
    raw, transform = min(variants, key=lambda item: item[0])
    canonical = np.frombuffer(raw, dtype=np.int8).reshape(np.asarray(board).shape).copy()
    header = bytes((board.shape[0], int(player) + 1))
    return hashlib.sha256(header + raw).digest(), canonical, transform
