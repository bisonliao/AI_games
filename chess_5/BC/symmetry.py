"""棋盘 D4 对称变换，以及专家查询/状态去重使用的 canonical key。"""

from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np


def transform_board(board: np.ndarray, transform: int) -> np.ndarray:
    """执行 4 种旋转及其水平镜像，共 8 种保持棋理等价的变换。"""
    if transform not in range(8):
        raise ValueError("transform must be in [0, 7]")
    result = np.rot90(np.asarray(board), transform % 4)
    if transform >= 4:
        result = np.fliplr(result)
    return np.ascontiguousarray(result)


def transform_action(action: int, size: int, transform: int) -> int:
    """将展平动作下标同步映射到变换后的棋盘坐标。"""
    marker = np.zeros((size, size), dtype=np.uint8)
    marker.flat[int(action)] = 1
    return int(np.flatnonzero(transform_board(marker, transform))[0])


def inverse_action(action: int, size: int, transform: int) -> int:
    """把 canonical 棋盘上的动作还原到调用方原棋盘坐标。"""
    marker = np.zeros((size, size), dtype=np.uint8)
    marker.flat[int(action)] = 1
    target = np.flatnonzero(marker)[0]
    for candidate in range(size * size):
        if transform_action(candidate, size, transform) == target:
            return candidate
    raise ValueError("action has no inverse")


def canonicalize(board: np.ndarray, player: int) -> Tuple[bytes, np.ndarray, int]:
    """选择 8 个对称棋盘中字节序最小者作为唯一表示。

    key 同时包含棋盘尺寸和当前行动方，避免相同落子布局在不同视角下错误复用标签。
    返回的 transform 描述原棋盘如何变成 canonical 棋盘。
    """
    variants = [(transform_board(board, t).tobytes(), t) for t in range(8)]
    raw, transform = min(variants, key=lambda item: item[0])
    canonical = np.frombuffer(raw, dtype=np.int8).reshape(np.asarray(board).shape).copy()
    header = bytes((board.shape[0], int(player) + 1))
    return hashlib.sha256(header + raw).digest(), canonical, transform
