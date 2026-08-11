"""数据生成和对弈共用的排名采样，以及一步必胜/必堵检测。"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rank_softmax_action(ranked_actions: Sequence[int], move_index: int,
                        rng: np.random.Generator, *, top_k: int = 4,
                        temperature: float = 1.0, stochastic_moves: int = 6,
                        force_greedy: bool = False) -> int:
    """按候选排名采样，并随本方落子数逐渐退火到 greedy。

    采样只依赖排名而非 logit 绝对尺度，避免不同 checkpoint 的 logit 标定差异
    改变探索强度。达到 stochastic_moves 或遇到强制战术时直接选择第一候选。
    """
    actions = np.asarray(ranked_actions, dtype=np.int64).reshape(-1)
    if actions.size == 0:
        raise RuntimeError("No legal actions available")
    if top_k < 1 or temperature < 0 or stochastic_moves < 0:
        raise ValueError("invalid controlled-sampling parameters")
    if (force_greedy or top_k == 1 or temperature == 0 or stochastic_moves == 0
            or move_index >= stochastic_moves):
        return int(actions[0])
    # decay 同时缩小候选数和温度，使越接近中盘的行为越确定。
    decay = 1.0 - max(0, move_index) / float(stochastic_moves)
    active_k = 1 + int(np.ceil((min(top_k, actions.size) - 1) * decay))
    candidates = actions[:active_k]
    rank_logits = -np.arange(active_k, dtype=np.float64) / (temperature * decay)
    rank_logits -= rank_logits.max()
    probabilities = np.exp(rank_logits); probabilities /= probabilities.sum()
    return int(rng.choice(candidates, p=probabilities))


def ranked_legal_actions(values: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """过滤非法位置，并稳定地按策略分数从高到低返回动作。"""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(action_mask, dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError("values and action_mask must have the same shape")
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        raise RuntimeError("No legal actions available")
    if not np.all(np.isfinite(values[legal])):
        raise RuntimeError("policy produced non-finite values for legal actions")
    return legal[np.argsort(-values[legal], kind="stable")]


def _has_five_from(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """只从刚落子的点向四个方向检查连续五子。"""
    size = board.shape[0]
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                count += 1; r += sign * dr; c += sign * dc
        if count >= 5:
            return True
    return False


def immediate_winning_moves(board: np.ndarray, player: int,
                            action_mask: np.ndarray | None = None) -> list[int]:
    """枚举当前方一步即可形成五连的全部合法动作。"""
    position = np.asarray(board, dtype=np.int8)
    mask = position.reshape(-1) == 0 if action_mask is None else \
        np.asarray(action_mask, dtype=bool).reshape(-1) & (position.reshape(-1) == 0)
    wins: list[int] = []
    for action in np.flatnonzero(mask):
        row, col = divmod(int(action), position.shape[0])
        position[row, col] = player
        if _has_five_from(position, row, col, player):
            wins.append(int(action))
        position[row, col] = 0
    return wins


def has_immediate_win_or_block(board: np.ndarray, player: int,
                               action_mask: np.ndarray) -> bool:
    """当前方能一步获胜或对手下一步能获胜时，要求策略停止随机采样。"""
    return bool(immediate_winning_moves(board, player, action_mask)
                or immediate_winning_moves(board, -player, action_mask))
