"""The shared controlled-stochastic BC_BEST black policy protocol."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


BLACK_TOP_K = 4
BLACK_TEMPERATURE = 1.5
BLACK_STOCHASTIC_MOVES = 6


def _has_five_from(board: np.ndarray, row: int, col: int, player: int) -> bool:
    size = board.shape[0]
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for sign in (-1, 1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < size and 0 <= c < size and board[r, c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= 5:
            return True
    return False


def immediate_winning_moves(board: np.ndarray, player: int, mask: np.ndarray) -> list[int]:
    position = np.asarray(board, dtype=np.int8).copy()
    wins: list[int] = []
    for raw_action in np.flatnonzero(np.asarray(mask, dtype=np.bool_)):
        action = int(raw_action)
        row, col = divmod(action, position.shape[0])
        position[row, col] = player
        if _has_five_from(position, row, col, player):
            wins.append(action)
        position[row, col] = 0
    return wins


def tactical_position(board: np.ndarray, player: int, mask: np.ndarray) -> bool:
    own_threat = (
        np.count_nonzero(board == player) >= 4
        and immediate_winning_moves(board, player, mask)
    )
    opponent_threat = (
        np.count_nonzero(board == -player) >= 4
        and immediate_winning_moves(board, -player, mask)
    )
    return bool(own_threat or opponent_threat)


def controlled_black_actions(
    policy: Any,
    boards: np.ndarray,
    masks: np.ndarray,
    rngs: Sequence[np.random.Generator],
    *,
    stochastic: bool = True,
) -> np.ndarray:
    """Select black actions under the exact protocol shared by train and eval."""
    boards = np.asarray(boards, dtype=np.int8)
    masks = np.asarray(masks, dtype=np.bool_)
    if len(boards) != len(masks) or len(boards) != len(rngs):
        raise ValueError("boards, masks, and rngs must have equal batch size")
    values = policy.values(boards, np.ones(len(boards), dtype=np.int8))
    actions = np.empty(len(boards), dtype=np.int64)
    for index, (board, mask, rng) in enumerate(zip(boards, masks, rngs)):
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise RuntimeError("black has no legal action")
        ranked = legal[np.argsort(-values[index, legal], kind="stable")]
        move_index = int(np.count_nonzero(board == 1))
        can_sample = stochastic and move_index < BLACK_STOCHASTIC_MOVES
        force_greedy = can_sample and tactical_position(board, 1, mask)
        if not can_sample or force_greedy:
            actions[index] = int(ranked[0])
            continue
        candidates = ranked[:min(BLACK_TOP_K, len(ranked))]
        logits = -np.arange(len(candidates), dtype=np.float64) / BLACK_TEMPERATURE
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        actions[index] = int(rng.choice(candidates, p=probabilities))
    return actions


def protocol_description() -> str:
    return (
        f"BC_BEST top-{BLACK_TOP_K} rank sampling, temperature "
        f"{BLACK_TEMPERATURE:g}, first {BLACK_STOCHASTIC_MOVES} black moves; "
        "tactical positions force greedy"
    )
