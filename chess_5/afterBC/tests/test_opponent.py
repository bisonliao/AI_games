from __future__ import annotations

import numpy as np

from afterBC.opponent import (
    BLACK_STOCHASTIC_MOVES,
    controlled_black_actions,
    protocol_description,
)


class FixedValuePolicy:
    def __init__(self, values: np.ndarray) -> None:
        self.fixed_values = np.asarray(values, dtype=np.float32)

    def values(self, boards, players):
        del players
        return np.repeat(self.fixed_values[None, :], len(boards), axis=0)


def test_controlled_black_sampling_is_seeded_and_limited_to_top_four() -> None:
    board = np.zeros((1, 9, 9), dtype=np.int8)
    mask = np.ones((1, 81), dtype=np.bool_)
    policy = FixedValuePolicy(-np.arange(81, dtype=np.float32))
    first = controlled_black_actions(
        policy, board, mask, [np.random.default_rng(123)], stochastic=True,
    )
    second = controlled_black_actions(
        policy, board, mask, [np.random.default_rng(123)], stochastic=True,
    )
    assert first.tolist() == second.tolist()
    assert int(first[0]) in {0, 1, 2, 3}


def test_controlled_black_sampling_forces_greedy_on_immediate_tactic() -> None:
    board = np.zeros((1, 9, 9), dtype=np.int8)
    board[0, 0, :4] = 1
    mask = board.reshape(1, -1) == 0
    values = np.zeros(81, dtype=np.float32)
    values[4] = 10.0
    values[20] = 9.0
    action = controlled_black_actions(
        FixedValuePolicy(values), board, mask,
        [np.random.default_rng(5)], stochastic=True,
    )
    assert action.tolist() == [4]


def test_controlled_black_is_greedy_after_stochastic_opening_window() -> None:
    board = np.zeros((1, 9, 9), dtype=np.int8)
    board.reshape(-1)[:BLACK_STOCHASTIC_MOVES] = 1
    mask = board.reshape(1, -1) == 0
    values = -np.arange(81, dtype=np.float32)
    action = controlled_black_actions(
        FixedValuePolicy(values), board, mask,
        [np.random.default_rng(5)], stochastic=True,
    )
    assert action.tolist() == [BLACK_STOCHASTIC_MOVES]
    assert "top-4" in protocol_description()
    assert "first 6" in protocol_description()
