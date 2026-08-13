import numpy as np

from TetrisEnv.pieces import PIECES
from TetrisEnv.tetris_env import SevenBag


def test_seven_bag_contains_each_piece_once():
    bag = SevenBag(np.random.default_rng(0))
    values = [bag.next() for _ in range(14)]
    assert set(values[:7]) == set(PIECES)
    assert set(values[7:]) == set(PIECES)


def test_seeded_bag_is_reproducible():
    a = SevenBag(np.random.default_rng(123))
    b = SevenBag(np.random.default_rng(123))
    assert [a.next() for _ in range(21)] == [b.next() for _ in range(21)]
