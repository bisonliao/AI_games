import numpy as np

from TetrisEnv.board import Board
from TetrisEnv.pieces import ActivePiece


def test_clear_full_line():
    board = Board()
    board.grid[-1, :] = 1
    board.grid[-2, 0] = 1
    assert board.clear_lines() == 1
    assert board.grid[-1, 0] == 1
    assert np.count_nonzero(board.grid[-2]) == 0


def test_srs_rotation_stays_inside_board():
    board = Board()
    piece = ActivePiece("T", 0, -1, 0)
    rotated = board.try_rotate_cw(piece)
    assert rotated is not None
    assert not board.collides(rotated)


def test_board_features_include_aggregate_height_holes_and_wells():
    board = Board()
    board.grid[-2:, 0] = 1
    board.grid[-1, 1] = 1
    board.grid[-3:, 2] = 1
    board.grid[-2, 2] = 0
    features = board.features()
    assert features.aggregate_height == 6
    assert features.max_height == 3
    assert features.holes == 1
    assert features.bumpiness == 6
    assert features.wells == 1
