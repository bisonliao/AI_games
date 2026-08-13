import pytest

from TetrisEnv.board import BoardFeatures
from TetrisEnv.reward import potential, shaped_reward


EMPTY = BoardFeatures(
    aggregate_height=0,
    max_height=0,
    holes=0,
    bumpiness=0,
    wells=0,
)


def test_placement_and_line_rewards_use_rebalanced_scale():
    reward = shaped_reward(
        piece_placed=True,
        lines_cleared=1,
        terminated=False,
        previous=EMPTY,
        current=EMPTY,
    )
    assert reward == pytest.approx(0.76)


def test_one_new_hole_outweighs_ordinary_placement_reward():
    one_hole = BoardFeatures(
        aggregate_height=0,
        max_height=0,
        holes=1,
        bumpiness=0,
        wells=0,
    )
    assert potential(one_hole) == pytest.approx(-0.02)
    reward = shaped_reward(
        piece_placed=True,
        lines_cleared=0,
        terminated=False,
        previous=EMPTY,
        current=one_hole,
    )
    assert reward < 0


def test_terminal_state_keeps_real_board_potential():
    one_hole = BoardFeatures(
        aggregate_height=0,
        max_height=0,
        holes=1,
        bumpiness=0,
        wells=0,
    )
    reward = shaped_reward(
        piece_placed=True,
        lines_cleared=0,
        terminated=True,
        previous=one_hole,
        current=one_hole,
    )
    assert reward == pytest.approx(-0.9898)
