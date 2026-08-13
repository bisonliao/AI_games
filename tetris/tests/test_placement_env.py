import numpy as np
import pytest

from TetrisEnv.placement_env import PlacementTetrisEnv
from TetrisEnv.pieces import PIECES, ActivePiece
from TetrisEnv.vector_runner import make_sync_vector_env


def _set_piece(env: PlacementTetrisEnv, kind: str) -> None:
    env.current = ActivePiece(kind, 0, 3, -2)
    env._terminated = False


def test_placement_env_checker_and_mask():
    env = PlacementTetrisEnv()
    obs, _ = env.reset(seed=4)
    assert env.observation_space.contains(obs)
    assert obs["action_mask"].shape == (40,)
    assert obs["action_mask"].dtype == np.int8
    assert obs["action_mask"].any()


@pytest.mark.parametrize(
    ("kind", "expected_rotations"),
    [("O", 1), ("I", 2), ("S", 2), ("Z", 2), ("T", 4), ("J", 4), ("L", 4)],
)
def test_mask_removes_duplicate_piece_rotations(kind, expected_rotations):
    env = PlacementTetrisEnv()
    env.reset(seed=0)
    _set_piece(env, kind)
    mask = env.action_mask().reshape(4, 10)
    assert np.count_nonzero(mask.any(axis=1)) == expected_rotations


def test_one_step_rotates_moves_drops_and_spawns_next_piece():
    env = PlacementTetrisEnv()
    env.reset(seed=0)
    _set_piece(env, "I")
    action = env.encode_action(rotation=1, target_column=0)
    assert env.action_mask()[action]
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert truncated is False
    assert terminated is False
    assert info["piece_placed"] is True
    assert info["survival_pieces"] == 1
    assert info["placement_rotation"] == 1
    assert info["placement_target_column"] == 0
    assert np.count_nonzero(next_obs["board"][:, 0]) == 4
    assert np.isfinite(reward)
    assert next_obs["current_piece"].sum() == 1


def test_placement_action_clears_a_completed_line_immediately():
    env = PlacementTetrisEnv()
    env.reset(seed=0)
    _set_piece(env, "I")
    env.board.grid[-1, 4:] = 1
    action = env.encode_action(rotation=0, target_column=0)
    next_obs, reward, terminated, _, info = env.step(action)
    assert terminated is False
    assert info["lines_cleared"] == 1
    assert np.count_nonzero(next_obs["board"][-1]) == 0
    assert reward > 0.5


def test_masked_action_is_rejected_without_changing_board():
    env = PlacementTetrisEnv()
    env.reset(seed=0)
    _set_piece(env, "O")
    board_before = env.board.grid.copy()
    duplicate_rotation = env.encode_action(rotation=1, target_column=0)
    assert not env.action_mask()[duplicate_rotation]
    with pytest.raises(ValueError, match="masked placement action"):
        env.step(duplicate_rotation)
    assert np.array_equal(env.board.grid, board_before)


def test_every_piece_always_has_a_legal_action_on_empty_board():
    env = PlacementTetrisEnv()
    env.reset(seed=0)
    for kind in PIECES:
        _set_piece(env, kind)
        assert env.action_mask().any()


def test_vector_runner_constructs_placement_environments():
    env = make_sync_vector_env(2, seeds=(10, 20))
    try:
        obs, _ = env.reset(seed=[10, 20])
        actions = np.asarray([np.flatnonzero(mask)[0] for mask in obs["action_mask"]])
        next_obs, _, _, _, infos = env.step(actions)
        assert next_obs["action_mask"].shape == (2, 40)
        assert np.all(infos["piece_placed"])
    finally:
        env.close()


def test_vector_runner_same_step_autoreset_returns_a_legal_mask():
    env = make_sync_vector_env(1, seeds=(10,))
    try:
        obs, _ = env.reset(seed=[10])
        for _ in range(100):
            action = np.asarray([np.flatnonzero(obs["action_mask"][0])[0]])
            obs, _, terminated, _, infos = env.step(action)
            if terminated[0]:
                assert obs["action_mask"][0].any()
                assert "final_info" in infos
                break
        else:
            pytest.fail("deterministic stacking did not terminate within 100 placements")
    finally:
        env.close()
