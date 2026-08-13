import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from TetrisEnv.tetris_env import TetrisEnv
from TetrisEnv.vector_runner import make_sync_vector_env


def test_env_checker_and_observation():
    env = TetrisEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    assert obs["board"].dtype == np.uint8


def test_hard_drop_places_piece_and_event_reward():
    env = TetrisEnv()
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(env.ACTION_HARD_DROP)
    assert info["piece_placed"] is True
    assert info["survival_pieces"] == 1
    assert np.isfinite(reward)
    assert truncated is False
    assert terminated is False


def test_vector_env_uses_distinct_reproducible_seeds():
    with pytest.raises(ValueError):
        make_sync_vector_env(2, seeds=(101, 101))
    first = make_sync_vector_env(2, seeds=(101, 202))
    second = make_sync_vector_env(2, seeds=(101, 202))
    try:
        first_obs, _ = first.reset(seed=[101, 202])
        second_obs, _ = second.reset(seed=[101, 202])
        assert np.array_equal(first_obs["current_piece"], second_obs["current_piece"])
        assert np.array_equal(first_obs["next_piece"], second_obs["next_piece"])
        assert not np.array_equal(first_obs["current_piece"][0], first_obs["current_piece"][1])
        first_actions = np.asarray([np.flatnonzero(mask)[0] for mask in first_obs["action_mask"]])
        second_actions = np.asarray([np.flatnonzero(mask)[0] for mask in second_obs["action_mask"]])
        first_obs, *_ = first.step(first_actions)
        second_obs, *_ = second.step(second_actions)
        first_actions = np.asarray([np.flatnonzero(mask)[0] for mask in first_obs["action_mask"]])
        second_actions = np.asarray([np.flatnonzero(mask)[0] for mask in second_obs["action_mask"]])
        first_obs, *_ = first.step(first_actions)
        second_obs, *_ = second.step(second_actions)
        assert np.array_equal(first_obs["current_piece"], second_obs["current_piece"])
    finally:
        first.close()
        second.close()
