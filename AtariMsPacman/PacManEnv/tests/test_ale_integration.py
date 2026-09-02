from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from PacManEnv import MsPacmanEnvConfig, make_env, make_vector_env


pytest.importorskip("cv2")


def deterministic_config(**changes) -> MsPacmanEnvConfig:
    base = replace(
        MsPacmanEnvConfig(),
        num_envs=1,
        frame_skip=1,
        noop_max=0,
        repeat_action_probability=0.0,
    )
    return replace(base, **changes)


def test_single_ale_environment_contract_and_grayscale_observation() -> None:
    env = make_env(deterministic_config(include_ram_metrics=True))
    try:
        observation, info = env.reset(seed=11)
        assert observation.shape == (4, 84, 84)
        assert observation.dtype == np.uint8
        assert env.observation_space.contains(observation)
        assert env.action_space.n == 9
        assert env.unwrapped.ale.getInt("max_num_frames_per_episode") == 0
        assert info["dots_eaten"] == 0

        frames = observation.reshape(4, 84, 84)
        for index in range(1, 4):
            np.testing.assert_array_equal(frames[0], frames[index])
        assert np.unique(frames[-1]).size > 2

        next_observation, reward, terminated, truncated, info = env.step(0)
        assert next_observation.shape == observation.shape
        assert reward == info["raw_reward"]
        assert not terminated
        assert not truncated
        assert info["game_over"] is False
    finally:
        env.close()


def test_fixed_seed_reproduces_observations_and_rewards() -> None:
    env = make_env(deterministic_config(noop_max=10))
    actions = [0, 1, 2, 3, 4, 0, 2]
    try:
        first_observation, _ = env.reset(seed=123)
        first_trajectory = []
        for action in actions:
            observation, reward, terminated, truncated, info = env.step(action)
            first_trajectory.append((observation, reward, terminated, truncated, info["lives"]))

        second_observation, _ = env.reset(seed=123)
        np.testing.assert_array_equal(first_observation, second_observation)
        for action, expected in zip(actions, first_trajectory, strict=True):
            actual = env.step(action)
            np.testing.assert_array_equal(expected[0], actual[0])
            assert expected[1:4] == actual[1:4]
            assert expected[4] == actual[4]["lives"]
    finally:
        env.close()


def test_life_losses_do_not_terminate_before_game_over() -> None:
    env = make_env(
        deterministic_config(frame_skip=4, include_ram_metrics=True)
    )
    rng = np.random.default_rng(4)
    raw_score = 0.0
    life_losses = 0
    try:
        env.reset(seed=3)
        for _ in range(5_000):
            action = int(rng.integers(env.action_space.n))
            _, reward, terminated, truncated, info = env.step(action)
            raw_score += info["raw_reward"]
            assert reward == info["raw_reward"]
            assert not truncated
            assert info["raw_score"] == raw_score
            if info["life_lost"]:
                life_losses += 1
                if not info["game_over"]:
                    assert not terminated
            if terminated:
                assert info["game_over"] is True
                assert info["lives"] == 0
                break
        else:
            pytest.fail("Random policy did not reach Game Over within 5,000 steps")
        assert life_losses >= 3
    finally:
        env.close()


def test_spawn_vector_environment_and_masked_reset() -> None:
    config = deterministic_config(num_envs=2, multiprocessing_context="spawn")
    env = make_vector_env(config)
    try:
        observations, infos = env.reset(seed=99)
        assert observations.shape == (2, 4, 84, 84)
        assert observations.dtype == np.uint8
        assert infos["episode_id"].tolist() == [0, 0]

        observations, _, terminated, truncated, _ = env.step(
            np.zeros(2, dtype=np.int64)
        )
        assert not np.any(terminated)
        assert not np.any(truncated)
        held_second_observation = observations[1].copy()

        reset_mask = np.array([True, False], dtype=np.bool_)
        reset_observations, reset_infos = env.reset(
            options={"reset_mask": reset_mask}
        )
        np.testing.assert_array_equal(reset_observations[1], held_second_observation)
        assert reset_infos["_episode_id"].tolist() == [True, False]
        assert reset_infos["episode_id"][0] == 1

        _, _, _, truncated, infos = env.step(np.zeros(2, dtype=np.int64))
        assert not np.any(truncated)
        assert infos["episode_id"].tolist() == [1, 0]
    finally:
        env.close()
    assert env.closed
