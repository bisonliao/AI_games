"""Gymnasium contract, deterministic physics, reward, and task tests."""

from dataclasses import replace

import numpy as np
from gymnasium.utils.env_checker import check_env

from env import BicycleBalanceEnv, BicycleEnvConfig, WindConfig


NO_WIND = WindConfig(enabled=False)


def make_env(**changes):
    return BicycleBalanceEnv(replace(BicycleEnvConfig(wind=NO_WIND), **changes))


def test_gymnasium_contract():
    env = make_env()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_seed_reproduces_trajectory():
    trajectories = []
    for _ in range(2):
        env = BicycleBalanceEnv()
        try:
            observation, _ = env.reset(seed=123)
            values = [observation]
            for action in (2, 0, 1, 2, 1):
                observation, reward, terminated, truncated, _ = env.step(action)
                values.append(np.append(observation, [reward, terminated, truncated]))
            trajectories.append(values)
        finally:
            env.close()
    for left, right in zip(*trajectories, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-7)


def test_reaction_wheel_pushes_frame_in_opposite_direction():
    rolls = []
    wheel_speeds = []
    for action in (0, 2):
        env = make_env()
        try:
            env.reset(
                seed=0,
                options={"initial_roll_rad": 0.0, "initial_roll_rate_rad_s": 0.0},
            )
            _, _, _, _, info = env.step(action)
            rolls.append(info["roll_rad"])
            wheel_speeds.append(info["reaction_wheel_speed_rad_s"])
        finally:
            env.close()
    assert rolls[0] > 0 > rolls[1]
    assert wheel_speeds[0] < 0 < wheel_speeds[1]


def test_drive_holds_target_speed_while_upright():
    env = make_env()
    try:
        observation, _ = env.reset(
            seed=0,
            options={"initial_roll_rad": 0.0, "initial_roll_rate_rad_s": 0.0},
        )
        for _ in range(10):
            observation, _, terminated, _, info = env.step(1)
            assert not terminated
        assert 1.9 <= info["forward_speed_mps"] <= 2.1
    finally:
        env.close()


def test_timeout_is_truncation_and_bootstrap_state_is_valid():
    env = make_env(max_episode_seconds=0.05)
    try:
        observation, _ = env.reset(
            seed=0,
            options={"initial_roll_rad": 0.0, "initial_roll_rate_rad_s": 0.0},
        )
        observation, _, terminated, truncated, info = env.step(1)
        assert not terminated and truncated
        assert info["outcome"] == "timeout"
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_success_reward_adds_progress_and_terminal_bonus():
    env = make_env(goal_distance_m=0.4, max_episode_seconds=2.0)
    total_reward = 0.0
    try:
        env.reset(
            seed=0,
            options={"initial_roll_rad": 0.0, "initial_roll_rate_rad_s": 0.0},
        )
        while True:
            _, reward, terminated, truncated, info = env.step(1)
            total_reward += reward
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info["outcome"] == "success"
        assert abs(total_reward - 1.004) < 1e-6
    finally:
        env.close()
