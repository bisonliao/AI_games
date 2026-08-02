"""Gymnasium, deterministic physics, steering, and radial-goal tests for env2."""

from dataclasses import replace

import numpy as np
from gymnasium.utils.env_checker import check_env

from env import WindConfig
from env2 import BicycleSteeringEnv, BicycleSteeringEnvConfig
from env2.baseline import pd_action


NO_WIND = WindConfig(enabled=False)


def make_env(**changes):
    """Build a fast deterministic steering environment for one unit test."""
    config = replace(BicycleSteeringEnvConfig(wind=NO_WIND), **changes)
    return BicycleSteeringEnv(config)


def test_steering_environment_gymnasium_contract():
    env = make_env()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_seed_reproduces_steering_trajectory():
    trajectories = []
    for _ in range(2):
        env = BicycleSteeringEnv()
        try:
            observation, _ = env.reset(seed=123)
            values = [observation]
            for action in (1, 0, 2, 2, 0):
                observation, reward, terminated, truncated, _ = env.step(action)
                values.append(np.append(observation, [reward, terminated, truncated]))
            trajectories.append(values)
        finally:
            env.close()
    for left, right in zip(*trajectories, strict=True):
        np.testing.assert_allclose(left, right, atol=1e-7)


def test_left_and_right_actions_move_handlebar_in_opposite_directions():
    angles = []
    for action in (1, 2):
        env = make_env()
        try:
            env.reset(
                seed=0,
                options={"initial_roll_rad": 0.0, "initial_roll_rate_rad_s": 0.0},
            )
            _, _, _, _, info = env.step(action)
            angles.append(info["steering_angle_rad"])
        finally:
            env.close()
    assert angles[0] > 0 > angles[1]


def test_scripted_steering_reaches_radial_goal():
    env = make_env(goal_distance_m=5.0, max_episode_seconds=8.0)
    try:
        observation, _ = env.reset(seed=0)
        while True:
            observation, _, terminated, truncated, info = env.step(
                pd_action(observation)
            )
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info["outcome"] == "success"
        assert info["success_reason"] == "distance"
        assert info["distance_from_start_m"] >= 5.0
    finally:
        env.close()


def test_surviving_time_limit_is_success_not_truncation():
    env = make_env(goal_distance_m=100.0, max_episode_seconds=0.05)
    try:
        env.reset(
            seed=0,
            options={"initial_roll_rad": 0.0, "initial_roll_rate_rad_s": 0.0},
        )
        _, reward, terminated, truncated, info = env.step(0)
        assert terminated and not truncated
        assert info["success"]
        assert info["outcome"] == "success"
        assert info["success_reason"] == "survival"
        assert reward >= env.config.success_reward
    finally:
        env.close()
