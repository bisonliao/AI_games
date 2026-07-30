from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from env import JumpEnv, JumpEnvConfig


def test_gymnasium_contract_and_terminal_transition() -> None:
    env = JumpEnv()
    try:
        check_env(env, skip_render_check=True)
        observation, info = env.reset(seed=7)
        assert observation.shape == (2, 64, 64)
        assert observation.dtype == np.uint8
        action = env.config.oracle_action(info["target_distance"])
        _, reward, terminated, truncated, final_info = env.step(action)
        assert terminated and not truncated
        assert reward > 1.0
        assert final_info["is_success"] is True
        assert {
            "hold_time_s",
            "target_distance",
            "landing_error",
            "landing_platform",
            "simulation_steps",
        } <= final_info.keys()
        with pytest.raises(RuntimeError, match="call reset"):
            env.step(action)
    finally:
        env.close()


def test_reset_seed_is_reproducible() -> None:
    env = JumpEnv()
    try:
        first_observation, first_info = env.reset(seed=1234)
        second_observation, second_info = env.reset(seed=1234)
        np.testing.assert_array_equal(first_observation, second_observation)
        np.testing.assert_array_equal(
            first_info["platform_a_xy"], second_info["platform_a_xy"]
        )
        np.testing.assert_array_equal(
            first_info["platform_b_xy"], second_info["platform_b_xy"]
        )
    finally:
        env.close()


def test_rgb_observation_is_two_binary_platform_channels() -> None:
    env = JumpEnv(JumpEnvConfig(observation_width=48, observation_height=32))
    try:
        observation, _ = env.reset(seed=9)
        assert observation.shape == (2, 32, 48)
        assert set(np.unique(observation)) <= {0, 1}
        assert observation[0].sum() > 0
        assert observation[1].sum() > 0
        assert not np.any(observation[0] & observation[1])
    finally:
        env.close()


def test_vector_observation_remains_available() -> None:
    env = JumpEnv(JumpEnvConfig(observation_mode="vector"))
    try:
        observation, info = env.reset(seed=3)
        assert observation.shape == (1,)
        assert observation.dtype == np.float32
        assert observation.item() == pytest.approx(
            info["target_distance"] / env.config.max_distance
        )
    finally:
        env.close()


def test_touchdown_distance_increases_with_action() -> None:
    env = JumpEnv()
    projections = []
    try:
        for action_value in (-0.8, -0.4, 0.0):
            _, info = env.reset(seed=1)
            direction = info["platform_b_xy"] - info["platform_a_xy"]
            direction = direction / np.linalg.norm(direction)
            _, _, _, _, final_info = env.step(
                np.asarray([action_value], dtype=np.float32)
            )
            travelled = final_info["landing_xy"] - info["platform_a_xy"]
            projections.append(float(np.dot(travelled, direction)))
        assert projections[0] < projections[1] < projections[2]
    finally:
        env.close()


def test_clear_under_and_overshoot_fail() -> None:
    env = JumpEnv()
    try:
        env.reset(seed=0)
        *_, under_info = env.step(np.asarray([-1.0], dtype=np.float32))
        assert under_info["is_success"] is False

        env.reset(seed=0)
        *_, over_info = env.step(np.asarray([1.0], dtype=np.float32))
        assert over_info["is_success"] is False
    finally:
        env.close()


def test_rgb_render() -> None:
    config = JumpEnvConfig(rgb_width=64, rgb_height=48)
    env = JumpEnv(config=config, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        image = env.render()
        assert image.shape == (48, 64, 3)
        assert image.dtype == np.uint8
    finally:
        env.close()


def test_hold_duration_maps_to_normalized_action() -> None:
    config = JumpEnvConfig(max_hold_seconds=1.0)
    assert config.action_from_hold_time(0.0).item() == pytest.approx(-1.0)
    assert config.action_from_hold_time(0.5).item() == pytest.approx(0.0)
    assert config.action_from_hold_time(1.0).item() == pytest.approx(1.0)
    assert config.action_from_hold_time(5.0).item() == pytest.approx(1.0)


def test_cubic_charge_curve_and_analytic_inverse() -> None:
    config = JumpEnvConfig(max_hold_seconds=1.0, charge_exponent=3.0)
    assert config.maximum_jump_distance == pytest.approx(
        config.charge_scale * config.flight_time
    )
    assert config.horizontal_speed_from_hold_time(0.0) == pytest.approx(0.0)
    assert config.horizontal_speed_from_hold_time(0.5) == pytest.approx(0.625)
    assert config.horizontal_speed_from_hold_time(1.0) == pytest.approx(5.0)

    for distance in (1.0, 2.0, 3.0, 4.0):
        action = float(config.oracle_action(distance).item())
        hold_time = (action + 1.0) * 0.5 * config.max_hold_seconds
        ideal_distance = (
            config.horizontal_speed_from_hold_time(hold_time)
            * config.flight_time
        )
        assert ideal_distance == pytest.approx(distance, abs=1e-6)


def test_linear_charge_curve_remains_available_for_old_checkpoints() -> None:
    config = JumpEnvConfig(charge_exponent=1.0)
    assert config.horizontal_speed_from_hold_time(0.5) == pytest.approx(2.5)
    expected_hold = 2.0 / (config.charge_scale * config.flight_time)
    action = float(config.oracle_action(2.0).item())
    actual_hold = (action + 1.0) * 0.5 * config.max_hold_seconds
    assert actual_hold == pytest.approx(expected_hold)


def test_playback_speed_must_be_positive() -> None:
    with pytest.raises(ValueError, match="playback_speed"):
        JumpEnv(playback_speed=0.0)
