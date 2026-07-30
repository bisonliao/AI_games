from __future__ import annotations

import numpy as np

from env import JumpEnvConfig, make_async_vector_env
from env.vector_env import final_info_at


def test_same_step_vector_env_returns_reset_observation_and_final_info() -> None:
    config = JumpEnvConfig(observation_mode="vector")
    envs = make_async_vector_env(2, config, context="spawn")
    try:
        observations, _ = envs.reset(seed=900)
        actions = np.stack(
            [
                config.oracle_action(float(value * config.max_distance))
                for value in observations[:, 0]
            ]
        )
        next_observations, _, terminated, truncated, infos = envs.step(actions)
        assert next_observations.shape == (2, 1)
        assert np.all(terminated)
        assert not np.any(truncated)
        assert "final_obs" in infos
        for index in range(2):
            final_info = final_info_at(infos, index)
            assert bool(final_info["is_success"])
            assert final_info["landing_platform"] == "B"
    finally:
        envs.close(terminate=True)


def test_vector_worker_seeds_are_distinct() -> None:
    config = JumpEnvConfig(observation_mode="vector")
    envs = make_async_vector_env(3, config, context="spawn")
    try:
        observations, _ = envs.reset(seed=42)
        assert len(np.unique(observations[:, 0])) == 3
    finally:
        envs.close(terminate=True)
