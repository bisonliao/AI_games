from dataclasses import replace

import pytest

from DQN.config import DQNConfig
from DQN.utils import actor_environment_seed, actor_policy_seed, linear_beta, linear_epsilon


def test_all_actors_share_the_same_global_epsilon_schedule() -> None:
    config = DQNConfig()
    assert linear_epsilon(0, config) == pytest.approx(0.9)
    assert linear_epsilon(500_000, config) == pytest.approx(0.475)
    assert linear_epsilon(1_000_000, config) == pytest.approx(0.05)
    assert linear_epsilon(5_000_000, config) == pytest.approx(0.05)


def test_replay_beta_schedule() -> None:
    config = DQNConfig()
    assert linear_beta(0, config) == pytest.approx(0.4)
    assert linear_beta(5_000_000, config) == pytest.approx(0.7)
    assert linear_beta(10_000_000, config) == pytest.approx(1.0)


def test_actor_seed_ranges_and_policy_rngs_are_disjoint() -> None:
    config = replace(DQNConfig(), num_actors=3)
    environment_seeds = [actor_environment_seed(config, i) for i in range(3)]
    policy_seeds = [actor_policy_seed(config, i) for i in range(3)]
    assert len(set(environment_seeds + policy_seeds)) == 6
    for actor_id in range(config.num_actors - 1):
        current_last_env_seed = (
            environment_seeds[actor_id] + config.actor_env.num_envs - 1
        )
        assert current_last_env_seed < environment_seeds[actor_id + 1]


def test_actor_chunk_must_align_with_vector_steps() -> None:
    with pytest.raises(ValueError, match="divisible"):
        replace(DQNConfig(), actor_transition_batch_size=65)


def test_default_output_directories_are_at_project_root() -> None:
    config = DQNConfig()
    project_root = config.runs_dir.parent
    assert project_root.name == "AtariMsPacMan"
    assert config.runs_dir == project_root / "runs"
    assert config.checkpoints_dir == project_root / "chkpt"
