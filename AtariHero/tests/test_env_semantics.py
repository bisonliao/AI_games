from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from HeroEnv.hero_env import CurriculumCheckpoint, HeroLevelRangeEnv
from curri_DQN.config import TrainConfig
from curri_DQN.actor import compute_training_reward
from curri_DQN.envs import DQNAtariWrapper
from curri_DQN.evaluator import build_eval_checkpoint_plan


class _BudgetEnv(gym.Env):
    observation_space = gym.spaces.Box(
        low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
    )
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {
            "hero_budget_decisions": 3
        }

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, dtype=np.uint8),
            0.0,
            False,
            False,
            {"is_success": False},
        )


class _FakeALE:
    def __init__(self) -> None:
        self.ram = np.zeros(128, dtype=np.uint8)
        self.ram[117] = 1  # Level 2.
        self.ram[28] = 0
        self._lives = 3

    def getRAM(self):
        return self.ram

    def lives(self):
        return self._lives


class _LevelAdvanceEnv(gym.Env):
    observation_space = gym.spaces.Box(
        low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
    )
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        super().__init__()
        self.ale = _FakeALE()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ale.ram[117] = 1
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self.ale.ram[117] = 2  # Level 3: miner in reset Level 2 was rescued.
        return np.ones(self.observation_space.shape, dtype=np.uint8), 1_000.0, False, False, {}


class _LifeLossEnv(_LevelAdvanceEnv):
    def step(self, action):
        self.ale._lives -= 1
        return np.ones(self.observation_space.shape, dtype=np.uint8), 0.0, False, False, {}


class _DelayedLevelAdvanceEnv(_LevelAdvanceEnv):
    def __init__(self) -> None:
        super().__init__()
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.steps = 0
        return observation, info

    def step(self, action):
        self.steps += 1
        if self.steps == 1:
            return (
                np.ones(self.observation_space.shape, dtype=np.uint8),
                1_000.0,
                False,
                False,
                {},
            )
        self.ale.ram[117] = 2
        return (
            np.ones(self.observation_space.shape, dtype=np.uint8),
            0.0,
            False,
            False,
            {},
        )


class _RewardSequenceEnv(gym.Env):
    observation_space = gym.spaces.Box(
        low=0, high=255, shape=(210, 160, 3), dtype=np.uint8
    )
    action_space = gym.spaces.Discrete(2)

    def __init__(self, rewards: list[float]) -> None:
        super().__init__()
        self.rewards = rewards
        self.index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        reward = self.rewards[self.index]
        self.index += 1
        return (
            np.zeros(self.observation_space.shape, dtype=np.uint8),
            reward,
            False,
            False,
            {"hero_ale_reward": reward},
        )


def test_curriculum_timeout_is_terminal_failure() -> None:
    env = DQNAtariWrapper(
        _BudgetEnv(), action_repeat=1, episode_timeout_decisions=3
    )
    env.reset(seed=0)
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["hero_time_limit_reached"]
    assert info["hero_episode_timeout_decisions"] == 3
    assert not info["is_success"]
    assert info["hero_terminal_reason"] == "timeout"


def test_action_repeat_aggregates_documented_event_rewards() -> None:
    env = DQNAtariWrapper(
        _RewardSequenceEnv([75.0, 50.0, 1000.0, 50.0]),
        action_repeat=4,
        episode_timeout_decisions=99,
    )
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(0)
    assert not terminated and not truncated
    assert reward == 100.998
    assert info["hero_walls_destroyed"] == 1
    assert info["hero_creatures_killed"] == 1
    assert info["hero_miner_rescued_events"] == 1
    assert info["hero_dynamite_bonus_sticks"] == 1
    assert info["hero_ale_reward"] == 1175.0
    assert info["hero_rl_reward"] == reward


def test_miner_reward_is_not_paid_again_on_delayed_level_transition() -> None:
    base = HeroLevelRangeEnv(_DelayedLevelAdvanceEnv(), max_level=2)
    base.reset(seed=0)
    base._reset_checkpoint = CurriculumCheckpoint(
        checkpoint_id="cp",
        task_id="task",
        path=Path("state.chkpt"),
        screenshot_path=Path("state.png"),
        stage=4,
        global_depth=0,
        level=2,
        room=1,
        local_band=0,
        lives=3,
        power_ratio=1.0,
        budget_decisions=500,
    )
    base._curriculum_identity = {
        "format_version": 2,
        "version": 1,
        "manifest_sha256": "test",
    }
    base._reset_level = 2
    env = DQNAtariWrapper(base, action_repeat=1, episode_timeout_decisions=10)
    env.reset(seed=0)
    base._reset_checkpoint = CurriculumCheckpoint(
        checkpoint_id="cp",
        task_id="task",
        path=Path("state.chkpt"),
        screenshot_path=Path("state.png"),
        stage=4,
        global_depth=0,
        level=2,
        room=1,
        local_band=0,
        lives=3,
        power_ratio=1.0,
        budget_decisions=500,
    )
    base._curriculum_identity = {
        "format_version": 2,
        "version": 1,
        "manifest_sha256": "test",
    }
    base._reset_level = 2

    _, first_reward, terminated, truncated, first_info = env.step(0)
    assert not terminated and not truncated
    assert first_reward == 99.998
    assert first_info["hero_miner_rescued_events"] == 1

    _, terminal_reward, terminated, truncated, terminal_info = env.step(0)
    assert terminated and not truncated
    assert terminal_info["is_success"]
    assert terminal_info["hero_miner_rescued_events"] == 0
    assert terminal_reward == -0.002
    assert np.isclose(first_reward + terminal_reward, 99.996)


def test_level_advance_is_miner_rescue_success() -> None:
    env = HeroLevelRangeEnv(_LevelAdvanceEnv(), max_level=2)
    observation, _ = env.reset(seed=0)
    env._reset_checkpoint = CurriculumCheckpoint(
        checkpoint_id="cp",
        task_id="task",
        path=Path("state.chkpt"),
        screenshot_path=Path("state.png"),
        stage=1,
        global_depth=1,
        level=2,
        room=1,
        local_band=1,
        lives=3,
        power_ratio=1.0,
        budget_decisions=100,
    )
    env._curriculum_identity = {
        "format_version": 2,
        "version": 1,
        "manifest_sha256": "test",
    }
    env._reset_level = 2
    next_observation, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["hero_miner_rescued"] and info["is_success"]
    assert reward == 100.0
    assert info["hero_ale_reward"] == 1000.0
    assert info["hero_terminal_reason"] == "miner-rescued"
    assert info["hero_next_level"] == 3
    assert np.array_equal(next_observation, observation)


def test_full_level_1_to_2_environment_stops_before_level_3() -> None:
    env = HeroLevelRangeEnv(_LevelAdvanceEnv())
    assert env.max_level == 2
    observation, _ = env.reset(seed=0)
    next_observation, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["hero_level_cap_reached"]
    assert info["hero_next_level"] == 3
    assert info["is_success"]
    assert reward == 100.0
    assert info["hero_terminal_reason"] == "miner-rescued"
    assert np.array_equal(next_observation, observation)


def test_curriculum_attempt_stops_on_first_life_loss() -> None:
    env = HeroLevelRangeEnv(_LifeLossEnv(), max_level=2)
    env.reset(seed=0)
    env._reset_checkpoint = CurriculumCheckpoint(
        checkpoint_id="cp",
        task_id="task",
        path=Path("state.chkpt"),
        screenshot_path=Path("state.png"),
        stage=1,
        global_depth=1,
        level=2,
        room=1,
        local_band=1,
        lives=3,
        power_ratio=1.0,
        budget_decisions=100,
    )
    env._curriculum_identity = {
        "format_version": 2,
        "version": 1,
        "manifest_sha256": "test",
    }
    env._reset_level = 2
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["hero_life_lost"]
    assert not info["is_success"]
    assert reward == -1.0
    assert info["hero_terminal_reason"] == "life-lost"


def test_full_game_attempt_stops_on_first_life_loss() -> None:
    env = HeroLevelRangeEnv(_LifeLossEnv(), max_level=2)
    env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["hero_life_lost"]
    assert not info["is_success"]
    assert reward == -1.0
    assert info["hero_terminal_reason"] == "life-lost"


class _CheckpointCatalog:
    def checkpoint_ids_for_stage(self, stage: int) -> tuple[str, ...]:
        return {1: ("a", "b"), 2: ("c",)}.get(stage, ())

    def checkpoint_ids_for_level_start(self, level: int) -> tuple[str, ...]:
        return {1: ("level1-start",), 2: ("level2-start",)}.get(level, ())


def test_evaluation_matrix_is_balanced_and_step_independent() -> None:
    config = SimpleNamespace(
        target_stage=2,
        eval_episodes=6,
        eval_current_stage_fraction=0.5,
        seed=7,
    )
    plan = build_eval_checkpoint_plan(config, _CheckpointCatalog())
    assert plan == build_eval_checkpoint_plan(config, _CheckpointCatalog())
    current = [identifier for stage, identifier, _ in plan if stage == 2]
    earlier = [identifier for stage, identifier, _ in plan if stage == 1]
    assert current == ["c", "c", "c"]
    assert earlier == ["a", "b", "a"]


def test_after_curriculum_uses_full_game_evaluation_matrix() -> None:
    config = SimpleNamespace(
        after_curri=True,
        target_stage=4,
        eval_episodes=3,
        seed=7,
    )
    plan = build_eval_checkpoint_plan(config, _CheckpointCatalog())
    assert len(plan) == 3
    assert sum(checkpoint == "level1-start" for _, checkpoint, _ in plan) == 1
    assert sum(checkpoint == "level2-start" for _, checkpoint, _ in plan) == 2
    assert all(stage == 0 for stage, _, _ in plan)


def test_after_curriculum_config_requires_weights_and_uses_distinct_run_path() -> None:
    config = TrainConfig(
        target_stage=1,
        after_curri=True,
        load_checkpoint="weights.pt",
    )
    config.validate()
    assert config.run_path.name.endswith("_afterCurri")


def test_training_reward_uses_documented_event_values() -> None:
    config = TrainConfig(decision_step_penalty=0.01)
    assert compute_training_reward(config, 0.0) == -0.01
    assert (
        compute_training_reward(config, 0.0, walls_destroyed=1)
        == 0.49
    )
    assert (
        compute_training_reward(config, 0.0, creatures_killed=1)
        == 0.49
    )
    assert (
        compute_training_reward(config, 0.0, miner_rescued_events=1)
        == 99.99
    )


def test_training_reward_penalizes_timeout_after_step_cost() -> None:
    config = TrainConfig(
        decision_step_penalty=0.01,
        timeout_terminal_reward=-1.0,
        life_lost_terminal_reward=-1.0,
    )
    assert (
        compute_training_reward(config, 1000.0, time_limit_reached=True)
        == -1.01
    )
    assert (
        compute_training_reward(config, 0.0, life_lost=True)
        == -1.01
    )
    assert compute_training_reward(config, 0.0, miner_rescued_events=1) == 99.99
