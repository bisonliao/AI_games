from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from PacManEnv.wrappers import GrayscaleFrameStack, MsPacmanTaskWrapper


def grayscale_frame(value: int) -> np.ndarray:
    return np.full((2, 2), value, dtype=np.uint8)


class FrameEnv(gym.Env):
    observation_space = Box(0, 255, (2, 2), dtype=np.uint8)
    action_space = Discrete(2)

    def __init__(self) -> None:
        self.value = 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.value = 1
        return grayscale_frame(self.value), {}

    def step(self, action):
        del action
        self.value += 1
        return grayscale_frame(self.value), 0.0, False, False, {}


def test_grayscale_frame_stack_order_reset_padding_and_no_aliasing() -> None:
    env = GrayscaleFrameStack(FrameEnv(), stack_size=4)
    reset_observation, _ = env.reset(seed=3)
    held_observation = reset_observation
    held_values = held_observation.copy()

    reset_frames = reset_observation.reshape(4, 2, 2)
    assert reset_observation.shape == (4, 2, 2)
    assert reset_observation.dtype == np.uint8
    for frame in reset_frames:
        np.testing.assert_array_equal(frame, np.ones((2, 2), dtype=np.uint8))

    next_observation, *_ = env.step(0)
    next_frames = next_observation.reshape(4, 2, 2)
    np.testing.assert_array_equal(
        next_frames[-1], np.full((2, 2), 2, dtype=np.uint8)
    )
    for frame in next_frames[:-1]:
        np.testing.assert_array_equal(frame, np.ones((2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(held_observation, held_values)
    assert not np.shares_memory(held_observation, next_observation)


@dataclass
class StepEvent:
    reward: float
    lives: int
    terminated: bool = False
    truncated: bool = False


class FakeALE:
    def __init__(self) -> None:
        self.current_lives = 3
        self.current_game_over = False
        self.ram = np.zeros(128, dtype=np.uint8)

    def lives(self) -> int:
        return self.current_lives

    def game_over(self, *, with_truncation: bool = True) -> bool:
        del with_truncation
        return self.current_game_over

    def getRAM(self) -> np.ndarray:
        return self.ram.copy()


class TaskEnv(gym.Env):
    observation_space = Box(0, 255, (1,), dtype=np.uint8)
    action_space = Discrete(2)

    def __init__(self, events: list[StepEvent]) -> None:
        self.events = events
        self.index = 0
        self.ale = FakeALE()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.index = 0
        self.ale.current_lives = 3
        self.ale.current_game_over = False
        return np.array([0], dtype=np.uint8), {
            "lives": 3,
            "episode_frame_number": 0,
        }

    def step(self, action):
        del action
        event = self.events[self.index]
        self.index += 1
        self.ale.current_lives = event.lives
        self.ale.current_game_over = event.terminated
        return (
            np.array([self.index], dtype=np.uint8),
            event.reward,
            event.terminated,
            event.truncated,
            {
                "lives": event.lives,
                "episode_frame_number": self.index * 4,
            },
        )


def test_task_wrapper_separates_life_loss_from_game_over() -> None:
    env = MsPacmanTaskWrapper(
        TaskEnv(
            [
                StepEvent(reward=10.0, lives=2),
                StepEvent(reward=20.0, lives=2),
                StepEvent(reward=0.0, lives=0, terminated=True),
            ]
        )
    )
    _, reset_info = env.reset()
    assert reset_info["episode_id"] == 0
    assert reset_info["raw_score"] == 0.0

    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 10.0
    assert info["raw_reward"] == 10.0
    assert info["raw_score"] == 10.0
    assert info["life_lost"] is True
    assert info["game_over"] is False
    assert not terminated and not truncated

    _, _, terminated, _, info = env.step(0)
    assert info["raw_score"] == 30.0
    assert info["life_lost"] is False
    assert not terminated

    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["game_over"] is True
    assert info["life_lost"] is True


def test_seeded_reset_starts_a_reproducible_episode_sequence() -> None:
    env = MsPacmanTaskWrapper(TaskEnv([StepEvent(reward=0.0, lives=3)]))
    _, first_info = env.reset(seed=7)
    _, next_info = env.reset()
    _, reseeded_info = env.reset(seed=7)

    assert first_info["episode_id"] == 0
    assert next_info["episode_id"] == 1
    assert reseeded_info["episode_id"] == 0


def test_task_wrapper_shapes_then_clips_reward_without_changing_raw_score() -> None:
    env = MsPacmanTaskWrapper(
        TaskEnv([StepEvent(10.0, 3), StepEvent(0.0, 3)]),
        step_cost=0.25,
        clip_training_reward=True,
    )
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert reward == 1.0
    assert info["raw_reward"] == 10.0
    assert info["raw_score"] == 10.0

    _, reward, _, _, info = env.step(0)
    assert reward == -0.25
    assert info["raw_reward"] == 0.0
    assert info["raw_score"] == 10.0


def test_task_wrapper_rejects_any_truncation() -> None:
    env = MsPacmanTaskWrapper(TaskEnv([StepEvent(0.0, 3, truncated=True)]))
    env.reset()
    with pytest.raises(RuntimeError, match="truncated"):
        env.step(0)


def test_optional_ram_metric() -> None:
    base_env = TaskEnv([StepEvent(0.0, 3)])
    base_env.ale.ram[119] = 17
    env = MsPacmanTaskWrapper(base_env, include_ram_metrics=True)
    _, reset_info = env.reset()
    assert reset_info["dots_eaten"] == 17
