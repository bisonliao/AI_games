"""Observation and task-semantics wrappers for Ms. Pac-Man."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class GrayscaleFrameStack(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Stack grayscale frames from oldest to newest as (T, H, W)."""

    def __init__(self, env: gym.Env, stack_size: int = 4):
        gym.utils.RecordConstructorArgs.__init__(self, stack_size=stack_size)
        gym.Wrapper.__init__(self, env)
        if stack_size <= 0:
            raise ValueError("stack_size must be positive")
        if not isinstance(env.observation_space, Box):
            raise TypeError("GrayscaleFrameStack requires a Box observation space")

        input_space = env.observation_space
        if len(input_space.shape) != 2:
            raise ValueError("GrayscaleFrameStack expects HW grayscale observations")
        if input_space.dtype != np.uint8:
            raise ValueError("GrayscaleFrameStack expects uint8 observations")

        height, width = input_space.shape
        self.stack_size = stack_size
        self._frames = np.empty((stack_size, height, width), dtype=np.uint8)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(stack_size, height, width),
            dtype=np.uint8,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        frame = self._validate_frame(observation)
        self._frames[:] = frame
        return self._observation(), info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = self._validate_frame(observation)
        return self._observation(), reward, terminated, truncated, info

    def _validate_frame(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation)
        expected_shape = self._frames.shape[-2:]
        if observation.shape != expected_shape or observation.dtype != np.uint8:
            raise RuntimeError(
                f"Expected uint8 grayscale observation {expected_shape}, got "
                f"{observation.shape} {observation.dtype}"
            )
        return observation

    def _observation(self) -> np.ndarray:
        # The copy prevents subsequent steps from mutating an observation held by a caller.
        return self._frames.copy()


class MsPacmanTaskWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Enforce full-game episodes while preserving raw score and life events."""

    def __init__(
        self,
        env: gym.Env,
        *,
        step_cost: float = 0.0,
        clip_training_reward: bool = False,
        include_ram_metrics: bool = False,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            step_cost=step_cost,
            clip_training_reward=clip_training_reward,
            include_ram_metrics=include_ram_metrics,
        )
        gym.Wrapper.__init__(self, env)
        if not np.isfinite(step_cost) or step_cost < 0.0:
            raise ValueError("step_cost must be finite and non-negative")
        self.step_cost = float(step_cost)
        self.clip_training_reward = clip_training_reward
        self.include_ram_metrics = include_ram_metrics
        self._raw_score = 0.0
        self._previous_lives = 0
        self._episode_id = -1

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, original_info = self.env.reset(seed=seed, options=options)
        info = dict(original_info)
        if seed is not None:
            # A seeded reset starts a new reproducible run. Subsequent unseeded
            # resets number the full games within that run.
            self._episode_id = 0
        else:
            self._episode_id += 1
        self._raw_score = 0.0
        self._previous_lives = self._read_lives(info)
        self._add_task_info(
            info,
            raw_reward=0.0,
            lives=self._previous_lives,
            life_lost=False,
            game_over=False,
        )
        return observation, info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, raw_reward, terminated, truncated, original_info = self.env.step(
            action
        )
        terminated = bool(terminated)
        truncated = bool(truncated)

        if truncated:
            raise RuntimeError(
                "Ms. Pac-Man was truncated; all time limits must remain disabled"
            )

        game_over = bool(self.unwrapped.ale.game_over(with_truncation=False))
        if terminated != game_over:
            raise RuntimeError(
                "Invalid episode boundary: terminated must exactly match ALE game_over"
            )

        info = dict(original_info)
        lives = self._read_lives(info)
        life_lost = lives < self._previous_lives
        raw_reward_value = float(raw_reward)
        self._raw_score += raw_reward_value
        self._add_task_info(
            info,
            raw_reward=raw_reward_value,
            lives=lives,
            life_lost=life_lost,
            game_over=game_over,
        )
        self._previous_lives = lives

        training_reward = raw_reward_value - self.step_cost
        if self.clip_training_reward:
            training_reward = float(np.clip(training_reward, -1.0, 1.0))

        return observation, training_reward, terminated, False, info

    def _read_lives(self, info: dict[str, Any]) -> int:
        if "lives" in info:
            return int(info["lives"])
        return int(self.unwrapped.ale.lives())

    def _add_task_info(
        self,
        info: dict[str, Any],
        *,
        raw_reward: float,
        lives: int,
        life_lost: bool,
        game_over: bool,
    ) -> None:
        info.update(
            raw_reward=raw_reward,
            raw_score=self._raw_score,
            lives=lives,
            life_lost=life_lost,
            game_over=game_over,
            emulator_frames=int(info.get("episode_frame_number", 0)),
            episode_id=self._episode_id,
        )
        if self.include_ram_metrics:
            info["dots_eaten"] = int(self.unwrapped.ale.getRAM()[119])
