"""Atari preprocessing without OpenCV."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

from HeroEnv import make_hero_level_1_to_2_env

from .config import TrainConfig
from .reward import HeroRewardEventParser, HeroRewardEvents


DEFAULT_EPISODE_TIMEOUT_DECISIONS = 500


def sample_reset_stage(
    rng: np.random.Generator,
    target_stage: int,
    current_stage_fraction: float,
) -> int:
    if target_stage == 1 or rng.random() < current_stage_fraction:
        return target_stage
    return int(rng.integers(1, target_stage))


def preprocess_frame(frame: np.ndarray, size: int = 84) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGB").convert("L")
    image = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


class DQNAtariWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        action_repeat: int = 4,
        episode_timeout_decisions: int = DEFAULT_EPISODE_TIMEOUT_DECISIONS,
        wall_event_reward: float = 0.5,
        creature_event_reward: float = 0.5,
        miner_event_reward: float = 100.0,
        decision_step_penalty: float = 0.002,
        terminal_failure_reward: float = -1.0,
        life_lost_terminal_reward: float = -1.0,
        frame_stack: int = 4,
        screen_size: int = 84,
    ) -> None:
        super().__init__(env)
        self.action_repeat = action_repeat
        if episode_timeout_decisions < 1:
            raise ValueError("episode_timeout_decisions must be positive")
        self.episode_timeout_decisions = episode_timeout_decisions
        self.wall_event_reward = wall_event_reward
        self.creature_event_reward = creature_event_reward
        self.miner_event_reward = miner_event_reward
        self.decision_step_penalty = decision_step_penalty
        self.terminal_failure_reward = terminal_failure_reward
        self.life_lost_terminal_reward = life_lost_terminal_reward
        self.reward_events = HeroRewardEventParser()
        self.frame_stack = frame_stack
        self.screen_size = screen_size
        self.frames: deque[np.ndarray] = deque(maxlen=frame_stack)
        self.episode_decisions = 0
        self.reward_events.reset()
        self.episode_budget: int | None = None
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, screen_size, screen_size),
            dtype=np.uint8,
        )

    def set_curriculum_stage(self, stage: int) -> None:
        self.env.set_curriculum_stage(stage)

    def checkpoint_ids_for_stage(self, stage: int) -> tuple[str, ...]:
        return self.env.checkpoint_ids_for_stage(stage)

    def checkpoint_ids_for_level_start(self, level: int) -> tuple[str, ...]:
        return self.env.checkpoint_ids_for_level_start(level)

    @property
    def curriculum_identity(self) -> dict[str, Any] | None:
        return self.env.curriculum_identity

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self.episode_decisions = 0
        self.reward_events.reset()
        # Use one uniform runtime cap for every training/evaluation episode.
        # The manifest demo budget remains metadata, not an episode-length
        # policy.
        self.episode_budget = self.episode_timeout_decisions
        info["hero_budget_decisions"] = self.episode_budget
        info["hero_episode_timeout_decisions"] = self.episode_budget
        frame = preprocess_frame(observation, self.screen_size)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return np.stack(self.frames), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        total_ale_reward = 0.0
        events = HeroRewardEvents()
        recent_frames: list[np.ndarray] = []
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        for repeat_index in range(self.action_repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            ale_reward = float(info.get("hero_ale_reward", reward))
            total_ale_reward += ale_reward
            events += self.reward_events.observe(ale_reward)
            if repeat_index >= self.action_repeat - 2:
                recent_frames.append(observation)
            if terminated or truncated:
                recent_frames.append(observation)
                break
        pooled = (
            np.maximum(recent_frames[-2], recent_frames[-1])
            if len(recent_frames) >= 2
            else recent_frames[-1]
        )
        self.frames.append(preprocess_frame(pooled, self.screen_size))
        self.episode_decisions += 1
        if info.get("hero_miner_rescued", False) and events.miner_rescued == 0:
            events += self.reward_events.mark_miner_rescued()
        if terminated and info.get("hero_miner_rescued", False):
            info["hero_terminal_reason"] = "miner-rescued"
        life_lost = bool(info.get("hero_life_lost", False))
        if (
            not terminated
            and not truncated
            and self.episode_budget is not None
            and self.episode_decisions >= self.episode_budget
        ):
            # This is an explicit task failure, not a continuing-state
            # Gymnasium time-limit truncation.
            terminated = True
            info["hero_time_limit_reached"] = True
            info["hero_miner_rescued"] = False
            info["is_success"] = False
            info["hero_terminal_reason"] = "timeout"
        if life_lost:
            rl_reward = (
                self.life_lost_terminal_reward - self.decision_step_penalty
            )
        elif info.get("hero_time_limit_reached", False):
            rl_reward = self.terminal_failure_reward - self.decision_step_penalty
        else:
            rl_reward = (
                self.wall_event_reward * events.walls_destroyed
                + self.creature_event_reward * events.creatures_killed
                + self.miner_event_reward * events.miner_rescued
                - self.decision_step_penalty
            )
        info["hero_episode_decisions"] = self.episode_decisions
        info["hero_episode_timeout_decisions"] = self.episode_budget
        info["hero_walls_destroyed"] = events.walls_destroyed
        info["hero_creatures_killed"] = events.creatures_killed
        info["hero_miner_rescued_events"] = events.miner_rescued
        info["hero_dynamite_bonus_sticks"] = events.dynamite_bonus_sticks
        info["hero_unmapped_ale_reward"] = events.unmapped_reward
        info["hero_ale_reward"] = total_ale_reward
        info["hero_rl_reward"] = rl_reward
        return np.stack(self.frames), rl_reward, terminated, truncated, info


def make_training_env(
    config: TrainConfig,
    stage: int,
) -> DQNAtariWrapper:
    if config.after_curri:
        base = make_hero_level_1_to_2_env(
            training=True,
            checkpoint_dir=Path(config.hero_checkpoint_dir),
            curriculum_stage=1,
            checkpoint_reset_probability=1.0,
            include_easier_stages=False,
            frameskip=1,
            repeat_action_probability=config.sticky_action_probability,
        )
    else:
        base = make_hero_level_1_to_2_env(
            training=True,
            checkpoint_dir=Path(config.hero_checkpoint_dir),
            curriculum_stage=stage,
            checkpoint_reset_probability=1.0,
            include_easier_stages=False,
            frameskip=1,
            repeat_action_probability=config.sticky_action_probability,
        )
    return DQNAtariWrapper(
        base,
        action_repeat=config.action_repeat,
        episode_timeout_decisions=config.episode_timeout_decisions,
        wall_event_reward=config.wall_event_reward,
        creature_event_reward=config.creature_event_reward,
        miner_event_reward=config.miner_event_reward,
        decision_step_penalty=config.decision_step_penalty,
        terminal_failure_reward=config.timeout_terminal_reward,
        life_lost_terminal_reward=config.life_lost_terminal_reward,
        frame_stack=config.frame_stack,
        screen_size=config.screen_size,
    )
