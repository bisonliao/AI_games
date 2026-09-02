"""Factories for single and process-parallel Ms. Pac-Man environments."""

from __future__ import annotations

from functools import partial
from typing import Literal

import ale_py
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from gymnasium.wrappers import AtariPreprocessing

from PacManEnv.config import MsPacmanEnvConfig
from PacManEnv.wrappers import GrayscaleFrameStack, MsPacmanTaskWrapper


def make_env(
    config: MsPacmanEnvConfig | None = None,
    *,
    render_mode: Literal["human", "rgb_array"] | None = None,
) -> gym.Env:
    """Create one grayscale Ms. Pac-Man environment with full-game semantics."""
    config = config or MsPacmanEnvConfig()
    gym.register_envs(ale_py)

    env = gym.make(
        "ALE/MsPacman-v5",
        frameskip=1,
        repeat_action_probability=config.repeat_action_probability,
        max_num_frames_per_episode=None,
        max_episode_steps=-1,
        mode=config.mode,
        difficulty=config.difficulty,
        render_mode=render_mode,
    )
    env = AtariPreprocessing(
        env,
        noop_max=config.noop_max,
        frame_skip=config.frame_skip,
        screen_size=config.screen_size,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = GrayscaleFrameStack(env, stack_size=config.frame_stack)
    return MsPacmanTaskWrapper(
        env,
        step_cost=config.step_cost,
        clip_training_reward=config.clip_training_reward,
        include_ram_metrics=config.include_ram_metrics,
    )


def make_vector_env(config: MsPacmanEnvConfig | None = None) -> AsyncVectorEnv:
    """Create independent ALE processes behind a Gymnasium vector interface."""
    config = config or MsPacmanEnvConfig()
    env_fns = [partial(make_env, config) for _ in range(config.num_envs)]
    return AsyncVectorEnv(
        env_fns,
        shared_memory=True,
        copy=True,
        context=config.multiprocessing_context,
        autoreset_mode=AutoresetMode.DISABLED,
    )
