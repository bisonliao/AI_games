"""Parallel environment factories for pixel SAC."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from .env import (
    DEFAULT_CAMERA_SCALE,
    DEFAULT_FRAME_STACK,
    DEFAULT_IMAGE_SIZE,
    PixelTaskEnv,
)


MONITOR_INFO_KEYS = (
    "is_success",
    "success",
    "failure",
    "ever_grasped",
    "ever_lifted",
    "stage_index",
    "time_limit_reached",
    "failure_reason",
)


def make_env_factory(
    *,
    task: str,
    rank: int,
    seed: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    frame_stack: int = DEFAULT_FRAME_STACK,
    max_episode_steps: int = 150,
    action_repeat: int = 8,
    camera_scale: float = DEFAULT_CAMERA_SCALE,
    monitor_dir: Optional[Path] = None,
    render_mode: Optional[str] = None,
) -> Callable[[], gym.Env]:
    """Return a picklable pixel environment factory."""

    def _init() -> gym.Env:
        env = PixelTaskEnv(
            task=task,
            image_size=image_size,
            frame_stack=frame_stack,
            max_episode_steps=max_episode_steps,
            action_repeat=action_repeat,
            camera_scale=camera_scale,
            seed=seed + rank,
            render_mode=render_mode,
        )
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        monitor_file = None if monitor_dir is None else str(monitor_dir / f"actor_{rank}")
        return Monitor(env, filename=monitor_file, info_keywords=MONITOR_INFO_KEYS)

    return _init


__all__ = ["MONITOR_INFO_KEYS", "make_env_factory"]
