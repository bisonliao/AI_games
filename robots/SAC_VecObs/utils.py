"""Environment factories shared by training and evaluation commands."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from .env import SACVectorTaskEnv


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
    max_episode_steps: int,
    action_repeat: int,
    monitor_dir: Optional[Path] = None,
    render_mode: Optional[str] = None,
) -> Callable[[], gym.Env]:
    """Return a picklable factory for DummyVecEnv/SubprocVecEnv."""

    def _init() -> gym.Env:
        env = SACVectorTaskEnv(
            task=task,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            action_repeat=action_repeat,
            seed=seed + rank,
        )
        # Defense in depth: SACVectorTaskEnv enforces the same limit itself,
        # while this standard wrapper prevents a future task adapter regression
        # from creating an unbounded episode inside SB3 evaluation.
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        monitor_file = None
        if monitor_dir is not None:
            monitor_file = str(monitor_dir / f"actor_{rank}")
        return Monitor(env, filename=monitor_file, info_keywords=MONITOR_INFO_KEYS)

    return _init


__all__ = ["MONITOR_INFO_KEYS", "make_env_factory"]
