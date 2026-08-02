"""Shared environment-number resolution for actors, training, and evaluation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from env import BicycleBalanceEnv, ENV_ID as REACTION_WHEEL_ENV_ID
from env2 import BicycleSteeringEnv, ENV_ID as STEERING_ENV_ID


ENVIRONMENT_IDS = {1: REACTION_WHEEL_ENV_ID, 2: STEERING_ENV_ID}


def gym_environment_id(env_id: int) -> str:
    """Translate the user-facing integer ID to a registered Gymnasium ID."""
    try:
        return ENVIRONMENT_IDS[int(env_id)]
    except (KeyError, ValueError) as error:
        raise ValueError(f"env_id must be one of {sorted(ENVIRONMENT_IDS)}") from error


def make_registered_environment(env_id: int) -> gym.Env:
    """Construct a headless registered environment for vector workers."""
    return gym.make(gym_environment_id(env_id))


def make_evaluation_environment(env_id: int, display: bool = False) -> Any:
    """Construct the selected unwrapped environment, optionally with GUI."""
    render_mode = "human" if display else None
    if env_id == 1:
        return BicycleBalanceEnv(render_mode=render_mode)
    if env_id == 2:
        return BicycleSteeringEnv(render_mode=render_mode)
    raise ValueError(f"env_id must be one of {sorted(ENVIRONMENT_IDS)}")
