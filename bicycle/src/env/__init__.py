"""Public environment API and Gymnasium registration for BicycleBalance-v0."""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

from .bicycle_env import BicycleBalanceEnv
from .config import BicycleEnvConfig, WindConfig


ENV_ID = "BicycleBalance-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="env:BicycleBalanceEnv")

__all__ = ["BicycleBalanceEnv", "BicycleEnvConfig", "WindConfig", "ENV_ID"]
