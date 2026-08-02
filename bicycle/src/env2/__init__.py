"""Public API and Gymnasium registration for the steering-balance task."""

from gymnasium.envs.registration import register, registry

from env.config import WindConfig

from .bicycle_env import BicycleSteeringEnv
from .config import BicycleSteeringEnvConfig


ENV_ID = "BicycleSteeringBalance-v0"

if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="env2:BicycleSteeringEnv")

__all__ = ["BicycleSteeringEnv", "BicycleSteeringEnvConfig", "WindConfig", "ENV_ID"]
