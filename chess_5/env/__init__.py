"""Gymnasium-compatible Gomoku environments."""

from gymnasium.envs.registration import register
from gymnasium.error import Error as GymnasiumError

from .gomoku_env import GomokuEnv, make_gomoku_env, make_vector_env

try:
    register(
        id="Gomoku-v0",
        entry_point="env.gomoku_env:GomokuEnv",
    )
except GymnasiumError:
    # Importing this package more than once can try to register the id again.
    pass

__all__ = ["GomokuEnv", "make_gomoku_env", "make_vector_env"]
