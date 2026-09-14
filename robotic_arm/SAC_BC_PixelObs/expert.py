"""Read-only vector teacher queried from a PixelTaskEnv state."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

from .common import DEFAULT_EXPERT


class VectorExpert:
    """Load the privileged SAC policy and query it without stepping another env."""

    def __init__(self, checkpoint=DEFAULT_EXPERT, vecnormalize=None, device="cpu") -> None:
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        prefix, steps, _ = self.checkpoint.stem.rsplit("_", 2)
        self.normalizer_path = Path(vecnormalize).resolve() if vecnormalize else (
            self.checkpoint.parent / f"{prefix}_vecnormalize_{steps}_steps.pkl"
        )
        if not self.normalizer_path.is_file():
            raise FileNotFoundError(
                f"Matching VecNormalize statistics not found: {self.normalizer_path}"
            )
        with self.normalizer_path.open("rb") as file:
            self.normalizer = pickle.load(file)
        if not isinstance(self.normalizer, VecNormalize):
            raise ValueError("The normalizer is not an SB3 VecNormalize checkpoint")
        self.normalizer.training = False
        self.normalizer.norm_reward = False
        self.model = SAC.load(self.checkpoint, device=device)
        if self.model.observation_space.shape != (52,) or self.model.action_space.shape != (4,):
            raise ValueError("Expected a 52-dimensional pick_place vector expert")

    def action(self, env) -> np.ndarray:
        """Return the expert action for the current state of a pixel environment."""
        state = env.task_env._get_observation().copy()
        normalized_state = self.normalizer.normalize_obs(state)
        action, _ = self.model.predict(normalized_state, deterministic=True)
        return np.asarray(action, dtype=np.float32)
