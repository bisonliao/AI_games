"""Atomic training checkpoint save/load helpers with legacy compatibility."""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import DQNConfig
from .learner import DQNLearner
from .replay import ReplayBuffer


def save_checkpoint(
    path: str | Path,
    learner: DQNLearner,
    config: DQNConfig,
    env_steps: int,
    rng: np.random.Generator,
    replay: ReplayBuffer | None = None,
    best_success_rate: float = 0.0,
    env_id: int = 1,
) -> None:
    """Atomically save learner state and optionally the potentially large replay."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "format_version": 2,
        "config": asdict(config),
        "learner": learner.state_dict(),
        "env_steps": int(env_steps),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "best_success_rate": float(best_success_rate),
        "env_id": int(env_id),
    }
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    if replay is not None:
        state["replay"] = replay.state_dict()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    learner: DQNLearner | None = None,
    replay: ReplayBuffer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint versions 1 or 2 and optionally restore live objects."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    if int(state.get("format_version", 0)) not in (1, 2):
        raise ValueError("unsupported checkpoint format")
    if learner is not None:
        learner.load_state_dict(state["learner"])
    if replay is not None and "replay" in state:
        replay.load_state_dict(state["replay"])
    return state


def online_state_from_checkpoint(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract online-network weights for standalone evaluation."""
    return state["learner"]["online"]
