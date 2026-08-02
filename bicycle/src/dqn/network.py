"""Dueling Q-network and NumPy-based actor weight synchronization helpers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    """Two-layer MLP with separate state-value and action-advantage heads."""

    def __init__(self, observation_dim: int = 5, action_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._reset_parameters()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Map a batch of normalized observations to one Q-value per action."""
        features = self.trunk(observations)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value[-1].weight, gain=1.0)
        nn.init.orthogonal_(self.advantage[-1].weight, gain=0.01)


def numpy_state_dict(module: nn.Module) -> dict[str, np.ndarray]:
    """Copy model weights to NumPy so multiprocessing never shares CUDA storage."""
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in module.state_dict().items()
    }


def load_numpy_state_dict(module: nn.Module, state: dict[str, np.ndarray]) -> None:
    """Load weights received by a CPU actor or evaluator process."""
    module.load_state_dict({key: torch.from_numpy(value) for key, value in state.items()})
