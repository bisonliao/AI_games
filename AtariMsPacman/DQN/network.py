"""Double-DQN compatible dueling convolutional Q-network."""

from __future__ import annotations

import math

import torch
from torch import nn


class DuelingQNetwork(nn.Module):
    """Classic Atari convolutional encoder with dueling value/advantage heads."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int] = (4, 84, 84),
        action_count: int = 9,
        hidden_size: int = 512,
    ) -> None:
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError("observation_shape must be (channels, height, width)")
        if action_count <= 0:
            raise ValueError("action_count must be positive")

        channels, height, width = observation_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            feature_size = self.encoder(
                torch.zeros(1, channels, height, width, dtype=torch.float32)
            ).shape[1]

        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )
        self.apply(self._initialize_layer)

    @staticmethod
    def _initialize_layer(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float().div_(255.0)
        else:
            observations = observations.float()
        features = self.encoder(observations)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
