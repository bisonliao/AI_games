"""Dueling Q-network."""

from __future__ import annotations

import torch
from torch import nn


class DuelingDQN(nn.Module):
    def __init__(self, action_count: int, frame_stack: int = 4) -> None:
        super().__init__()
        self.action_count = action_count
        self.features = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_count),
        )
        self.value = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.features(observation.float().div_(255.0))
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
