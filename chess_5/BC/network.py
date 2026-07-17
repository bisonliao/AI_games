"""Fully convolutional residual policy network."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.body(x))


class GomokuPolicyNet(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 96, num_res_blocks: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True)]
        layers.extend(ResidualBlock(hidden_channels) for _ in range(num_res_blocks))
        self.trunk = nn.Sequential(*layers)
        self.policy = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                                    nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(self.trunk(x)).flatten(1)
