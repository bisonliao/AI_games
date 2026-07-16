"""DQN网络结构：定义带残差主干、优势头和价值头的Dueling Gomoku Q网络。"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DuelingGomokuQNet(nn.Module):
    """Board-size agnostic dueling Q-network for Gomoku.

    Input shape is ``[batch, 3, board_size, board_size]``:
    own stones, opponent stones, empty cells.
    Output shape is ``[batch, board_size * board_size]``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 96,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        ]
        layers.extend(ResidualBlock(hidden_channels) for _ in range(num_res_blocks))
        self.trunk = nn.Sequential(*layers)

        self.advantage = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        advantage = self.advantage(features).flatten(start_dim=1)
        value = self.value(self.value_pool(features))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
