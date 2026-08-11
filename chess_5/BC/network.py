"""全卷积残差策略网络：输入棋盘三通道，输出每个交叉点的一个 logit。"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """保持空间尺寸不变的两层残差块，跳连有助于稳定较深网络训练。"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将卷积分支与原输入相加后激活。"""
        return self.relu(x + self.body(x))


class GomokuPolicyNet(nn.Module):
    """不含全连接层的落子策略，因此可保留棋盘位置对应关系。"""
    def __init__(self, in_channels: int = 3, hidden_channels: int = 96, num_res_blocks: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True)]
        layers.extend(ResidualBlock(hidden_channels) for _ in range(num_res_blocks))
        self.trunk = nn.Sequential(*layers)
        self.policy = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                                    nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回形状 [batch, board_size * board_size] 的未归一化动作分数。"""
        return self.policy(self.trunk(x)).flatten(1)
