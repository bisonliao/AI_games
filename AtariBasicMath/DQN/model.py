from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NetworkSpec:
    input_channels: int
    macro_dim: int
    num_actions: int
    image_size: int = 84

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class DuelingDQN(nn.Module):
    def __init__(self, spec: NetworkSpec) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = nn.Sequential(
            nn.Conv2d(spec.input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, spec.input_channels, spec.image_size, spec.image_size)
            encoded_dim = int(self.encoder(dummy).shape[1])

        self.trunk = nn.Sequential(
            nn.Linear(encoded_dim + spec.macro_dim, 512),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, spec.num_actions),
        )

    def forward(self, images: torch.Tensor, macro: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder(images)
        if self.spec.macro_dim:
            if macro is None:
                raise ValueError("macro input is required by this network")
            features = torch.cat((features, macro), dim=1)
        features = self.trunk(features)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
