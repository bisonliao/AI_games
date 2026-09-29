"""Local copy of the validated multi-view visual feature extractor."""
from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MultiViewCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        n_views: int = 3,
        frame_stack: int = 2,
        visual_feature_dim: int = 64,
        proprio_feature_dim: int = 64,
        visual_head_version: int = 2,
    ) -> None:
        image_space = observation_space.spaces["image"]
        proprio_space = observation_space.spaces["proprio"]
        channels, height, width = image_space.shape
        if channels != n_views * frame_stack * 3:
            raise ValueError("image channels do not match view/frame configuration")
        super().__init__(observation_space, features_dim=1)
        self.n_views = int(n_views)
        self.frame_stack = int(frame_stack)
        self.visual_feature_dim = int(visual_feature_dim)
        self.visual_head_version = int(visual_head_version)
        if self.visual_head_version != 2:
            raise ValueError("TD3_BC_PixelObs requires visual_head_version=2")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            encoded_size = self.encoder(th.zeros(1, 3, height, width)).shape[1]
        self.visual_head = nn.Sequential(
            nn.Linear(encoded_size, visual_feature_dim),
            nn.LayerNorm(visual_feature_dim, elementwise_affine=False),
            nn.LeakyReLU(0.01),
        )
        self.proprio_head = nn.Sequential(
            nn.Linear(proprio_space.shape[0], proprio_feature_dim),
            nn.LayerNorm(proprio_feature_dim),
            nn.ReLU(),
        )
        self._features_dim = n_views * frame_stack * visual_feature_dim + proprio_feature_dim

    def encode_visual(self, image: th.Tensor) -> th.Tensor:
        image = image.float()
        if image.max().detach().item() > 1.5:
            image = image / 255.0
        batch = image.shape[0]
        image = image.reshape(batch * self.frame_stack * self.n_views, 3, image.shape[-2], image.shape[-1])
        return self.visual_head(self.encoder(image)).reshape(
            batch, self.frame_stack * self.n_views, self.visual_feature_dim
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        visual = self.encode_visual(observations["image"])
        visual = visual.reshape(visual.shape[0], -1)
        proprio = self.proprio_head(observations["proprio"].float())
        return th.cat([visual, proprio], dim=1)
