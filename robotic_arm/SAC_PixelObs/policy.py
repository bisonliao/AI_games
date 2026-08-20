"""CNN feature extractor for synchronized multi-view pixel observations."""

from __future__ import annotations

from typing import Dict

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MultiViewCombinedExtractor(BaseFeaturesExtractor):
    """Shared RGB encoder per view/frame plus a proprioception branch."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        n_views: int = 3,
        frame_stack: int = 3,
        visual_feature_dim: int = 64,
        proprio_feature_dim: int = 64,
        visual_head_version: int = 1,
    ) -> None:
        image_space = observation_space.spaces["image"]
        proprio_space = observation_space.spaces["proprio"]
        channels, height, width = image_space.shape
        expected_channels = n_views * frame_stack * 3
        if channels != expected_channels:
            raise ValueError(
                f"image has {channels} channels; expected {expected_channels} "
                f"for {frame_stack} frames and {n_views} views"
            )
        if len(proprio_space.shape) != 1:
            raise ValueError("proprio observation must be one-dimensional")

        # BaseFeaturesExtractor initializes torch.nn.Module; child modules
        # must only be assigned after this call.
        super().__init__(observation_space, features_dim=1)
        self.n_views = int(n_views)
        self.frame_stack = int(frame_stack)
        self.visual_feature_dim = int(visual_feature_dim)
        self.visual_head_version = int(visual_head_version)
        if self.visual_head_version not in {1, 2}:
            raise ValueError("visual_head_version must be 1 or 2")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.zeros(1, 3, height, width)
            encoded_size = int(self.encoder(sample).shape[1])
        if self.visual_head_version == 1:
            # Kept only so checkpoints created before the visual-collapse fix
            # can still be loaded with their original architecture.
            self.visual_head = nn.Sequential(
                nn.Linear(encoded_size, self.visual_feature_dim),
                nn.LayerNorm(self.visual_feature_dim),
                nn.ReLU(),
            )
        else:
            # A learned LayerNorm bias drove every visual feature below zero
            # in previous runs; the following ReLU then made the branch
            # permanently gradient-dead.  A non-affine normalization cannot
            # learn that global negative shift, and LeakyReLU retains a
            # gradient even for negative activations.
            self.visual_head = nn.Sequential(
                nn.Linear(encoded_size, self.visual_feature_dim),
                nn.LayerNorm(self.visual_feature_dim, elementwise_affine=False),
                nn.LeakyReLU(negative_slope=0.01),
            )
        self.proprio_head = nn.Sequential(
            nn.Linear(proprio_space.shape[0], proprio_feature_dim),
            nn.LayerNorm(proprio_feature_dim),
            nn.ReLU(),
        )
        features_dim = (
            self.n_views * self.frame_stack * self.visual_feature_dim
            + proprio_feature_dim
        )
        self._features_dim = features_dim

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        visual = self.encode_visual(observations["image"])
        batch_size = visual.shape[0]
        visual = visual.reshape(
            batch_size,
            self.frame_stack * self.n_views * self.visual_feature_dim,
        )
        proprio = self.proprio_head(observations["proprio"].float())
        return th.cat([visual, proprio], dim=1)

    def encode_visual(self, image: th.Tensor) -> th.Tensor:
        """Encode RGB crops as ``(batch, frame*view, feature)``.

        This public, side-effect-free path is also used by the low-frequency
        visual-health probe, avoiding a second implementation of image
        preprocessing and view reshaping.
        """

        image = image.float()
        if image.max().detach().item() > 1.5:
            image = image / 255.0
        batch_size = image.shape[0]
        image = image.reshape(
            batch_size,
            self.frame_stack * self.n_views,
            3,
            image.shape[-2],
            image.shape[-1],
        )
        image = image.reshape(
            batch_size * self.frame_stack * self.n_views,
            3,
            image.shape[-2],
            image.shape[-1],
        )
        return self.visual_head(self.encoder(image)).reshape(
            batch_size,
            self.frame_stack * self.n_views,
            self.visual_feature_dim,
        )


__all__ = ["MultiViewCombinedExtractor"]
