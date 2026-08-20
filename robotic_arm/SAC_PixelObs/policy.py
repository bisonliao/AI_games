"""CNN feature extractor for synchronized multi-view pixel observations."""

from __future__ import annotations

from typing import Dict

import torch as th
import torch.nn.functional as F
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
        augmentation_pad: int = 4,
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
        self.augmentation_pad = int(augmentation_pad)
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
        self.visual_head = nn.Sequential(
            nn.Linear(encoded_size, self.visual_feature_dim),
            nn.LayerNorm(self.visual_feature_dim),
            nn.ReLU(),
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
        image = observations["image"].float()
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
        if self.training and self.augmentation_pad > 0:
            image = self._random_shift(image, self.augmentation_pad)
        visual = self.visual_head(self.encoder(image))
        visual = visual.reshape(
            batch_size,
            self.frame_stack * self.n_views * self.visual_feature_dim,
        )
        proprio = self.proprio_head(observations["proprio"].float())
        return th.cat([visual, proprio], dim=1)

    @staticmethod
    def _random_shift(image: th.Tensor, pad: int) -> th.Tensor:
        """Apply a shared random pixel translation to each RGB crop.

        This is the small-image augmentation used by DrQ-style pixel RL. The
        same shift is applied to each RGB view/frame crop independently; it is
        disabled automatically when the policy is in evaluation mode.
        """

        batch, _, height, width = image.shape
        padded = F.pad(image, (pad, pad, pad, pad), mode="replicate")
        top = th.randint(0, 2 * pad + 1, (batch,), device=image.device)
        left = th.randint(0, 2 * pad + 1, (batch,), device=image.device)
        rows = th.arange(height, device=image.device)[None, :, None] + top[:, None, None]
        cols = th.arange(width, device=image.device)[None, None, :] + left[:, None, None]
        batch_index = th.arange(batch, device=image.device)[:, None, None]
        return padded[batch_index, :, rows, cols].permute(0, 3, 1, 2)


__all__ = ["MultiViewCombinedExtractor"]
