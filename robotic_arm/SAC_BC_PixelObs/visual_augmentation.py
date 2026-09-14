"""Geometry-preserving augmentation for stacked synchronized views."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ConsistentMultiViewAugmentation:
    """Apply one image-space translation to every view and history frame."""

    def __init__(self, shift_pixels: int = 4, n_views: int = 3, channels_per_view: int = 3):
        if shift_pixels < 0:
            raise ValueError("shift_pixels must be non-negative")
        self.shift_pixels = int(shift_pixels)
        self.n_views = int(n_views)
        self.channels_per_view = int(channels_per_view)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if self.shift_pixels == 0:
            return images
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        batch, channels, height, width = images.shape
        group_size = self.n_views * self.channels_per_view
        if channels % group_size != 0:
            raise ValueError("image channels are incompatible with stacked views")
        frames = channels // group_size
        # The same offset is sampled for every frame and view of one sample.
        offsets = torch.randint(
            -self.shift_pixels,
            self.shift_pixels + 1,
            (batch, 2),
            device=images.device,
        )
        padded = F.pad(images.float(), (self.shift_pixels,) * 4, mode="replicate")
        output = torch.empty_like(images.float())
        for index in range(batch):
            dx, dy = [int(value) for value in offsets[index]]
            top = self.shift_pixels + dy
            left = self.shift_pixels + dx
            output[index] = padded[index, :, top : top + height, left : left + width]
        return output.to(dtype=images.dtype)


__all__ = ["ConsistentMultiViewAugmentation"]
