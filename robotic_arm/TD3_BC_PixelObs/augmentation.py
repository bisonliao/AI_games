"""Geometry-preserving DrQ image augmentation for stacked multi-view inputs."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ConsistentMultiViewAugmentation:
    """Shift every view and history frame in one observation identically."""

    def __init__(self, shift_pixels: int = 4, n_views: int = 3, channels_per_view: int = 3):
        if shift_pixels < 0:
            raise ValueError("shift_pixels must be non-negative")
        self.shift_pixels = int(shift_pixels)
        self.n_views = int(n_views)
        self.channels_per_view = int(channels_per_view)

    def sample_offsets(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.shift_pixels == 0:
            return torch.zeros(batch_size, 2, dtype=torch.long, device=device)
        return torch.randint(
            -self.shift_pixels,
            self.shift_pixels + 1,
            (batch_size, 2),
            dtype=torch.long,
            device=device,
        )

    def apply(
        self,
        images: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.shift_pixels == 0:
            return images
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        batch, channels, height, width = images.shape
        if channels % (self.n_views * self.channels_per_view) != 0:
            raise ValueError("image channels are incompatible with stacked views")
        if offsets is None:
            offsets = self.sample_offsets(batch, images.device)
        if tuple(offsets.shape) != (batch, 2):
            raise ValueError(f"offsets must have shape ({batch}, 2)")
        offsets = offsets.to(device=images.device, dtype=torch.long)
        if torch.any(offsets.abs() > self.shift_pixels):
            raise ValueError("offset exceeds configured shift_pixels")
        padding = self.shift_pixels
        padded = F.pad(images.float(), (padding,) * 4, mode="replicate")
        windows = padded.unfold(2, height, 1).unfold(3, width, 1)
        crop_x = offsets[:, 0] + padding
        crop_y = offsets[:, 1] + padding
        batch_indices = torch.arange(batch, device=images.device)
        shifted = windows[batch_indices, :, crop_y, crop_x]
        return shifted.to(dtype=images.dtype)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.apply(images)
