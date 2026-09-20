"""Geometry-preserving augmentation for stacked synchronized views."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ConsistentMultiViewAugmentation:
    """Apply one image-space translation to every view and history frame.

    Offsets are sampled independently for batch elements, but every channel of
    one observation receives the same crop. Since views and history frames are
    packed along the channel axis, this preserves their geometric alignment.
    """

    def __init__(self, shift_pixels: int = 4, n_views: int = 3, channels_per_view: int = 3):
        if shift_pixels < 0:
            raise ValueError("shift_pixels must be non-negative")
        self.shift_pixels = int(shift_pixels)
        self.n_views = int(n_views)
        self.channels_per_view = int(channels_per_view)

    def sample_offsets(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Sample integer ``(x, y)`` shifts for a batch."""
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
        """Replicate-pad and crop a complete batch without Python loops."""
        if self.shift_pixels == 0:
            return images
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        batch, channels, height, width = images.shape
        group_size = self.n_views * self.channels_per_view
        if channels % group_size != 0:
            raise ValueError("image channels are incompatible with stacked views")
        if offsets is None:
            offsets = self.sample_offsets(batch, images.device)
        if offsets.shape != (batch, 2):
            raise ValueError(
                f"offsets must have shape ({batch}, 2), got {tuple(offsets.shape)}"
            )
        offsets = offsets.to(device=images.device, dtype=torch.long)
        if torch.any(offsets.abs() > self.shift_pixels):
            raise ValueError("offset exceeds configured shift_pixels")

        original_dtype = images.dtype
        padding = self.shift_pixels
        padded = F.pad(images.float(), (padding,) * 4, mode="replicate")
        # unfold creates a view over all possible HxW crops. Advanced indexing
        # then selects one crop per batch element without CPU synchronization.
        windows = padded.unfold(2, height, 1).unfold(3, width, 1)
        crop_x = offsets[:, 0] + padding
        crop_y = offsets[:, 1] + padding
        batch_indices = torch.arange(batch, device=images.device)
        shifted = windows[batch_indices, :, crop_y, crop_x]
        return shifted.to(dtype=original_dtype)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.apply(images)


__all__ = ["ConsistentMultiViewAugmentation"]
