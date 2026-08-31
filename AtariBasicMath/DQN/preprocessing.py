from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


EncodedObservation = tuple[np.ndarray, np.ndarray]


def _grayscale_resize(image: np.ndarray, size: int = 84) -> np.ndarray:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    pil_image = pil_image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil_image, dtype=np.uint8).copy()


def encode_observation(observation: Any, goal_conditioned: bool) -> EncodedObservation:
    if goal_conditioned:
        current = _grayscale_resize(observation["current"])
        goal = _grayscale_resize(observation["goal"])
        pixels = np.stack((current, goal), axis=0)
        macro = np.concatenate(
            (
                np.asarray(observation["macro"], dtype=np.float32),
                np.asarray(observation["current_answer"], dtype=np.float32),
                np.asarray(observation["cursor"], dtype=np.float32),
            )
        ).astype(np.float32, copy=False)
    else:
        pixels = _grayscale_resize(observation)[None, ...]
        macro = np.empty((0,), dtype=np.float32)
    return pixels, macro


def observation_to_tensors(
    observation: EncodedObservation,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels, macro = observation
    image_tensor = torch.from_numpy(pixels).unsqueeze(0).to(device=device, dtype=torch.float32)
    image_tensor = image_tensor.div_(255.0)
    macro_tensor = torch.from_numpy(macro).unsqueeze(0).to(device=device, dtype=torch.float32)
    return image_tensor, macro_tensor


def batch_to_tensors(
    pixels: np.ndarray,
    macro: np.ndarray,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_tensor = torch.from_numpy(pixels).to(device=device, dtype=torch.float32).div_(255.0)
    macro_tensor = torch.from_numpy(macro).to(device=device, dtype=torch.float32)
    return image_tensor, macro_tensor
