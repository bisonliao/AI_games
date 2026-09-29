"""Configuration and artifact paths for the offline TD3+BC pipeline."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.utils import get_device, set_random_seed

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
DEFAULT_DATASET = REPO / "SAC_BC_PixelObs" / "runs" / "expert_showcase_v3"
DEFAULT_BC_MODEL = REPO / "SAC_BC_PixelObs" / "runs" / "bc_model.pt"
TB_ROOT = REPO / "tb_logs"


@dataclass(frozen=True)
class EnvConfig:
    task: str = "pick_place"
    image_size: int = 96
    frame_stack: int = 2
    camera_scale: float = 1.0
    max_episode_steps: int = 150
    action_repeat: int = 8

    def __post_init__(self) -> None:
        if self.task != "pick_place":
            raise ValueError("only pick_place is supported")
        if self.image_size < 32 or self.frame_stack < 1:
            raise ValueError("image_size and frame_stack are invalid")
        if self.max_episode_steps < 1 or self.action_repeat < 1:
            raise ValueError("episode and action limits must be positive")
        if self.camera_scale <= 0:
            raise ValueError("camera_scale must be positive")

    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "image": spaces.Box(
                    0,
                    255,
                    shape=(9 * self.frame_stack, self.image_size, self.image_size),
                    dtype=np.uint8,
                ),
                "proprio": spaces.Box(
                    -np.inf, np.inf, shape=(26,), dtype=np.float32
                ),
            }
        )

    def action_space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def as_dict(self) -> dict:
        return asdict(self)


def setup(seed: int, device: str = "auto", threads: int = 8) -> torch.device:
    """Configure reproducibility and bounded CPU parallelism."""
    torch.set_num_threads(max(1, int(threads)))
    set_random_seed(seed)
    return get_device(device)


def run_directory(requested: Path | None = None) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
        if not path.is_relative_to(PACKAGE):
            raise ValueError(f"run directory must be under {PACKAGE}")
    else:
        path = PACKAGE / "runs" / f"td3_bc_{datetime.now():%Y%m%d_%H%M%S}_pid{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def tensorboard_directory() -> Path:
    TB_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"td3_bc_{datetime.now():%Y%m%d_%H%M%S}_pid{os.getpid()}"
    suffix = 1
    while True:
        name = stem if suffix == 1 else f"{stem}_{suffix}"
        path = TB_ROOT / name
        try:
            path.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            return path
