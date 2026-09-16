"""Shared configuration and SAC-compatible visual actor helpers."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.utils import get_device, set_random_seed
from stable_baselines3.sac.policies import Actor

from SAC_PixelObs.env import PixelTaskEnv
from SAC_PixelObs.policy import MultiViewCombinedExtractor

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
DEFAULT_EXPERT = REPO / (
    "SAC_VecObs/runs/pick_place_20260915_101759/checkpoints/"
    "pick_place_sac_2000000_steps.zip"
)
FORMAT_VERSION = 2
TB_LOG_ROOT = REPO / "tb_logs"


@dataclass(frozen=True)
class EnvConfig:
    """Pixel environment settings persisted in every model checkpoint."""

    task: str = "pick_place"
    image_size: int = 96
    frame_stack: int = 2
    camera_scale: float = 1.0
    max_episode_steps: int = 150
    action_repeat: int = 8

    def __post_init__(self) -> None:
        if self.task != "pick_place":
            raise ValueError("This pipeline currently supports pick_place only")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if min(self.frame_stack, self.max_episode_steps, self.action_repeat) < 1:
            raise ValueError("stack, horizon and repeat must be positive")
        if not math.isfinite(self.camera_scale) or self.camera_scale <= 0:
            raise ValueError("camera_scale must be finite and positive")

    def make_env(self, seed: int = 0, gui: bool = False, randomize: bool = True):
        from .randomized_env import RandomizedPixelTaskEnv

        return RandomizedPixelTaskEnv(
            task=self.task,
            image_size=self.image_size,
            frame_stack=self.frame_stack,
            camera_scale=self.camera_scale,
            max_episode_steps=self.max_episode_steps,
            action_repeat=self.action_repeat,
            seed=seed,
            render_mode="human" if gui else None,
            randomize=randomize,
        )

    def spaces(self) -> tuple[spaces.Dict, spaces.Box]:
        observation = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(9 * self.frame_stack, self.image_size, self.image_size),
                    dtype=np.uint8,
                ),
                "proprio": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(26,),
                    dtype=np.float32,
                ),
            }
        )
        action = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        return observation, action


def output_path(path: str | Path) -> Path:
    """Resolve an artifact path and keep writes inside this package."""

    result = Path(path).expanduser().resolve()
    if not result.is_relative_to(PACKAGE):
        raise ValueError(f"Output must be inside {PACKAGE}: {result}")
    return result


def new_run(stage: str, requested: str | Path | None = None) -> Path:
    path = output_path(
        requested
        or PACKAGE / "runs" / f"{stage}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def new_tensorboard_run(stage: str) -> Path:
    """Create a stage/timestamp/PID log directory under the fixed TB root."""
    if stage not in {"bc_pretrain", "sac_finetune", "evaluate"}:
        raise ValueError(f"Unsupported TensorBoard stage: {stage}")

    run_name = f"{stage}_{datetime.now():%Y%m%d_%H%M%S}_pid{os.getpid()}"
    # TensorBoard is the one package artifact intentionally stored at the
    # repository root so all experiments share one dashboard directory.
    log_root = TB_LOG_ROOT.resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    # A process may start the same stage twice within one second. Keep those
    # runs separate instead of merging their event files.
    suffix = 1
    while True:
        directory_name = run_name if suffix == 1 else f"{run_name}_{suffix}"
        log_dir = log_root / directory_name
        try:
            log_dir.mkdir(exist_ok=False)
        except FileExistsError:
            suffix += 1
        else:
            return log_dir


def write_json(path: str | Path, data: Any) -> None:
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_json(path: str | Path, data: Any) -> None:
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def setup(seed: int, device: str = "auto", threads: int = 4) -> torch.device:
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    set_random_seed(seed)
    return get_device(device)


def policy_kwargs(config: EnvConfig) -> dict[str, Any]:
    return {
        "net_arch": [256, 256],
        "features_extractor_class": MultiViewCombinedExtractor,
        "features_extractor_kwargs": {
            "n_views": 3,
            "frame_stack": config.frame_stack,
            "visual_head_version": 2,
        },
        "share_features_extractor": False,
    }


def make_actor(
    config: EnvConfig,
    device: str | torch.device = "cpu",
    initial_std: float = 0.1,
) -> Actor:
    observation_space, action_space = config.spaces()
    extractor = MultiViewCombinedExtractor(
        observation_space,
        n_views=3,
        frame_stack=config.frame_stack,
        visual_head_version=2,
    )
    actor = Actor(
        observation_space,
        action_space,
        [256, 256],
        extractor,
        extractor.features_dim,
    ).to(device)
    with torch.no_grad():
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(math.log(initial_std))
    return actor


def save_actor(path: str | Path, actor: Actor, config: EnvConfig, **metadata: Any) -> None:
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "env_config": asdict(config),
        "actor_state": {
            key: value.detach().cpu() for key, value in actor.state_dict().items()
        },
        **metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_actor(path: str | Path, device: str | torch.device = "cpu") -> tuple[Actor, EnvConfig, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION or "actor_state" not in payload:
        raise ValueError("Incompatible BC checkpoint; regenerate it with this pipeline")
    config = EnvConfig(**payload["env_config"])
    actor = make_actor(config, device)
    actor.load_state_dict(payload["actor_state"], strict=True)
    actor.set_training_mode(False)
    return actor, config, payload


def predict_actor(actor: Actor, observation: dict[str, np.ndarray]) -> np.ndarray:
    with torch.inference_mode():
        tensors, _ = actor.obs_to_tensor(observation)
        return actor(tensors, deterministic=True).cpu().numpy()[0]
