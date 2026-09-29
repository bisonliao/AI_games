"""Validated, fixed offline transition storage for expert episodes."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import EnvConfig

REQUIRED = ("image", "proprio", "action", "reward", "terminated", "truncated", "stage")


@dataclass
class OfflineBatch:
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: dict[str, torch.Tensor]
    terminals: torch.Tensor
    stages: torch.Tensor


class OfflineDataset:
    def __init__(
        self,
        root: Path,
        config: EnvConfig,
        successful_only: bool = True,
        seed: int = 0,
    ):
        self.root = Path(root).expanduser().resolve()
        self.config = config
        self.successful_only = bool(successful_only)
        self.rng = np.random.default_rng(seed)
        self.episodes = []
        paths = sorted(self.root.glob("episode_*.npz"))
        if not paths:
            raise FileNotFoundError(f"no episode_*.npz files under {self.root}")
        for path in paths:
            with np.load(path, allow_pickle=False) as raw:
                missing = [key for key in REQUIRED if key not in raw]
                if missing:
                    raise ValueError(f"{path} missing fields {missing}")
                if int(raw.get("format_version", -1)) != 2:
                    raise ValueError(f"unsupported format in {path}")
                success = bool(raw["success"]) if "success" in raw else False
                if self.successful_only and not success:
                    continue
                episode = {key: raw[key].copy() for key in raw.files}
            self._validate_episode(path, episode)
            self.episodes.append(episode)
        if not self.episodes:
            raise ValueError("no usable episodes found")
        self._build_arrays()

    def _validate_episode(self, path: Path, episode: dict) -> None:
        length = len(episode["action"])
        if length < 2:
            raise ValueError(f"{path} is too short")
        if any(len(episode[key]) != length for key in REQUIRED):
            raise ValueError(f"inconsistent lengths in {path}")
        expected_image = (9 * self.config.frame_stack, self.config.image_size, self.config.image_size)
        if tuple(episode["image"].shape[1:]) != expected_image:
            raise ValueError(f"{path} image shape is {episode['image'].shape[1:]}, expected {expected_image}")
        if tuple(episode["proprio"].shape[1:]) != (26,) or tuple(episode["action"].shape[1:]) != (4,):
            raise ValueError(f"{path} has incompatible proprio/action shapes")

    def _build_arrays(self) -> None:
        images, proprios, actions, rewards, next_images, next_proprios = [], [], [], [], [], []
        terminals, stages, episode_ids = [], [], []
        for episode_id, episode in enumerate(self.episodes):
            n = len(episode["action"])
            images.append(episode["image"])
            proprios.append(episode["proprio"])
            actions.append(episode["action"].astype(np.float32))
            rewards.append(episode["reward"].astype(np.float32))
            next_images.append(np.concatenate([episode["image"][1:], episode["image"][-1:]], axis=0))
            next_proprios.append(np.concatenate([episode["proprio"][1:], episode["proprio"][-1:]], axis=0))
            terminals.append(episode["terminated"].astype(np.float32))
            stages.append(episode["stage"].astype(np.int64))
            episode_ids.append(np.full(n, episode_id, dtype=np.int64))
        self.image = np.concatenate(images)
        self.proprio = np.concatenate(proprios)
        self.action = np.concatenate(actions)
        self.reward = np.concatenate(rewards)
        self.next_image = np.concatenate(next_images)
        self.next_proprio = np.concatenate(next_proprios)
        self.terminal = np.concatenate(terminals)
        self.stage = np.concatenate(stages)
        self.episode_id = np.concatenate(episode_ids)

    def __len__(self) -> int:
        return len(self.action)

    def sample(self, batch_size: int, device: torch.device | str) -> OfflineBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = self.rng.integers(0, len(self), size=batch_size)
        tensor = lambda value: torch.as_tensor(value[indices], device=device)
        return OfflineBatch(
            observations={"image": tensor(self.image), "proprio": tensor(self.proprio)},
            actions=tensor(self.action).float(),
            rewards=tensor(self.reward).float().reshape(-1, 1),
            next_observations={"image": tensor(self.next_image), "proprio": tensor(self.next_proprio)},
            terminals=tensor(self.terminal).float().reshape(-1, 1),
            stages=tensor(self.stage).long(),
        )

    def metadata(self) -> dict:
        digest = hashlib.sha256()
        for path in sorted(self.root.glob("episode_*.npz")):
            digest.update(path.name.encode())
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
        return {
            "root": str(self.root),
            "sha256": digest.hexdigest(),
            "episodes": len(self.episodes),
            "transitions": len(self),
            "successful_only": self.successful_only,
            "env_config": self.config.as_dict(),
        }

    def write_metadata(self, path: Path) -> None:
        path.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
