"""Versioned episode storage used by behavior cloning and SAC."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .common import FORMAT_VERSION, output_path

REQUIRED_FIELDS = (
    "image",
    "proprio",
    "action",
    "reward",
    "terminated",
    "truncated",
)


def save_episode(
    path: str | Path,
    *,
    image,
    proprio,
    action,
    reward,
    terminated,
    truncated,
    success: bool,
    episode_return: float,
    stage,
    failure_reason: str = "",
    episode_metadata: dict | None = None,
) -> None:
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "image": np.asarray(image, dtype=np.uint8),
        "proprio": np.asarray(proprio, dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "reward": np.asarray(reward, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "success": np.asarray(bool(success)),
        "episode_return": np.asarray(float(episode_return), dtype=np.float32),
        "stage": np.asarray(stage, dtype=np.int8),
        "failure_reason": np.asarray(str(failure_reason)),
        "episode_metadata": np.asarray(json.dumps(episode_metadata or {}, default=str)),
    }
    lengths = {key: len(value) for key, value in arrays.items() if key in REQUIRED_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent episode lengths: {lengths}")
    np.savez_compressed(path, format_version=np.asarray(FORMAT_VERSION), **arrays)


def load_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as raw:
        missing = [key for key in REQUIRED_FIELDS if key not in raw]
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        result = {key: raw[key].copy() for key in raw.files}
    if int(result.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"Unsupported dataset format in {path}")
    return result


def episode_paths(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("episode_*.npz"))


def load_episodes(root: str | Path, successful_only: bool = False) -> list[dict]:
    episodes = []
    for path in episode_paths(root):
        episode = load_episode(path)
        if successful_only and not bool(episode["success"]):
            continue
        episodes.append(episode)
    return episodes


def split_episodes(
    episodes: list[dict],
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Split complete episodes so no transition can cross the split boundary."""
    if len(episodes) < 5:
        raise ValueError("At least five episodes are required for episode-level splitting")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    indices = list(range(len(episodes)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, int(round(len(indices) * validation_fraction)))
    validation_indices = set(indices[:validation_count])
    train = [episode for index, episode in enumerate(episodes) if index not in validation_indices]
    validation = [episode for index, episode in enumerate(episodes) if index in validation_indices]
    if not train or not validation:
        raise ValueError("Episode split produced an empty train or validation set")
    return train, validation


class EpisodeDataset:
    """Transition-level indexing backed by complete, pre-split episodes."""

    def __init__(self, episodes: list[dict]):
        self.episodes = list(episodes)
        self.index: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            self.index.extend(
                (episode_index, transition_index)
                for transition_index in range(len(episode["action"]))
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        episode_index, transition_index = self.index[index]
        episode = self.episodes[episode_index]
        return (
            episode["image"][transition_index],
            episode["proprio"][transition_index],
            episode["action"][transition_index],
        )


def write_metadata(root: str | Path, **metadata) -> None:
    root = output_path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "metadata.json.tmp"
    temporary.write_text(
        json.dumps(
            {"format_version": FORMAT_VERSION, **metadata},
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    temporary.replace(root / "metadata.json")
