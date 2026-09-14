"""Prior/online replay with explicit source-ratio sampling."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch


class TransitionReplay:
    """Small in-memory replay used to hold one source of transitions."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.data = deque(maxlen=int(capacity))
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated, truncated=False) -> None:
        self.data.append(
            (
                {key: np.array(value, copy=True) for key, value in obs.items()},
                np.array(action, dtype=np.float32, copy=True),
                float(reward),
                {key: np.array(value, copy=True) for key, value in next_obs.items()},
                bool(terminated),
                bool(truncated),
            )
        )

    def extend_episode(self, episode: dict) -> None:
        last = len(episode["action"]) - 1
        for index in range(last + 1):
            next_index = min(index + 1, last)
            self.add(
                {"image": episode["image"][index], "proprio": episode["proprio"][index]},
                episode["action"][index],
                episode["reward"][index],
                {"image": episode["image"][next_index], "proprio": episode["proprio"][next_index]},
                episode["terminated"][index],
                episode["truncated"][index],
            )

    def __len__(self) -> int:
        return len(self.data)

    def sample_rows(self, count: int) -> list:
        if len(self) < count:
            raise ValueError(f"Need {count} samples, have {len(self)}")
        indices = self.rng.choice(len(self), count, replace=False)
        return [self.data[int(index)] for index in indices]


class MixedReplay:
    """Replay that samples a requested prior/online mixture when possible."""

    def __init__(self, capacity: int = 100_000, prior_ratio: float = 0.25, seed: int = 0) -> None:
        if not 0.0 <= prior_ratio <= 1.0:
            raise ValueError("prior_ratio must be in [0, 1]")
        self.prior = TransitionReplay(capacity, seed)
        self.online = TransitionReplay(capacity, seed + 1)
        self.prior_ratio = float(prior_ratio)
        self.rng = np.random.default_rng(seed + 2)

    def add_prior_episode(self, episode: dict) -> None:
        self.prior.extend_episode(episode)

    def add_online(self, *args, **kwargs) -> None:
        self.online.add(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.prior) + len(self.online)

    def sample(self, batch_size: int, device: str = "cpu"):
        requested_prior = int(round(batch_size * self.prior_ratio))
        prior_count = min(requested_prior, len(self.prior))
        online_count = min(batch_size - prior_count, len(self.online))
        prior_count = min(batch_size - online_count, len(self.prior))
        if prior_count + online_count < batch_size:
            raise ValueError("Insufficient prior/online transitions")

        rows = self.prior.sample_rows(prior_count) + self.online.sample_rows(online_count)
        self.rng.shuffle(rows)
        observations = {
            key: torch.as_tensor(np.stack([row[0][key] for row in rows]), device=device)
            for key in rows[0][0]
        }
        next_observations = {
            key: torch.as_tensor(np.stack([row[3][key] for row in rows]), device=device)
            for key in rows[0][3]
        }
        actions = torch.as_tensor(np.stack([row[1] for row in rows]), device=device)
        rewards = torch.tensor([row[2] for row in rows], dtype=torch.float32, device=device).reshape(-1, 1)
        dones = torch.tensor(
            [row[4] or row[5] for row in rows], dtype=torch.float32, device=device
        ).reshape(-1, 1)
        return observations, actions, rewards, next_observations, dones, prior_count, online_count
