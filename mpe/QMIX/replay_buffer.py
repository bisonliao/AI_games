"""Episode replay for recurrent QMIX."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass
class Episode:
    observations: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    filled: np.ndarray
    length: int


class EpisodeBuilder:
    """Build one episode, retaining the final t+1 observation and state."""

    def __init__(
        self,
        initial_observations: np.ndarray,
        initial_state: np.ndarray,
        max_episode_len: int,
    ) -> None:
        self.max_episode_len = int(max_episode_len)
        self._observations = [np.asarray(initial_observations, dtype=np.float32)]
        self._states = [np.asarray(initial_state, dtype=np.float32)]
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._terminated: list[float] = []

    def add(
        self,
        actions: np.ndarray,
        reward: float,
        terminated: bool,
        next_observations: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        if len(self._actions) >= self.max_episode_len:
            raise RuntimeError("episode builder exceeded max_episode_len")
        self._actions.append(np.asarray(actions, dtype=np.int64))
        self._rewards.append(float(reward))
        self._terminated.append(float(terminated))
        self._observations.append(
            np.asarray(next_observations, dtype=np.float32)
        )
        self._states.append(np.asarray(next_state, dtype=np.float32))

    def finish(self) -> Episode:
        length = len(self._actions)
        if length == 0:
            raise RuntimeError("cannot store an empty episode")
        n_agents, obs_dim = self._observations[0].shape
        state_dim = self._states[0].shape[0]
        observations = np.zeros(
            (self.max_episode_len + 1, n_agents, obs_dim), dtype=np.float32
        )
        states = np.zeros(
            (self.max_episode_len + 1, state_dim), dtype=np.float32
        )
        actions = np.zeros(
            (self.max_episode_len, n_agents), dtype=np.int64
        )
        rewards = np.zeros((self.max_episode_len, 1), dtype=np.float32)
        terminated = np.zeros((self.max_episode_len, 1), dtype=np.float32)
        filled = np.zeros((self.max_episode_len, 1), dtype=np.float32)

        observations[: length + 1] = np.asarray(self._observations)
        states[: length + 1] = np.asarray(self._states)
        actions[:length] = np.asarray(self._actions)
        rewards[:length, 0] = np.asarray(self._rewards)
        terminated[:length, 0] = np.asarray(self._terminated)
        filled[:length, 0] = 1.0
        return Episode(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            filled=filled,
            length=length,
        )


class EpisodeReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self._episodes: list[Episode] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._episodes)

    def add(self, episode: Episode) -> None:
        if len(self._episodes) < self.capacity:
            self._episodes.append(episode)
        else:
            self._episodes[self._next_index] = episode
        self._next_index = (self._next_index + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if batch_size > len(self._episodes):
            raise ValueError(
                f"cannot sample {batch_size} episodes from {len(self._episodes)}"
            )
        episodes = random.sample(self._episodes, batch_size)

        def stack(name: str) -> np.ndarray:
            return np.stack([getattr(episode, name) for episode in episodes])

        return {
            "observations": torch.as_tensor(
                stack("observations"), dtype=torch.float32, device=device
            ),
            "states": torch.as_tensor(
                stack("states"), dtype=torch.float32, device=device
            ),
            "actions": torch.as_tensor(
                stack("actions"), dtype=torch.long, device=device
            ),
            "rewards": torch.as_tensor(
                stack("rewards"), dtype=torch.float32, device=device
            ),
            "terminated": torch.as_tensor(
                stack("terminated"), dtype=torch.float32, device=device
            ),
            "filled": torch.as_tensor(
                stack("filled"), dtype=torch.float32, device=device
            ),
        }
