from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


class ReplayBuffer:
    """Fixed-size NumPy replay buffer owned exclusively by the learner."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int = 1,
        action_dim: int = 1,
        *,
        observation_shape: Sequence[int] | None = None,
        observation_dtype: np.dtype | type[np.generic] = np.float32,
        store_next_observations: bool = False,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.observation_shape = (
            tuple(int(value) for value in observation_shape)
            if observation_shape is not None
            else (obs_dim,)
        )
        self.observation_dtype = np.dtype(observation_dtype)
        self.store_next_observations = store_next_observations
        self.observations = np.empty(
            (capacity, *self.observation_shape), dtype=self.observation_dtype
        )
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_observations = (
            np.empty(
                (capacity, *self.observation_shape), dtype=self.observation_dtype
            )
            if store_next_observations
            else None
        )
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    @property
    def allocated_bytes(self) -> int:
        arrays = [
            self.observations,
            self.actions,
            self.rewards,
            self.terminated,
            self.truncated,
        ]
        if self.next_observations is not None:
            arrays.append(self.next_observations)
        return sum(array.nbytes for array in arrays)

    def add_batch(self, batch: Mapping[str, np.ndarray]) -> int:
        observations = np.asarray(
            batch["observations"], dtype=self.observation_dtype
        )
        count = len(observations)
        if count == 0:
            return 0
        if count > self.capacity:
            start = count - self.capacity
            batch = {key: np.asarray(value)[start:] for key, value in batch.items()}
            observations = np.asarray(
                batch["observations"], dtype=self.observation_dtype
            )
            count = self.capacity

        indices = (np.arange(count) + self._position) % self.capacity
        self.observations[indices] = observations
        self.actions[indices] = batch["actions"]
        self.rewards[indices] = batch["rewards"]
        if self.next_observations is not None:
            self.next_observations[indices] = batch["next_observations"]
        self.terminated[indices] = batch["terminated"]
        self.truncated[indices] = batch["truncated"]
        self._position = int((self._position + count) % self.capacity)
        self._size = min(self._size + count, self.capacity)
        return count

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self._size < batch_size:
            raise ValueError(f"Need {batch_size} samples, replay has {self._size}")
        indices = self._rng.integers(0, self._size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            # The bandit learner never reads next_observations. When storage is
            # disabled, returning the terminal observation keeps a conventional
            # transition-shaped batch without duplicating large pixel tensors.
            "next_observations": (
                self.next_observations[indices]
                if self.next_observations is not None
                else self.observations[indices]
            ),
            "terminated": self.terminated[indices],
            "truncated": self.truncated[indices],
        }
