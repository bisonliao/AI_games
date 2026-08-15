"""Compact preallocated replay buffer."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TransitionBatch:
    obs: dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: dict[str, np.ndarray]
    terminated: np.ndarray


def concatenate_transition_batches(batches: list[TransitionBatch]) -> TransitionBatch:
    """Concatenate transition batches in the supplied order."""
    if not batches:
        raise ValueError("cannot concatenate an empty transition batch list")
    return TransitionBatch(
        obs={
            key: np.concatenate([batch.obs[key] for batch in batches], axis=0)
            for key in batches[0].obs
        },
        actions=np.concatenate([batch.actions for batch in batches], axis=0),
        rewards=np.concatenate([batch.rewards for batch in batches], axis=0),
        next_obs={
            key: np.concatenate([batch.next_obs[key] for batch in batches], axis=0)
            for key in batches[0].next_obs
        },
        terminated=np.concatenate([batch.terminated for batch in batches], axis=0),
    )


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._size = 0
        self._position = 0
        self._obs: dict[str, np.ndarray] | None = None
        self._next_obs: dict[str, np.ndarray] | None = None
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminated = np.empty(capacity, dtype=np.bool_)

    def __len__(self) -> int:
        return self._size

    def _allocate(self, obs: dict[str, np.ndarray]) -> None:
        self._obs = {k: np.empty((self.capacity, *v.shape), dtype=v.dtype) for k, v in obs.items()}
        self._next_obs = {k: np.empty((self.capacity, *v.shape), dtype=v.dtype) for k, v in obs.items()}

    def add_batch(self, batch: TransitionBatch) -> None:
        n = len(batch.actions)
        if n == 0:
            return
        if self._obs is None:
            self._allocate({k: np.asarray(v[0]) for k, v in batch.obs.items()})
        assert self._obs is not None and self._next_obs is not None
        for i in range(n):
            idx = self._position
            for key in self._obs:
                self._obs[key][idx] = batch.obs[key][i]
                self._next_obs[key][idx] = batch.next_obs[key][i]
            self._actions[idx] = batch.actions[i]
            self._rewards[idx] = batch.rewards[i]
            self._terminated[idx] = batch.terminated[i]
            self._position = (idx + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        if self._size < batch_size:
            raise ValueError("not enough transitions")
        indices = self.rng.integers(0, self._size, size=batch_size)
        assert self._obs is not None and self._next_obs is not None
        return TransitionBatch(
            obs={k: v[indices].copy() for k, v in self._obs.items()},
            actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_obs={k: v[indices].copy() for k, v in self._next_obs.items()},
            terminated=self._terminated[indices].copy(),
        )
