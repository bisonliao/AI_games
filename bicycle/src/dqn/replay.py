"""Uniform fixed-capacity replay buffer used by the DQN learner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .nstep import NStepTransition


@dataclass(slots=True)
class ReplayBatch:
    """Uniform NumPy minibatch copied out of the replay ring."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    discounts: np.ndarray


class ReplayBuffer:
    """Fixed-size ring buffer with uniform random minibatch sampling."""

    def __init__(self, capacity: int, observation_shape: tuple[int, ...]) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.observation_shape = observation_shape
        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: NStepTransition) -> None:
        """Insert one transition, overwriting the oldest item when full."""
        index = self._position
        self.observations[index] = transition.observation
        self.actions[index] = transition.action
        self.rewards[index] = transition.reward
        self.next_observations[index] = transition.next_observation
        self.discounts[index] = transition.discount
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def extend(self, transitions: Iterable[NStepTransition]) -> int:
        count = 0
        for transition in transitions:
            self.add(transition)
            count += 1
        return count

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        """Sample a uniform minibatch without replacement."""
        if self._size < batch_size:
            raise ValueError("not enough transitions to sample")
        indices = rng.choice(self._size, size=batch_size, replace=False)
        return ReplayBatch(
            observations=self.observations[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_observations=self.next_observations[indices].copy(),
            discounts=self.discounts[indices].copy(),
        )

    def state_dict(self) -> dict[str, object]:
        size = self._size
        use_full_arrays = size == self.capacity
        return {
            "capacity": self.capacity,
            "observation_shape": self.observation_shape,
            "observations": self.observations.copy()
            if use_full_arrays
            else self.observations[:size].copy(),
            "next_observations": self.next_observations.copy()
            if use_full_arrays
            else self.next_observations[:size].copy(),
            "actions": self.actions.copy() if use_full_arrays else self.actions[:size].copy(),
            "rewards": self.rewards.copy() if use_full_arrays else self.rewards[:size].copy(),
            "discounts": self.discounts.copy()
            if use_full_arrays
            else self.discounts[:size].copy(),
            "position": self._position,
            "size": self._size,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("replay capacity mismatch")
        size = int(state["size"])
        self.observations[:size] = np.asarray(state["observations"])
        self.next_observations[:size] = np.asarray(state["next_observations"])
        self.actions[:size] = np.asarray(state["actions"])
        self.rewards[:size] = np.asarray(state["rewards"])
        self.discounts[:size] = np.asarray(state["discounts"])
        self._position = int(state["position"])
        self._size = size
