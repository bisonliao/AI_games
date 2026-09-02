"""Memory-bounded prioritized replay for uint8 Atari frame stacks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from DQN.messages import TransitionChunk


@dataclass(slots=True)
class ReplaySample:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    weights: np.ndarray
    indices: np.ndarray


class PrioritizedReplayBuffer:
    """Proportional prioritized replay backed by sum/min segment trees."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        *,
        alpha: float,
        priority_epsilon: float,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self.observations = np.empty(
            (capacity, *observation_shape), dtype=np.uint8
        )
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)

        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2
        self.sum_tree = np.zeros(2 * self.tree_capacity, dtype=np.float64)
        self.min_tree = np.full(2 * self.tree_capacity, np.inf, dtype=np.float64)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def add(self, chunk: TransitionChunk) -> None:
        count = len(chunk)
        if count <= 0:
            return
        if count > self.capacity:
            start = count - self.capacity
            chunk = TransitionChunk(
                actor_id=chunk.actor_id,
                observations=chunk.observations[start:],
                actions=chunk.actions[start:],
                rewards=chunk.rewards[start:],
                next_observations=chunk.next_observations[start:],
                terminated=chunk.terminated[start:],
                epsilon=chunk.epsilon,
                policy_version=chunk.policy_version,
            )
            count = self.capacity

        indices = (np.arange(count) + self.position) % self.capacity
        self.observations[indices] = chunk.observations
        self.actions[indices] = chunk.actions
        self.rewards[indices] = chunk.rewards
        self.next_observations[indices] = chunk.next_observations
        self.terminated[indices] = chunk.terminated
        self._set_priorities(
            indices, np.full(count, self.max_priority, dtype=np.float64)
        )
        self.position = int((self.position + count) % self.capacity)
        self.size = min(self.size + count, self.capacity)

    def sample(
        self, batch_size: int, beta: float, rng: np.random.Generator
    ) -> ReplaySample:
        if self.size < batch_size:
            raise ValueError("not enough replay entries for the requested batch")
        total_priority = self.sum_tree[1]
        if total_priority <= 0:
            raise RuntimeError("replay priority sum must be positive")

        segment = total_priority / batch_size
        masses = (np.arange(batch_size) + rng.random(batch_size)) * segment
        indices = np.fromiter(
            (self._find_prefix_sum_index(mass) for mass in masses),
            dtype=np.int64,
            count=batch_size,
        )
        scaled_priorities = self.sum_tree[indices + self.tree_capacity]
        probabilities = scaled_priorities / total_priority
        minimum_probability = self.min_tree[1] / total_priority
        maximum_weight = (self.size * minimum_probability) ** (-beta)
        weights = (self.size * probabilities) ** (-beta) / maximum_weight

        return ReplaySample(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            terminated=self.terminated[indices],
            weights=weights.astype(np.float32),
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float64)
        if np.any(~np.isfinite(priorities)) or np.any(priorities <= 0):
            raise ValueError("priorities must be finite and positive")
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self._set_priorities(indices, priorities)

    def _set_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        scaled = np.power(priorities + self.priority_epsilon, self.alpha)
        for index, value in zip(indices, scaled, strict=True):
            tree_index = int(index) + self.tree_capacity
            self.sum_tree[tree_index] = value
            self.min_tree[tree_index] = value
            tree_index //= 2
            while tree_index >= 1:
                self.sum_tree[tree_index] = (
                    self.sum_tree[2 * tree_index]
                    + self.sum_tree[2 * tree_index + 1]
                )
                self.min_tree[tree_index] = min(
                    self.min_tree[2 * tree_index],
                    self.min_tree[2 * tree_index + 1],
                )
                tree_index //= 2

    def _find_prefix_sum_index(self, mass: float) -> int:
        index = 1
        while index < self.tree_capacity:
            left = 2 * index
            if mass <= self.sum_tree[left]:
                index = left
            else:
                mass -= self.sum_tree[left]
                index = left + 1
        replay_index = index - self.tree_capacity
        return min(replay_index, self.size - 1)
